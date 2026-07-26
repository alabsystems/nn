// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Proc-macro crate for nn kernel and model attributes.
//!
//! Provides `#[kernel]` which:
//! - Preserves the original Rust function as a reference implementation
//! - Generates an MSL (Metal Shading Language) source constant
//! - Generates a Kani verification harness (`#[cfg(kani)]`)
//! - Emits differential test coverage metadata
//!
//! Provides `#[model]` which:
//! - Preserves the original Rust function as the executable model reference
//! - Lowers the function body to a `ModelDef` intermediate representation
//! - Validates model IR structure (parameter usage, step ordering)
//! - Emits a metadata module with signature, step, and IR debug info
//!
//! The heavy lifting (IR lowering, codegen) lives in `nn-dsl`. This crate
//! is a thin shim that bridges proc-macro entry points to the library.

mod compile_gen;

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    spanned::Spanned,
};

struct KernelAttrs {
    precision: nn_dsl::PrecisionTier,
    bounds: nn_dsl::InputBounds,
}

struct ModelAttrs {
    verify: bool,
}

impl Default for KernelAttrs {
    fn default() -> Self {
        Self {
            precision: nn_dsl::PrecisionTier::Normal,
            bounds: nn_dsl::InputBounds::new(),
        }
    }
}

impl Parse for KernelAttrs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut attrs = Self::default();
        let mut seen_precision = false;
        let mut seen_bounds = false;

        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;

            if key == "precision" {
                input.parse::<syn::Token![=]>()?;
                if seen_precision {
                    return Err(syn::Error::new(
                        key.span(),
                        "duplicate `precision` kernel attribute",
                    ));
                }
                let value: syn::LitStr = input.parse()?;
                attrs.precision = nn_dsl::PrecisionTier::parse(&value.value()).map_err(|_| {
                    syn::Error::new(
                        value.span(),
                        "unsupported precision tier; expected \"strict\", \"normal\", or \"relaxed\"",
                    )
                })?;
                seen_precision = true;
            } else if key == "bounds" {
                if seen_bounds {
                    return Err(syn::Error::new(
                        key.span(),
                        "duplicate `bounds` kernel attribute",
                    ));
                }
                let content;
                syn::parenthesized!(content in input);
                parse_bounds_content(&content, &mut attrs.bounds)?;
                seen_bounds = true;
            } else {
                return Err(syn::Error::new(
                    key.span(),
                    "unsupported kernel attribute key; expected `precision` or `bounds`",
                ));
            }

            if input.is_empty() {
                break;
            }
            input.parse::<syn::Token![,]>()?;
        }

        Ok(attrs)
    }
}

/// Parse the contents of `bounds(x = "-1e4..1e4", alpha = "1e-8..1e3")`.
fn parse_bounds_content(
    input: ParseStream<'_>,
    bounds: &mut nn_dsl::InputBounds,
) -> syn::Result<()> {
    while !input.is_empty() {
        let param_name: syn::Ident = input.parse()?;
        input.parse::<syn::Token![=]>()?;
        let range_str: syn::LitStr = input.parse()?;
        let bound = nn_dsl::InputBound::parse(&range_str.value())
            .map_err(|e| syn::Error::new(range_str.span(), format!("invalid bound: {e}")))?;
        bounds.insert(param_name.to_string(), bound);

        if input.is_empty() {
            break;
        }
        input.parse::<syn::Token![,]>()?;
    }
    Ok(())
}

impl Parse for ModelAttrs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut attrs = Self { verify: false };

        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;

            if key == "verify" {
                attrs.verify = true;
            } else {
                return Err(syn::Error::new(
                    key.span(),
                    "unsupported model attribute key; expected `verify`",
                ));
            }

            if input.is_empty() {
                break;
            }
            input.parse::<syn::Token![,]>()?;
        }

        Ok(attrs)
    }
}

/// Mark a function as an nn kernel.
///
/// The function must use the kernel-safe Rust subset:
/// - Scalar types: `f32`, `f16`
/// - Arithmetic: `+`, `-`, `*`, `/`
/// - Math methods: `.sin()`, `.cos()`, `.sqrt()`, `.rsqrt()`, `.exp()`,
///   `.abs()`, `.recip()`, `.powi(n)`, `.clamp(lo, hi)`, `.max(v)`, `.min(v)`
/// - Reduction intrinsic: `nn_dsl::sum_reduce([a, b, c, ...])`
/// - Let bindings
/// - A single return expression
///
/// # Example
///
/// ```text
/// #[nn_macros::kernel]
/// fn snake(x: f32, alpha: f32) -> f32 {
///     x + (1.0 / alpha) * (alpha * x).sin().powi(2)
/// }
/// ```
///
/// Expands to:
/// - The original `fn snake(...)` (unchanged)
/// - `const SNAKE_MSL: &str = "..."` with the generated Metal shader
/// - `#[cfg(kani)] fn kani_verify_snake()` Kani proof harness
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as KernelAttrs);
    let func = parse_macro_input!(item as syn::ItemFn);

    match expand_kernel(attrs, &func) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Mark a function as an nn model entrypoint.
///
/// `#[model]` lowers the function body into a `ModelDef` IR:
/// - The original function body is preserved and executable.
/// - The body is lowered to model IR and validated (parameter usage, step ordering).
/// - A metadata module is emitted with signature, step, and IR debug info.
#[proc_macro_attribute]
pub fn model(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as ModelAttrs);
    let func = parse_macro_input!(item as syn::ItemFn);

    match expand_model(attrs, &func) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_kernel(attrs: KernelAttrs, func: &syn::ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let kernel_def = nn_dsl::Lowerer::lower_fn(func)
        .map_err(|err| syn::Error::new(func.sig.ident.span(), err.to_string()))?;

    let contract = nn_dsl::PrecisionContract::bootstrap(attrs.precision, kernel_def.return_type);

    let ir_err = |e: nn_dsl::IRError| syn::Error::new(func.sig.ident.span(), e.to_string());
    let msl_source = nn_dsl::emit_msl_with_contract(&kernel_def, contract).map_err(ir_err)?;
    let kani_source = nn_dsl::emit_kani_harness(&kernel_def).map_err(ir_err)?;
    let difftest_source =
        nn_dsl::emit_differential_test_with_bounds(&kernel_def, attrs.precision, &attrs.bounds)
            .map_err(ir_err)?;

    let span = func.sig.ident.span();
    let upper = kernel_def.name.to_uppercase();
    let msl_const_name = syn::Ident::new(&format!("{upper}_MSL"), span);
    let descriptor_const_name = syn::Ident::new(&format!("{upper}_DESCRIPTOR"), span);

    let kani_items = parse_generated_source(func, &kani_source, "kani harness")?;
    let difftest_items = parse_generated_source(func, &difftest_source, "differential test")?;

    let param_count = kernel_def.params.len();
    let entry_point = format!("{}_kernel", kernel_def.name);
    let fast_math = contract.fast_math;
    let meta_tokens = emit_kernel_meta(&kernel_def, attrs.precision, contract, span);

    Ok(quote! {
        #func

        /// Generated MSL source for this kernel.
        pub const #msl_const_name: &str = #msl_source;

        /// Generated kernel descriptor bundling MSL source and metadata.
        ///
        /// Use with `KernelPipeline::from_descriptor` for type-safe dispatch.
        pub const #descriptor_const_name: nn_dsl::KernelDescriptor = nn_dsl::KernelDescriptor::new(
            #msl_const_name,
            #entry_point,
            #param_count,
            #fast_math,
        );

        #(#kani_items)*
        #(#difftest_items)*
        #meta_tokens
    })
}

/// Emit the `#[doc(hidden)]` metadata module for a kernel.
fn emit_kernel_meta(
    kernel_def: &nn_dsl::KernelDef,
    tier: nn_dsl::PrecisionTier,
    contract: nn_dsl::PrecisionContract,
    span: proc_macro2::Span,
) -> proc_macro2::TokenStream {
    let meta_mod_name = syn::Ident::new(&format!("__{}_kernel_meta", kernel_def.name), span);
    let node_count = kernel_def.nodes.len();
    let param_count = kernel_def.params.len();
    let ir_debug = nn_dsl::ir_pretty_print(kernel_def);
    let precision_tier = tier.as_str();
    let fast_math = contract.fast_math;
    let abs_budget = contract.differential_abs_budget;
    let rel_budget = contract.differential_rel_budget;

    quote! {
        #[doc(hidden)]
        pub mod #meta_mod_name {
            pub const NODE_COUNT: usize = #node_count;
            pub const PARAM_COUNT: usize = #param_count;
            pub const IR_DEBUG: &str = #ir_debug;
            pub const PRECISION_TIER: &str = #precision_tier;
            pub const FAST_MATH: bool = #fast_math;
            pub const DIFFERENTIAL_ABS_BUDGET: f32 = #abs_budget;
            pub const DIFFERENTIAL_REL_BUDGET: f32 = #rel_budget;
        }
    }
}

#[allow(deprecated)] // ModelDef IR is deprecated; proc macro still uses it for metadata emission
fn expand_model(attrs: ModelAttrs, func: &syn::ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let model_name = func.sig.ident.to_string();
    let meta_mod_name = syn::Ident::new(
        &format!("__{model_name}_model_meta"),
        func.sig.ident.span(),
    );

    let mut input_names = Vec::with_capacity(func.sig.inputs.len());
    let mut input_types = Vec::with_capacity(func.sig.inputs.len());
    let mut input_ranks: Vec<proc_macro2::TokenStream> = Vec::with_capacity(func.sig.inputs.len());
    for input in &func.sig.inputs {
        match input {
            syn::FnArg::Receiver(receiver) => {
                return Err(syn::Error::new(
                    receiver.span(),
                    "#[model] can only be applied to free functions",
                ));
            }
            syn::FnArg::Typed(pat_ty) => {
                match &*pat_ty.pat {
                    syn::Pat::Ident(pat_ident) => input_names.push(pat_ident.ident.to_string()),
                    other => {
                        return Err(syn::Error::new(
                            other.span(),
                            "#[model] parameters must be identifier bindings",
                        ));
                    }
                }
                let ty = &pat_ty.ty;
                let ty_str = quote!(#ty).to_string();
                input_types.push(ty_str);
                match extract_tensor_rank(&pat_ty.ty) {
                    Some(rank) => input_ranks.push(quote! { Some(#rank) }),
                    None => input_ranks.push(quote! { None }),
                }
            }
        }
    }

    let input_count = input_names.len();
    let output_type = match &func.sig.output {
        syn::ReturnType::Default => "()".to_string(),
        syn::ReturnType::Type(_, ty) => quote!(#ty).to_string(),
    };
    let output_rank = match &func.sig.output {
        syn::ReturnType::Default => quote! { None },
        syn::ReturnType::Type(_, ty) => match extract_tensor_rank(ty) {
            Some(rank) => quote! { Some(#rank) },
            None => quote! { None },
        },
    };

    // Lower the function body into a ModelDef.
    let model_def = nn_dsl::lower_model_fn(func)
        .map_err(|err| syn::Error::new(func.sig.ident.span(), err.to_string()))?;
    model_def
        .validate()
        .map_err(|err| syn::Error::new(func.sig.ident.span(), err.to_string()))?;

    let step_count = model_def.steps.len();
    let callee_names: Vec<String> = model_def.steps.iter().map(|s| s.callee.clone()).collect();
    let ir_debug = nn_dsl::model_ir_pretty_print(&model_def);
    let model_def_json = serde_json::to_string(&model_def).map_err(|err| {
        syn::Error::new(
            func.sig.ident.span(),
            format!("ModelDef serialization failed: {err}"),
        )
    })?;

    // When `verify` is set, classify callees at proc-macro time (avoids requiring
    // nn_dsl at the expansion site) and emit a structural verification test.
    let verify_test = if attrs.verify {
        let test_fn_name = syn::Ident::new(
            &format!("__test_verify_structure_{model_name}"),
            func.sig.ident.span(),
        );
        let unverifiable: Vec<&str> = callee_names
            .iter()
            .filter(|name| !nn_dsl::classify_callee_name(name).is_verifiable())
            .map(String::as_str)
            .collect();

        if unverifiable.is_empty() {
            quote! {
                #[cfg(test)]
                #[test]
                fn #test_fn_name() {
                    // Structural verification: all callees classified as verifiable
                    // by nn_dsl::classify_callee_name at proc-macro expansion time.
                }
            }
        } else {
            let msg = format!("model uses unverifiable ops: {unverifiable:?}");
            quote! {
                #[cfg(test)]
                #[test]
                fn #test_fn_name() {
                    panic!(#msg);
                }
            }
        }
    } else {
        quote! {}
    };

    let compile_fns = compile_gen::generate_compile_fns(func);
    Ok(quote! {
        #func

        #[doc(hidden)]
        pub mod #meta_mod_name {
            pub const MODEL_NAME: &str = #model_name;
            pub const INPUT_COUNT: usize = #input_count;
            pub const INPUT_NAMES: [&str; #input_count] = [#(#input_names),*];
            pub const INPUT_TYPES: [&str; #input_count] = [#(#input_types),*];
            pub const INPUT_RANKS: [Option<usize>; #input_count] = [#(#input_ranks),*];
            pub const OUTPUT_TYPE: &str = #output_type;
            pub const OUTPUT_RANK: Option<usize> = #output_rank;
            pub const STEP_COUNT: usize = #step_count;
            pub const CALLEE_NAMES: [&str; #step_count] = [#(#callee_names),*];
            pub const IR_DEBUG: &str = #ir_debug;
            pub const MODEL_DEF_JSON: &str = #model_def_json;
        }

        #verify_test
        #compile_fns
    })
}

/// Extract the const-generic rank `D` from a `Tensor<D, ..>` type, if present.
///
/// Recognizes the pattern: path segment named "Tensor" with a leading const
/// generic argument that is a literal integer. Returns `None` for non-tensor types.
fn extract_tensor_rank(ty: &syn::Type) -> Option<usize> {
    let path = match ty {
        syn::Type::Path(tp) => &tp.path,
        _ => return None,
    };
    let last = path.segments.last()?;
    if last.ident != "Tensor" {
        return None;
    }
    let args = match &last.arguments {
        syn::PathArguments::AngleBracketed(ab) => &ab.args,
        _ => return None,
    };
    let first_arg = args.first()?;
    match first_arg {
        syn::GenericArgument::Const(syn::Expr::Lit(lit)) => match &lit.lit {
            syn::Lit::Int(int_lit) => int_lit.base10_parse::<usize>().ok(),
            _ => None,
        },
        _ => None,
    }
}

fn parse_generated_source(
    func: &syn::ItemFn,
    source: &str,
    label: &str,
) -> syn::Result<Vec<syn::Item>> {
    let parsed: syn::File = syn::parse_str(source).map_err(|err| {
        syn::Error::new(
            func.sig.ident.span(),
            format!("failed to parse generated {label}: {err}"),
        )
    })?;
    Ok(parsed.items)
}
