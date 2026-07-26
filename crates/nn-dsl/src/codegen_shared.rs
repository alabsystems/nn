// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic codegen helpers shared between MSL (nn-dsl) and HIP (nn-cuda).
//!
//! Functions here contain pure algorithmic logic with no GPU syntax dependencies.
//! Each backend calls these with its own type/variable naming conventions.
//!
//! Part of #3338 (cross-backend codegen deduplication).

/// Compute row-major strides for a shape. Returns `None` if any stride product overflows.
pub fn row_major_strides(shape: &[usize]) -> Option<Vec<usize>> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1].checked_mul(shape[i + 1])?;
    }
    Some(strides)
}

/// Compute 1D convolution output length. Returns `None` if the result is non-positive.
pub fn conv_output_len(
    input_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    let effective_kernel = dilation * (kernel_size - 1) + 1;
    let numerator = input_len + 2 * padding;
    if numerator < effective_kernel || stride == 0 {
        return None;
    }
    Some((numerator - effective_kernel) / stride + 1)
}

/// Emit statements for `powi(base, exp)` using binary exponentiation.
///
/// `metal::pow(x, y)` and HIP `powf(x, y)` compute `exp(y * log(x))`,
/// which is undefined for negative `x`. Rust's `powi` uses integer
/// exponentiation, so we expand to multiplications which are correct
/// for all bases.
///
/// For small exponents (≤3), emits a single inline multiplication chain.
/// For larger exponents, uses repeated squaring with O(log n) temporaries:
///
/// - `powi(0)`  → `T(1)`
/// - `powi(1)`  → `b`
/// - `powi(2)`  → `b * b`
/// - `powi(3)`  → `b * b * b`
/// - `powi(8)`  → `p2 = b*b; p4 = p2*p2; p8 = p4*p4;`
/// - `powi(-2)` → `T(1) / (b * b)`
///
/// The `tid` parameter is the node index, used to generate unique temporary
/// variable names that don't collide with other locals.
pub fn powi_stmts(base: &str, exp: i32, ret_type: &str, tid: usize) -> String {
    let abs_exp = exp.unsigned_abs();

    // Special case: powi(0) = 1
    if abs_exp == 0 {
        return format!("    {ret_type} t{tid} = {ret_type}(1);");
    }

    // For small exponents (1-3), use direct inline expansion (no temporaries).
    if abs_exp <= 3 {
        let power = match abs_exp {
            1 => base.to_string(),
            _ => {
                let terms: Vec<&str> = std::iter::repeat_n(base, abs_exp as usize).collect();
                terms.join(" * ")
            }
        };
        return if exp > 0 {
            format!("    {ret_type} t{tid} = {power};")
        } else if abs_exp == 1 {
            format!("    {ret_type} t{tid} = {ret_type}(1) / {power};")
        } else {
            format!("    {ret_type} t{tid} = {ret_type}(1) / ({power});")
        };
    }

    // Binary exponentiation for abs_exp >= 4.
    // Build a sequence of squaring steps and multiply in remaining bits.
    let mut lines = Vec::new();
    let mut n = abs_exp;
    // Track which squared temporaries to multiply together.
    let mut sq_vars: Vec<String> = Vec::new();
    let mut bit_pos = 0u32;
    let base_var = format!("t{tid}_base");
    lines.push(format!("    {ret_type} {base_var} = {base};"));

    // prev holds the name of the current squared value: base, base^2, base^4, ...
    let mut prev = base_var;

    while n > 0 {
        if n & 1 == 1 {
            sq_vars.push(prev.clone());
        }
        n >>= 1;
        if n > 0 {
            bit_pos += 1;
            let next = format!("t{tid}_p{}", 1u64 << bit_pos);
            lines.push(format!("    {ret_type} {next} = {prev} * {prev};"));
            prev = next;
        }
    }

    // Multiply all collected terms for the final result.
    let product = sq_vars.join(" * ");
    if exp > 0 {
        lines.push(format!("    {ret_type} t{tid} = {product};"));
    } else {
        lines.push(format!(
            "    {ret_type} t{tid} = {ret_type}(1) / ({product});"
        ));
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_powi_zero() {
        let s = powi_stmts("x", 0, "float", 5);
        assert_eq!(s, "    float t5 = float(1);");
    }

    #[test]
    fn test_powi_one() {
        let s = powi_stmts("x", 1, "float", 5);
        assert_eq!(s, "    float t5 = x;");
    }

    #[test]
    fn test_powi_neg_one() {
        // abs_exp == 1: no parens around denominator
        let s = powi_stmts("x", -1, "float", 5);
        assert_eq!(s, "    float t5 = float(1) / x;");
    }

    #[test]
    fn test_powi_two() {
        let s = powi_stmts("x", 2, "float", 5);
        assert_eq!(s, "    float t5 = x * x;");
    }

    #[test]
    fn test_powi_neg_two() {
        let s = powi_stmts("x", -2, "float", 5);
        assert_eq!(s, "    float t5 = float(1) / (x * x);");
    }

    #[test]
    fn test_powi_three() {
        let s = powi_stmts("x", 3, "float", 5);
        assert_eq!(s, "    float t5 = x * x * x;");
    }

    #[test]
    fn test_powi_eight_repeated_squaring() {
        let s = powi_stmts("x", 8, "float", 5);
        assert!(s.contains("t5_base"), "should have base variable");
        assert!(s.contains("t5_p2"), "should have p2 squaring step");
        assert!(s.contains("t5_p4"), "should have p4 squaring step");
        assert!(s.contains("t5_p8"), "should have p8 squaring step");
        assert!(s.contains("float t5 = t5_p8;"), "result is p8");
    }

    #[test]
    fn test_powi_neg_eight() {
        let s = powi_stmts("x", -8, "float", 5);
        assert!(s.contains("float(1) / (t5_p8)"));
    }

    #[test]
    fn test_powi_five_mixed_bits() {
        // 5 = 101 in binary → base * p4
        let s = powi_stmts("x", 5, "float", 7);
        assert!(s.contains("t7_base"));
        assert!(s.contains("t7_p2"));
        assert!(s.contains("t7_p4"));
        assert!(s.contains("t7_base * t7_p4"));
    }
}
