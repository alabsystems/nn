// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source export for [`CompiledPlan`] — used by `build.rs` metallib
//! precompilation.

use std::path::Path;

use crate::codegen_msl_tensor_emit::emit_tensor_msl;
use crate::ir::ScalarType;

use super::{CompiledPlan, CompiledStep};

/// A single MSL source entry ready for metallib compilation.
#[derive(Debug)]
pub struct MslSource {
    /// Step index in the plan.
    pub step_index: usize,
    /// Kernel name (used as metallib function lookup key).
    pub kernel_name: String,
    /// Complete MSL source code.
    pub msl: String,
}

impl CompiledPlan {
    /// Generate MSL source code for all Dispatch steps in this plan.
    ///
    /// Returns one `MslSource` per Dispatch step. Non-dispatch steps
    /// (Passthrough, InputForward, IdentityPassthrough) are skipped.
    pub fn generate_msl(
        &self,
        dtype: ScalarType,
    ) -> Result<Vec<MslSource>, crate::codegen_msl_tensor::TensorMSLCodegenError> {
        let mut sources = Vec::new();
        for (i, step) in self.steps.iter().enumerate() {
            if let CompiledStep::Dispatch { kernel, .. } = step {
                let msl = emit_tensor_msl(kernel.def(), dtype)?;
                sources.push(MslSource {
                    step_index: i,
                    kernel_name: kernel.name().to_string(),
                    msl,
                });
            }
        }
        Ok(sources)
    }

    /// Export MSL sources to `.metal` files in the given directory.
    ///
    /// Creates one `.metal` file per Dispatch step, named
    /// `{step_index}_{kernel_name}.metal`. Returns the list of created
    /// file paths.
    pub fn export_msl(
        &self,
        dtype: ScalarType,
        dir: impl AsRef<Path>,
    ) -> Result<Vec<std::path::PathBuf>, ExportMslError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        let sources = self.generate_msl(dtype)?;
        let mut paths = Vec::with_capacity(sources.len());
        for src in &sources {
            let filename = format!("{}_{}.metal", src.step_index, src.kernel_name);
            let path = dir.join(filename);
            std::fs::write(&path, &src.msl)?;
            paths.push(path);
        }
        Ok(paths)
    }
}

/// Errors from MSL export.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExportMslError {
    /// I/O error writing MSL files.
    #[error("MSL export I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// MSL codegen error.
    #[error("MSL codegen error: {0}")]
    Codegen(#[from] crate::codegen_msl_tensor::TensorMSLCodegenError),
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::ir::{IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType};
    use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
    use crate::trace_compile::{CompiledKernel, CompiledPlan, CompiledStep};

    /// Build a plan with 1 Dispatch + 3 non-Dispatch steps.
    fn make_mixed_plan() -> CompiledPlan {
        let relu_kernel = KernelDef::new(
            "relu",
            vec![Param::new("x", ScalarType::F32)],
            ScalarType::F32,
            vec![
                IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
                IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
                IRNode::new(
                    NodeId::new(2),
                    IRNodeKind::MinMax {
                        op: crate::ir::MinMaxKind::Max,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        );

        let tensor_def = TensorKernelDef::new(
            "elementwise_relu",
            vec![
                TensorNode::new(
                    TensorNodeId::new(0),
                    TensorOpKind::Input {
                        name: "x".to_string(),
                        shape: vec![2, 3],
                    },
                    vec![2, 3],
                ),
                TensorNode::new(
                    TensorNodeId::new(1),
                    TensorOpKind::Elementwise {
                        kernel: relu_kernel,
                        inputs: vec![TensorNodeId::new(0)],
                    },
                    vec![2, 3],
                ),
            ],
            TensorNodeId::new(1),
        );

        CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::Dispatch {
                    kernel: CompiledKernel::new(tensor_def),
                    weight_data: HashMap::new(),
                    external_node_ids: None,
                },
                CompiledStep::Passthrough {
                    op_name: "reshape".to_string(),
                    output_shape: vec![6],
                },
                CompiledStep::IdentityPassthrough,
            ],
            input_shapes: vec![vec![2, 3]],
            output_step: 1,
            weight_names: vec![],
        }
    }

    #[test]
    fn test_generate_msl_skips_non_dispatch_steps() {
        let plan = make_mixed_plan();
        let sources = plan.generate_msl(ScalarType::F32).unwrap();

        // Only the single Dispatch step (index 1) should produce MSL.
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].step_index, 1);
        assert_eq!(sources[0].kernel_name, "elementwise_relu");
        assert!(!sources[0].msl.is_empty());
    }

    #[test]
    fn test_generate_msl_contains_kernel_function() {
        let plan = make_mixed_plan();
        let sources = plan.generate_msl(ScalarType::F32).unwrap();

        // MSL should contain a kernel function declaration (Metal attribute syntax).
        assert!(sources[0].msl.contains("[[kernel]]"));
    }

    #[test]
    fn test_generate_msl_empty_plan() {
        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::IdentityPassthrough,
            ],
            input_shapes: vec![vec![1]],
            output_step: 0,
            weight_names: vec![],
        };
        let sources = plan.generate_msl(ScalarType::F32).unwrap();
        assert!(sources.is_empty());
    }

    #[test]
    fn test_export_msl_creates_files() {
        let plan = make_mixed_plan();
        let dir = std::env::temp_dir().join(format!("nn_msl_export_test_{}", std::process::id()));

        let paths = plan.export_msl(ScalarType::F32, &dir).unwrap();

        assert_eq!(paths.len(), 1);
        assert!(paths[0].exists());
        assert!(paths[0]
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .contains("elementwise_relu"));

        // Verify file content is non-empty MSL (Metal attribute syntax).
        let content = std::fs::read_to_string(&paths[0]).unwrap();
        assert!(content.contains("[[kernel]]"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
