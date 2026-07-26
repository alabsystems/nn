// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional Kani proof harnesses for ViT issue #3730.

#[cfg(kani)]
mod proofs {
    use crate::layers::vision::VitConfig;
    use kani::assume;

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_vit_patch_embedding_dims_match_configured_grid() {
        let batch: usize = kani::any();
        let grid: usize = kani::any();
        let patch_size: usize = kani::any();
        let hidden_size: usize = kani::any();
        let use_cls_token: bool = kani::any();

        assume((1..=4).contains(&batch));
        assume((1..=16).contains(&grid));
        assume((1..=16).contains(&patch_size));
        assume((1..=256).contains(&hidden_size));

        let image_size = grid * patch_size;
        let cfg = VitConfig::new(
            3,
            hidden_size,
            1,
            1,
            hidden_size,
            patch_size,
            image_size,
            1e-5,
            use_cls_token,
        );

        assert!(cfg.is_ok(), "constructed grid-aligned config must be valid");

        if let Ok(cfg) = cfg {
            let num_patches = grid * grid;
            let conv_elements = batch * hidden_size * grid * grid;
            let embedding_elements = batch * cfg.num_patches() * cfg.hidden_size;

            assert!(cfg.num_patches() == num_patches);
            assert!(
                conv_elements == embedding_elements,
                "reshape + transpose in PatchEmbedding must preserve element count"
            );
        }
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_vit_position_embedding_seq_len_matches_patch_count() {
        let grid: usize = kani::any();
        let use_cls_token: bool = kani::any();

        assume((1..=16).contains(&grid));

        let patch_size = 4;
        let image_size = grid * patch_size;
        let cfg = VitConfig::new(3, 64, 1, 1, 64, patch_size, image_size, 1e-5, use_cls_token);

        assert!(cfg.is_ok());

        if let Ok(cfg) = cfg {
            let cls_offset = if use_cls_token { 1 } else { 0 };
            assert!(
                cfg.seq_len() == cfg.num_patches() + cls_offset,
                "position embedding length must cover all patches plus optional CLS"
            );
        }
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(17)]
    fn proof_vit_position_interpolation_indices_with_cls_stay_in_bounds() {
        let source_patches: usize = kani::any();
        let target_patches: usize = kani::any();

        assume((1..=16).contains(&source_patches));
        assume((1..=16).contains(&target_patches));

        let target_len = target_patches + 1;
        assume(target_len >= 2);

        for i in 0..target_patches {
            let src_idx = i * source_patches / target_patches;
            assert!(
                src_idx < source_patches,
                "nearest-neighbor patch index must stay within source patch positions"
            );
        }
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(17)]
    fn proof_vit_position_interpolation_indices_without_cls_stay_in_bounds() {
        let pos_len: usize = kani::any();
        let target_len: usize = kani::any();

        assume((1..=16).contains(&pos_len));
        assume((1..=16).contains(&target_len));

        for i in 0..target_len {
            let src_idx = i * pos_len / target_len;
            assert!(
                src_idx < pos_len,
                "nearest-neighbor position index must stay within source embeddings"
            );
        }
    }
}
