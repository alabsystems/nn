// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::layers::vision::{Bottleneck, C2f, ConvBnAct, PanNeck, Upsample2d, UpsampleMode};
use crate::layers::{Activation, BatchNorm, Conv2d, Conv2dConfig};
use crate::{DType, Device};

/// Helper: create a ConvBnAct from channel counts without VarBuilder.
fn make_conv_bn(in_c: usize, out_c: usize, k: usize, stride: usize) -> ConvBnAct {
    let padding = k / 2;
    let weight = DynTensor::full(&[out_c, in_c, k, k], 0.01, DType::F32, &Device::Cpu).unwrap();
    let cfg = Conv2dConfig::new(padding, stride, 1);
    let conv = Conv2d::new(weight, None, cfg).unwrap();
    let bn = BatchNorm::new(
        DynTensor::zeros(&[out_c], DType::F32, &Device::Cpu).unwrap(),
        DynTensor::ones(&[out_c], DType::F32, &Device::Cpu).unwrap(),
        Some(DynTensor::ones(&[out_c], DType::F32, &Device::Cpu).unwrap()),
        Some(DynTensor::zeros(&[out_c], DType::F32, &Device::Cpu).unwrap()),
        1e-5,
    )
    .unwrap();
    ConvBnAct::new(conv, bn, Some(Activation::Silu))
}

/// Helper: create a C2f block.
fn make_c2f(in_c: usize, out_c: usize, n: usize) -> C2f {
    let hidden = out_c / 2;
    let cv1 = make_conv_bn(in_c, 2 * hidden, 1, 1);
    let cat_channels = (2 + n) * hidden;
    let cv2 = make_conv_bn(cat_channels, out_c, 1, 1);
    let bottlenecks = (0..n)
        .map(|_| {
            let b_cv1 = make_conv_bn(hidden, hidden, 3, 1);
            let b_cv2 = make_conv_bn(hidden, hidden, 3, 1);
            Bottleneck::new(b_cv1, b_cv2, false)
        })
        .collect();
    C2f::new(cv1, cv2, bottlenecks)
}

/// Helper: build a PAN neck from channel specs.
fn make_pan(c3: usize, c4: usize, c5: usize, n: usize) -> PanNeck {
    let upsample = Upsample2d::new(2.0, 2.0, UpsampleMode::Nearest).unwrap();
    let up1_c2f = make_c2f(c5 + c4, c4, n);
    let up2_c2f = make_c2f(c4 + c3, c3, n);
    let down1_conv = make_conv_bn(c3, c3, 3, 2);
    let down1_c2f = make_c2f(c3 + c4, c4, n);
    let down2_conv = make_conv_bn(c4, c4, 3, 2);
    let down2_c2f = make_c2f(c4 + c5, c5, n);
    PanNeck::new(
        up1_c2f, up2_c2f, upsample, down1_conv, down1_c2f, down2_conv, down2_c2f,
    )
}

#[test]
fn test_pan_neck_output_shapes() {
    let (c3, c4, c5) = (16, 32, 64);
    let pan = make_pan(c3, c4, c5, 1);

    let p3 = DynTensor::full(&[1, 16, 8, 8], 0.5, DType::F32, &Device::Cpu).unwrap();
    let p4 = DynTensor::full(&[1, 32, 4, 4], 0.5, DType::F32, &Device::Cpu).unwrap();
    let p5 = DynTensor::full(&[1, 64, 2, 2], 0.5, DType::F32, &Device::Cpu).unwrap();

    let (n3, n4, n5) = pan.forward_multi(&p3, &p4, &p5).unwrap();
    assert_eq!(n3.dims(), &[1, 16, 8, 8]);
    assert_eq!(n4.dims(), &[1, 32, 4, 4]);
    assert_eq!(n5.dims(), &[1, 64, 2, 2]);
}

#[test]
fn test_pan_neck_batch_size() {
    let (c3, c4, c5) = (16, 32, 64);
    let pan = make_pan(c3, c4, c5, 1);

    let p3 = DynTensor::full(&[2, 16, 8, 8], 0.5, DType::F32, &Device::Cpu).unwrap();
    let p4 = DynTensor::full(&[2, 32, 4, 4], 0.5, DType::F32, &Device::Cpu).unwrap();
    let p5 = DynTensor::full(&[2, 64, 2, 2], 0.5, DType::F32, &Device::Cpu).unwrap();

    let (n3, n4, n5) = pan.forward_multi(&p3, &p4, &p5).unwrap();
    assert_eq!(n3.dims()[0], 2);
    assert_eq!(n4.dims()[0], 2);
    assert_eq!(n5.dims()[0], 2);
}

#[test]
fn test_pan_neck_multi_bottleneck() {
    let (c3, c4, c5) = (32, 64, 128);
    let pan = make_pan(c3, c4, c5, 3);

    let p3 = DynTensor::full(&[1, 32, 16, 16], 0.5, DType::F32, &Device::Cpu).unwrap();
    let p4 = DynTensor::full(&[1, 64, 8, 8], 0.5, DType::F32, &Device::Cpu).unwrap();
    let p5 = DynTensor::full(&[1, 128, 4, 4], 0.5, DType::F32, &Device::Cpu).unwrap();

    let (n3, n4, n5) = pan.forward_multi(&p3, &p4, &p5).unwrap();
    assert_eq!(n3.dims(), &[1, 32, 16, 16]);
    assert_eq!(n4.dims(), &[1, 64, 8, 8]);
    assert_eq!(n5.dims(), &[1, 128, 4, 4]);
}
