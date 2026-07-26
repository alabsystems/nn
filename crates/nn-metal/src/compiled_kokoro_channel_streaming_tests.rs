// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`ChannelStreamingSession`] and [`StreamReceiver`].
//!
//! These tests exercise the channel-based streaming API's state machine
//! behavior (construction, cancellation, iterator interface, drop cleanup)
//! without requiring GPU hardware or Kokoro weights.

use super::*;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

/// Helper: create a dummy `DynTensor` of shape `[1, T]` with dtype I64.
fn dummy_input_ids(len: usize) -> DynTensor {
    DynTensor::zeros(&[1, len], DType::I64, &Device::Cpu).unwrap()
}

/// Helper: create a dummy style tensor `[1, 512]`.
fn dummy_style() -> DynTensor {
    DynTensor::zeros(&[1, 512], DType::F32, &Device::Cpu).unwrap()
}

fn default_stream_config() -> KokoroStreamConfig {
    KokoroStreamConfig::new(480).expect("valid config")
}

// -------------------------------------------------------------------
// Construction tests
// -------------------------------------------------------------------

#[test]
fn test_empty_session_construction() {
    let session = ChannelStreamingSession::new(
        Vec::new(),
        dummy_style(),
        1.0,
        default_stream_config(),
    );
    assert_eq!(session.total_chunks(), 0);
    assert_eq!(session.remaining(), 0);
    assert_eq!(session.delivered_count(), 0);
    assert!(!session.is_cancelled());
}

#[test]
fn test_session_with_chunks() {
    let chunks = vec![dummy_input_ids(10), dummy_input_ids(20), dummy_input_ids(30)];
    let session = ChannelStreamingSession::new(
        chunks,
        dummy_style(),
        1.0,
        default_stream_config(),
    );
    assert_eq!(session.total_chunks(), 3);
    assert_eq!(session.remaining(), 3);
    assert_eq!(session.delivered_count(), 0);
}

// -------------------------------------------------------------------
// Receiver tests
// -------------------------------------------------------------------

#[test]
fn test_take_receiver_once() {
    let mut session = ChannelStreamingSession::new(
        Vec::new(),
        dummy_style(),
        1.0,
        default_stream_config(),
    );
    let rx = session.take_receiver();
    assert!(rx.is_some());
    let rx2 = session.take_receiver();
    assert!(rx2.is_none());
}

#[test]
fn test_receiver_initial_state() {
    let mut session = ChannelStreamingSession::new(
        vec![dummy_input_ids(10), dummy_input_ids(20)],
        dummy_style(),
        1.0,
        default_stream_config(),
    );
    let rx = session.take_receiver().unwrap();
    assert_eq!(rx.total_chunks(), 2);
    assert_eq!(rx.remaining(), 2);
    assert_eq!(rx.delivered_count(), 0);
    assert!(!rx.is_done());
}

// -------------------------------------------------------------------
// Cancel tests
// -------------------------------------------------------------------

#[test]
fn test_cancel_session() {
    let session = ChannelStreamingSession::new(
        vec![dummy_input_ids(10)],
        dummy_style(),
        1.0,
        default_stream_config(),
    );
    assert!(!session.is_cancelled());
    session.cancel();
    assert!(session.is_cancelled());
}

#[test]
fn test_cancel_via_receiver() {
    let mut session = ChannelStreamingSession::new(
        vec![dummy_input_ids(10)],
        dummy_style(),
        1.0,
        default_stream_config(),
    );
    let rx = session.take_receiver().unwrap();
    assert!(!session.is_cancelled());
    rx.cancel();
    assert!(session.is_cancelled());
}

// -------------------------------------------------------------------
// StreamChunk enum tests
// -------------------------------------------------------------------

#[test]
fn test_stream_chunk_audio() {
    let chunk = StreamChunk::Audio(vec![0.1, 0.2, 0.3]);
    assert!(matches!(chunk, StreamChunk::Audio(_)));
}

#[test]
fn test_stream_chunk_done() {
    let chunk = StreamChunk::Done;
    assert!(matches!(chunk, StreamChunk::Done));
}

#[test]
fn test_stream_chunk_error() {
    let chunk = StreamChunk::Error("test error".into());
    assert!(matches!(chunk, StreamChunk::Error(_)));
}

#[test]
fn test_stream_chunk_clone() {
    let original = StreamChunk::Audio(vec![1.0, 2.0]);
    let cloned = original.clone();
    if let (StreamChunk::Audio(a), StreamChunk::Audio(b)) = (&original, &cloned) {
        assert_eq!(a, b);
    } else {
        panic!("clone should preserve variant");
    }
}

// -------------------------------------------------------------------
// Channel communication tests (using manual sender)
// -------------------------------------------------------------------

#[test]
fn test_receiver_gets_audio_from_channel() {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let delivered = Arc::new(AtomicUsize::new(0));
    let mut receiver = StreamReceiver {
        inner: rx,
        cancel_flag: cancel,
        chunks_delivered: delivered,
        total_chunks: 2,
        done: false,
    };

    // Send an audio chunk.
    tx.send(StreamChunk::Audio(vec![0.5, -0.5])).unwrap();
    let chunk = receiver.next_chunk();
    assert!(matches!(chunk, Some(StreamChunk::Audio(ref pcm)) if pcm == &[0.5, -0.5]));
    assert!(!receiver.is_done());

    // Send Done.
    tx.send(StreamChunk::Done).unwrap();
    let chunk = receiver.next_chunk();
    assert!(matches!(chunk, Some(StreamChunk::Done)));
    assert!(receiver.is_done());

    // After Done, further calls return None.
    assert!(receiver.next_chunk().is_none());
}

#[test]
fn test_receiver_try_next_empty() {
    let (tx, rx) = mpsc::channel::<StreamChunk>();
    let cancel = Arc::new(AtomicBool::new(false));
    let delivered = Arc::new(AtomicUsize::new(0));
    let mut receiver = StreamReceiver {
        inner: rx,
        cancel_flag: cancel,
        chunks_delivered: delivered,
        total_chunks: 1,
        done: false,
    };

    // No data yet.
    assert!(receiver.try_next_chunk().is_none());
    assert!(!receiver.is_done());

    // Send data, then try again.
    tx.send(StreamChunk::Audio(vec![1.0])).unwrap();
    let chunk = receiver.try_next_chunk();
    assert!(matches!(chunk, Some(StreamChunk::Audio(_))));
}

#[test]
fn test_receiver_disconnected_returns_none() {
    let (tx, rx) = mpsc::channel::<StreamChunk>();
    let cancel = Arc::new(AtomicBool::new(false));
    let delivered = Arc::new(AtomicUsize::new(0));
    let mut receiver = StreamReceiver {
        inner: rx,
        cancel_flag: cancel,
        chunks_delivered: delivered,
        total_chunks: 1,
        done: false,
    };

    // Drop sender.
    drop(tx);

    // Receiver should see disconnection.
    assert!(receiver.next_chunk().is_none());
    assert!(receiver.is_done());
}

#[test]
fn test_receiver_error_marks_done() {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let delivered = Arc::new(AtomicUsize::new(0));
    let mut receiver = StreamReceiver {
        inner: rx,
        cancel_flag: cancel,
        chunks_delivered: delivered,
        total_chunks: 3,
        done: false,
    };

    tx.send(StreamChunk::Error("synthesis failed".into())).unwrap();
    let chunk = receiver.next_chunk();
    assert!(matches!(chunk, Some(StreamChunk::Error(_))));
    assert!(receiver.is_done());
    assert!(receiver.next_chunk().is_none());
}

// -------------------------------------------------------------------
// Iterator interface tests
// -------------------------------------------------------------------

#[test]
fn test_iterator_yields_audio() {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let delivered = Arc::new(AtomicUsize::new(0));
    let mut receiver = StreamReceiver {
        inner: rx,
        cancel_flag: cancel,
        chunks_delivered: delivered,
        total_chunks: 3,
        done: false,
    };

    tx.send(StreamChunk::Audio(vec![1.0, 2.0])).unwrap();
    tx.send(StreamChunk::Audio(vec![3.0, 4.0])).unwrap();
    tx.send(StreamChunk::Done).unwrap();

    let collected: Vec<Vec<f32>> = receiver.by_ref().collect();
    assert_eq!(collected.len(), 2);
    assert_eq!(collected[0], vec![1.0, 2.0]);
    assert_eq!(collected[1], vec![3.0, 4.0]);
    assert!(receiver.is_done());
}

#[test]
fn test_iterator_stops_on_error() {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let delivered = Arc::new(AtomicUsize::new(0));
    let mut receiver = StreamReceiver {
        inner: rx,
        cancel_flag: cancel,
        chunks_delivered: delivered,
        total_chunks: 3,
        done: false,
    };

    tx.send(StreamChunk::Audio(vec![1.0])).unwrap();
    tx.send(StreamChunk::Error("fail".into())).unwrap();
    tx.send(StreamChunk::Audio(vec![2.0])).unwrap(); // should not be yielded

    let collected: Vec<Vec<f32>> = receiver.by_ref().collect();
    assert_eq!(collected.len(), 1);
    assert_eq!(collected[0], vec![1.0]);
}

#[test]
fn test_iterator_stops_on_disconnect() {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let delivered = Arc::new(AtomicUsize::new(0));
    let receiver = StreamReceiver {
        inner: rx,
        cancel_flag: cancel,
        chunks_delivered: delivered,
        total_chunks: 2,
        done: false,
    };

    tx.send(StreamChunk::Audio(vec![1.0])).unwrap();
    drop(tx); // disconnect

    let collected: Vec<Vec<f32>> = receiver.collect();
    assert_eq!(collected.len(), 1);
}

// -------------------------------------------------------------------
// Drop cleanup test
// -------------------------------------------------------------------

#[test]
fn test_drop_session_closes_channel() {
    let mut session = ChannelStreamingSession::new(
        vec![dummy_input_ids(10)],
        dummy_style(),
        1.0,
        default_stream_config(),
    );
    let mut rx = session.take_receiver().unwrap();

    // Drop session (and its sender).
    drop(session);

    // Receiver should see the channel close.
    assert!(rx.next_chunk().is_none());
    assert!(rx.is_done());
}

#[test]
fn test_drop_receiver_allows_session_to_proceed() {
    let mut session = ChannelStreamingSession::new(
        vec![dummy_input_ids(10)],
        dummy_style(),
        1.0,
        default_stream_config(),
    );
    let rx = session.take_receiver().unwrap();

    // Drop receiver.
    drop(rx);

    // Session should still be functional (though drive() would stop
    // when it detects the receiver is gone).
    assert_eq!(session.total_chunks(), 1);
}
