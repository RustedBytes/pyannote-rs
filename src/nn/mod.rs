pub mod segmentation;
pub mod speaker_identification;

use burn::backend::ndarray::{NdArray, NdArrayDevice};

/// Default backend used for inference with Burn models.
pub type BurnBackend = NdArray<f32>;
/// Default device used for inference with Burn models.
pub type BurnDevice = NdArrayDevice;
