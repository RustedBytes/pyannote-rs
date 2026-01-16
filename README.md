# native-pyannote-rs

Pyannote audio diarization in Rust.

This is a fork of https://github.com/thewh1teagle/pyannote-rs with Rust native crate for audio feature extraction using [kaldi-native-fbank](https://crates.io/crates/kaldi-native-fbank) instead of bindings to C++ variant (knf-rs).

## Features

- Compute 1 hour of audio in less than a minute on CPU.
- Pure Rust inference with Burn (ndarray backend), no onnxruntime runtime dependency.
- Accurate timestamps with Pyannote segmentation.
- Identify speakers with wespeaker embeddings.

## Examples

See [examples](examples)

## Usage

```rust
use pyannote_rs::{read_wav, EmbeddingExtractor, EmbeddingManager, Segmenter};

fn main() -> anyhow::Result<()> {
    let (samples, sample_rate) = read_wav("audio.wav")?;
    let mut segmenter = Segmenter::new("src/nn/segmentation/model.bpk")?;
    let mut extractor = EmbeddingExtractor::new("src/nn/speaker_identification/model.bpk")?;
    let mut speakers = EmbeddingManager::new(4);

    for segment in segmenter.iter_segments(&samples, sample_rate)? {
        let segment = segment?;
        let embedding = extractor.extract(&segment.samples, sample_rate)?;
        let speaker_id = speakers.upsert(&embedding, 0.5).unwrap_or(0);

        println!("{:.2}-{:.2} => speaker {}", segment.start, segment.end, speaker_id);
    }

    Ok(())
}
```

<details>
<summary>How it works</summary>

pyannote-rs uses 2 models for speaker diarization:

1. **Segmentation**: [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) identifies when speech occurs.
2. **Speaker Identification**: [wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) identifies who is speaking.

Inference is powered by [Burn](https://burn.dev/) with the ndarray backend using bundled `.bpk` weights.

- The segmentation model processes up to 10s of audio, using a sliding window approach (iterating in chunks).
- The embedding model processes filter banks (audio features) extracted with [kaldi-native-fbank](https://crates.io/crates/kaldi-native-fbank).

Speaker comparison (e.g., determining if Alice spoke again) is done using cosine similarity.
</details>

## Running with Burn on Metal or CUDA

The library ships with the ndarray (CPU) backend by default. To run the models on GPU/Metal, switch the Burn backend and enable the corresponding feature in `Cargo.toml`:

- **Apple Silicon (Metal via WGPU)**: change the backend aliases in `src/nn/mod.rs` to:
  ```rust
  use burn::backend::wgpu::{AutoGraphicsApi, Wgpu, WgpuDevice};

  pub type BurnBackend = Wgpu<f32, AutoGraphicsApi>;
  pub type BurnDevice = WgpuDevice;
  ```
  and enable the WGPU feature: `burn = { version = "~0.20", features = ["wgpu"] }`. WGPU will pick Metal automatically on M1/M2 Macs.
- **NVIDIA GPU (CUDA)**: use the CUDA backend instead:
  ```rust
  use burn::backend::cudarc::{Cuda, CudaDevice};

  pub type BurnBackend = Cuda<f32>;
  pub type BurnDevice = CudaDevice;
  ```
  and enable CUDA JIT in Cargo: `burn = { version = "~0.20", features = ["cuda-jit"] }`. Ensure the NVIDIA driver and CUDA toolkit (12.x) are installed and visible to the build.

After switching the backend and rebuilding dependencies, run any example in release mode to warm up kernels (first call will compile kernels): `cargo run --release --example infinite`.

## Credits

Big thanks to [pyannote-onnx](https://github.com/pengzhendong/pyannote-onnx) and [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
