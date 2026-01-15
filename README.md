# native-pyannote-rs

Pyannote audio diarization in Rust.

This is a fork of https://github.com/thewh1teagle/pyannote-rs with Rust native crate for audio feature extraction using [kaldi-native-fbank](https://crates.io/crates/kaldi-native-fbank) instead of bindings to C++ variant (knf-rs).

## Features

- Compute 1 hour of audio in less than a minute on CPU.
- Faster performance with DirectML on Windows and CoreML on macOS.
- Accurate timestamps with Pyannote segmentation.
- Identify speakers with wespeaker embeddings.

## Examples

See [examples](examples)

<details>
<summary>How it works</summary>

pyannote-rs uses 2 models for speaker diarization:

1. **Segmentation**: [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) identifies when speech occurs.
2. **Speaker Identification**: [wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) identifies who is speaking.

Inference is powered by [onnxruntime](https://onnxruntime.ai/).

- The segmentation model processes up to 10s of audio, using a sliding window approach (iterating in chunks).
- The embedding model processes filter banks (audio features) extracted with [kaldi-native-fbank](https://crates.io/crates/kaldi-native-fbank).

Speaker comparison (e.g., determining if Alice spoke again) is done using cosine similarity.
</details>

## Credits

Big thanks to [pyannote-onnx](https://github.com/pengzhendong/pyannote-onnx) and [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
