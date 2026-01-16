use crate::nn::{self, BurnBackend, BurnDevice};
use anyhow::{Context, Result, anyhow, bail};
use burn::tensor::{Tensor, TensorData};
use kaldi_native_fbank::online::FeatureComputer;
use kaldi_native_fbank::{FbankComputer, FbankOptions, OnlineFeature};
use ndarray::{Array1, Array2, s};
use std::path::Path;

const TARGET_FRAME_COUNT: usize = 200;

#[derive(Debug, Clone)]
pub struct Embedding {
    values: Vec<f32>,
}

impl Embedding {
    pub fn new(values: Vec<f32>) -> Self {
        Self { values }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub fn into_inner(self) -> Vec<f32> {
        self.values
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(values: Vec<f32>) -> Self {
        Self::new(values)
    }
}

impl AsRef<[f32]> for Embedding {
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}

#[derive(Debug)]
pub struct EmbeddingExtractor {
    model: nn::speaker_identification::Model<BurnBackend>,
    device: BurnDevice,
}

impl EmbeddingExtractor {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let device = BurnDevice::default();
        let model_path = model_path
            .as_ref()
            .to_str()
            .context("Model path must be valid UTF-8")?;
        let model = nn::speaker_identification::Model::from_file(model_path, &device);

        Ok(Self { model, device })
    }

    pub fn extract(&mut self, samples: &[i16], sample_rate: u32) -> Result<Embedding> {
        let mut samples_f32 = vec![0.0; samples.len()];
        normalize_i16_to_f32(samples, &mut samples_f32);
        self.extract_from_f32(&samples_f32, sample_rate)
    }

    pub fn extract_f32(&mut self, samples: &[f32], sample_rate: u32) -> Result<Embedding> {
        self.extract_from_f32(samples, sample_rate)
    }

    fn extract_from_f32(&mut self, samples: &[f32], sample_rate: u32) -> Result<Embedding> {
        if sample_rate == 0 {
            bail!("sample_rate cannot be zero");
        }
        if samples.is_empty() {
            bail!("samples cannot be empty");
        }

        let sample_rate = sample_rate as f32;
        let mut fbank_opts = FbankOptions::default();
        fbank_opts.mel_opts.num_bins = 80;
        fbank_opts.use_energy = false;

        {
            let frame_opts = &mut fbank_opts.frame_opts;
            frame_opts.dither = 0.0;
            frame_opts.samp_freq = sample_rate;
            frame_opts.snip_edges = true;
        }

        let fbank = FbankComputer::new(fbank_opts).map_err(|e| anyhow!(e))?;
        let mut online_feature = OnlineFeature::new(FeatureComputer::Fbank(fbank));
        online_feature.accept_waveform(sample_rate, samples);
        online_feature.input_finished();

        let frames = online_feature.features;
        if frames.is_empty() {
            bail!("No features computed");
        }

        let num_bins = frames[0].len();
        let mut flattened = Vec::with_capacity(frames.len() * num_bins);
        for frame in &frames {
            if frame.len() != num_bins {
                bail!("Inconsistent feature dimensions");
            }
            flattened.extend_from_slice(frame);
        }

        let features = Array2::from_shape_vec((frames.len(), num_bins), flattened)?;
        let original_mean = features.mean_axis(ndarray::Axis(0)).context("mean")?;
        let features = adjust_feature_length(features, TARGET_FRAME_COUNT, &original_mean);
        let mean = features.mean_axis(ndarray::Axis(0)).context("mean")?;
        let features: Array2<f32> = features - &mean;
        let frame_count = features.nrows();

        let (features, _) = features.into_raw_vec_and_offset();
        let data = TensorData::new(features, [1, frame_count, num_bins]);
        let input = Tensor::<BurnBackend, 3>::from_data(data, &self.device);
        let output = self.model.forward(input);
        let output_data = output.into_data();
        let shape = output_data.shape.clone();

        if shape.len() != 2 {
            bail!("Unexpected embedding output shape: {:?}", shape);
        }
        if shape[0] != 1 {
            bail!("Expected batch size 1, got {}", shape[0]);
        }

        let values = output_data
            .into_vec::<f32>()
            .map_err(|err| anyhow!("Failed to read embedding output: {err}"))?;

        Ok(Embedding::new(values))
    }
}

fn normalize_i16_to_f32(samples: &[i16], output: &mut [f32]) {
    for (input, output) in samples.iter().zip(output.iter_mut()) {
        *output = *input as f32 / 32768.0;
    }
}

fn adjust_feature_length(
    features: Array2<f32>,
    target_frames: usize,
    pad_value: &Array1<f32>,
) -> Array2<f32> {
    let frame_count = features.nrows();
    let num_bins = features.ncols();

    if frame_count > target_frames {
        let start = (frame_count - target_frames) / 2;
        return features.slice(s![start..start + target_frames, ..]).to_owned();
    }

    if frame_count == target_frames {
        return features;
    }

    let mut padded = Array2::zeros((target_frames, num_bins));
    let offset = (target_frames - frame_count) / 2;
    padded
        .slice_mut(s![offset..offset + frame_count, ..])
        .assign(&features);

    fill_with_mean(padded.slice_mut(s![..offset, ..]), pad_value);

    let end_padding = target_frames - offset - frame_count;
    if end_padding > 0 {
        fill_with_mean(
            padded.slice_mut(s![target_frames - end_padding.., ..]),
            pad_value,
        );
    }

    padded
}

fn fill_with_mean(mut view: ndarray::ArrayViewMut2<'_, f32>, mean: &Array1<f32>) {
    for mut row in view.rows_mut() {
        row.assign(mean);
    }
}
