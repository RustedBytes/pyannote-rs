use crate::session;
use eyre::{bail, eyre, Context, ContextCompat, Result};
use kaldi_native_fbank::online::FeatureComputer;
use kaldi_native_fbank::{FbankComputer, FbankOptions, OnlineFeature};
use ndarray::Array2;
use ort::{session::Session, value::Tensor};
use std::path::Path;

#[derive(Debug)]
pub struct EmbeddingExtractor {
    session: Session,
}

impl EmbeddingExtractor {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = session::create_session(model_path.as_ref())?;
        Ok(Self { session })
    }

    pub fn convert_integer_to_float_audio(samples: &[i16], output: &mut [f32]) {
        for (input, output) in samples.iter().zip(output.iter_mut()) {
            *output = *input as f32 / 32768.0;
        }
    }

    pub fn compute(&mut self, samples: &[i16]) -> Result<impl Iterator<Item = f32>> {
        // Convert to f32 precisely
        let mut samples_f32 = vec![0.0; samples.len()];
        Self::convert_integer_to_float_audio(samples, &mut samples_f32);
        let samples = &samples_f32;

        let mut fbank_opts = FbankOptions::default();
        fbank_opts.mel_opts.num_bins = 80;
        fbank_opts.use_energy = false;

        {
            let frame_opts = &mut fbank_opts.frame_opts;
            frame_opts.dither = 0.0;
            frame_opts.samp_freq = 16000.0;
            frame_opts.snip_edges = true;
        }

        let sample_rate = fbank_opts.frame_opts.samp_freq;
        let fbank = FbankComputer::new(fbank_opts).map_err(|e| eyre!(e))?;
        let mut online_feature = OnlineFeature::new(FeatureComputer::Fbank(fbank));
        online_feature.accept_waveform(sample_rate, samples);
        online_feature.input_finished();

        let frames = online_feature.features;
        if frames.is_empty() {
            bail!("No features computed.");
        }
        let num_bins = frames[0].len();
        let mut flattened = Vec::with_capacity(frames.len() * num_bins);
        for frame in &frames {
            if frame.len() != num_bins {
                bail!("Inconsistent feature dimensions.");
            }
            flattened.extend_from_slice(frame);
        }

        let features = Array2::from_shape_vec((frames.len(), num_bins), flattened)?;
        let mean = features.mean_axis(ndarray::Axis(0)).context("mean")?;
        let features: Array2<f32> = features - mean;
        let features = features.insert_axis(ndarray::Axis(0)); // Add batch dimension
        let inputs = ort::inputs![
        "feats" => Tensor::from_array(features)? // takes ownership of `features`
        ];

        let ort_outs = self.session.run(inputs)?;
        let ort_out = ort_outs
            .get("embs")
            .context("Output tensor not found")?
            .try_extract_tensor::<f32>()
            .context("Failed to extract tensor")?;

        // Collect the tensor data into a Vec to own it
        let embeddings: Vec<f32> = ort_out.1.iter().copied().collect();

        // Return an iterator over the Vec
        Ok(embeddings.into_iter())
    }
}
