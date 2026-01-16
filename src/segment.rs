use crate::nn::{self, BurnBackend, BurnDevice};
use anyhow::{Context, Result, anyhow, bail};
use burn::tensor::{Tensor, TensorData};
use std::{cmp::Ordering, collections::VecDeque, path::Path};

#[derive(Debug, Clone)]
#[repr(C)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub samples: Vec<i16>,
}

#[derive(Debug)]
pub struct Segmenter {
    model: nn::segmentation::Model<BurnBackend>,
    device: BurnDevice,
}

impl Segmenter {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let device = BurnDevice::default();
        let model_path = model_path
            .as_ref()
            .to_str()
            .context("Model path must be valid UTF-8")?;
        let model = nn::segmentation::Model::from_file(model_path, &device);

        Ok(Self { model, device })
    }

    pub fn iter_segments<'a>(
        &'a mut self,
        samples: &'a [i16],
        sample_rate: u32,
    ) -> Result<impl Iterator<Item = Result<Segment>> + 'a> {
        if sample_rate == 0 {
            bail!("sample_rate cannot be zero");
        }
        if samples.is_empty() {
            bail!("samples cannot be empty");
        }

        let frame_size = 270;
        let frame_start = 721;
        let window_size = (sample_rate * 10) as usize; // 10 seconds
        let mut is_speeching = false;
        let mut offset = frame_start;
        let mut start_offset = 0.0;

        let padded_samples = pad_to_window(samples, window_size);

        let mut start_iter = (0..padded_samples.len()).step_by(window_size);
        let model = &self.model;
        let device = self.device.clone();
        let mut segments_queue = VecDeque::new();

        Ok(std::iter::from_fn(move || {
            if let Some(start) = start_iter.next() {
                let end = (start + window_size).min(padded_samples.len());
                let window = &padded_samples[start..end];

                let mut window_f32 = Vec::with_capacity(window.len());
                window_f32.extend(window.iter().map(|&x| x as f32));

                let data = TensorData::new(window_f32, [1, 1, window.len()]);
                let input = Tensor::<BurnBackend, 3>::from_data(data, &device);
                let output = model.forward(input);
                let output_data = output.into_data();
                let shape = output_data.shape.clone();

                if shape.len() != 3 {
                    return Some(Err(anyhow!(
                        "Unexpected segmentation output shape: {:?}",
                        shape
                    )));
                }

                let batch = shape[0];
                let frames = shape[1];
                let classes = shape[2];

                if classes == 0 {
                    return Some(Err(anyhow!("Segmentation model returned zero classes")));
                }

                let values = match output_data.into_vec::<f32>() {
                    Ok(values) => values,
                    Err(err) => {
                        return Some(Err(anyhow!("Failed to read model output: {err}")));
                    }
                };

                for batch_index in 0..batch {
                    let batch_offset = batch_index * frames * classes;
                    for frame_index in 0..frames {
                        let start_idx = batch_offset + frame_index * classes;
                        let class_scores = &values[start_idx..start_idx + classes];

                        let max_index = match find_max_index(class_scores) {
                            Ok(index) => index,
                            Err(e) => return Some(Err(e)),
                        };

                        if max_index != 0 {
                            if !is_speeching {
                                start_offset = offset as f64;
                                is_speeching = true;
                            }
                        } else if is_speeching {
                            let start = start_offset / sample_rate as f64;
                            let end = offset as f64 / sample_rate as f64;

                            let start_f64 = start * (sample_rate as f64);
                            let end_f64 = end * (sample_rate as f64);

                            // Ensure indices are within bounds
                            let start_idx = start_f64.min((samples.len() - 1) as f64) as usize;
                            let end_idx = end_f64.min(samples.len() as f64) as usize;

                            let segment_samples = &padded_samples[start_idx..end_idx];

                            is_speeching = false;

                            let segment = Segment {
                                start,
                                end,
                                samples: segment_samples.to_vec(),
                            };
                            segments_queue.push_back(segment);
                        }
                        offset += frame_size;
                    }
                }
            }
            segments_queue.pop_front().map(Ok)
        }))
    }

    pub fn segments(&mut self, samples: &[i16], sample_rate: u32) -> Result<Vec<Segment>> {
        self.iter_segments(samples, sample_rate)?.collect()
    }

    pub fn segments_f32(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<Segment>> {
        let converted = f32_to_i16(samples);
        self.segments(&converted, sample_rate)
    }
}

fn pad_to_window(samples: &[i16], window_size: usize) -> Vec<i16> {
    if window_size == 0 {
        return samples.to_vec();
    }
    let remainder = samples.len() % window_size;
    if remainder == 0 {
        return samples.to_vec();
    }

    let pad = window_size - remainder;
    let mut padded = Vec::with_capacity(samples.len() + pad);
    padded.extend_from_slice(samples);
    padded.extend(std::iter::repeat(0).take(pad));
    padded
}

fn find_max_index(row: &[f32]) -> Result<usize> {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index)
        .context("row should not be empty")
}

pub fn get_segments<P: AsRef<Path>>(
    samples: &[i16],
    sample_rate: u32,
    model_path: P,
) -> Result<Vec<Segment>> {
    let mut segmenter = Segmenter::new(model_path)?;
    segmenter.segments(samples, sample_rate)
}

pub fn get_segments_f32<P: AsRef<Path>>(
    samples: &[f32],
    sample_rate: u32,
    model_path: P,
) -> Result<Vec<Segment>> {
    let mut segmenter = Segmenter::new(model_path)?;
    segmenter.segments_f32(samples, sample_rate)
}

fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|s| {
            let clipped = s.clamp(-1.0, 1.0);
            (clipped * i16::MAX as f32) as i16
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::pad_to_window;

    #[test]
    fn does_not_add_padding_when_aligned() {
        let samples = vec![1i16; 20];
        let padded = pad_to_window(&samples, 10);
        assert_eq!(padded.len(), 20);
        assert_eq!(padded, samples);
    }

    #[test]
    fn pads_up_to_window_size() {
        let samples = vec![1i16; 15];
        let padded = pad_to_window(&samples, 10);
        assert_eq!(padded.len(), 20);
        assert_eq!(&padded[..15], samples.as_slice());
        assert!(padded[15..].iter().all(|&x| x == 0));
    }
}
