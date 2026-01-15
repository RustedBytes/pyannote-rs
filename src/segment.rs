use crate::session;
use eyre::{Context, ContextCompat, Result, bail, eyre};
use ndarray::{ArrayBase, Axis, IxDyn, ViewRepr};
use ort::session::Session;
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
    session: Session,
}

impl Segmenter {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = session::create_session(model_path.as_ref())?;
        Ok(Self { session })
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
        let session = &mut self.session;
        let mut segments_queue = VecDeque::new();

        Ok(std::iter::from_fn(move || {
            if let Some(start) = start_iter.next() {
                let end = (start + window_size).min(padded_samples.len());
                let window = &padded_samples[start..end];

                let array = ndarray::Array1::from_iter(window.iter().map(|&x| x as f32));
                let array = array.view().insert_axis(Axis(0)).insert_axis(Axis(1));

                let tensor = match ort::value::TensorRef::from_array_view(array.into_dyn()) {
                    Ok(tensor) => tensor,
                    Err(e) => return Some(Err(eyre!("Failed to prepare inputs: {:?}", e))),
                };

                let inputs = ort::inputs![tensor];

                let ort_outs = match session.run(inputs) {
                    Ok(outputs) => outputs,
                    Err(e) => return Some(Err(eyre!("Failed to run the session: {:?}", e))),
                };

                let ort_out = match ort_outs.get("output").context("Output tensor not found") {
                    Ok(output) => output,
                    Err(e) => return Some(Err(eyre!("Output tensor error: {:?}", e))),
                };

                let ort_out = match ort_out
                    .try_extract_tensor::<f32>()
                    .context("Failed to extract tensor")
                {
                    Ok(tensor) => tensor,
                    Err(e) => return Some(Err(eyre!("Tensor extraction error: {:?}", e))),
                };

                let (shape, data) = ort_out; // (&Shape, &[f32])
                let shape_slice: Vec<usize> = (0..shape.len()).map(|i| shape[i] as usize).collect();
                let view = match ndarray::ArrayViewD::<f32>::from_shape(
                    ndarray::IxDyn(&shape_slice),
                    data,
                ) {
                    Ok(view) => view,
                    Err(e) => return Some(Err(e.into())),
                };

                for row in view.outer_iter() {
                    for sub_row in row.axis_iter(Axis(0)) {
                        let max_index = match find_max_index(sub_row) {
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

fn find_max_index(row: ArrayBase<ViewRepr<&f32>, IxDyn>) -> Result<usize> {
    let (max_index, _) = row
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.partial_cmp(b.1)
                .context("Comparison error")
                .unwrap_or(Ordering::Equal)
        })
        .context("sub_row should not be empty")?;
    Ok(max_index)
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
