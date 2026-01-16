// Models are loaded from src/nn/*/model.bpk.
// wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav
// cargo run --example infinite 6_speakers.wav

use anyhow::Result;
use pyannote_rs::{EmbeddingExtractor, EmbeddingManager, Segmenter};

fn process_segment(
    segment: pyannote_rs::Segment,
    embedding_extractor: &mut EmbeddingExtractor,
    embedding_manager: &mut EmbeddingManager,
    search_threshold: f32,
    sample_rate: u32,
) -> Result<()> {
    let embedding = embedding_extractor.extract(&segment.samples, sample_rate)?;

    let speaker = embedding_manager
        .upsert(&embedding, search_threshold)
        .or_else(|| embedding_manager.best_match(&embedding))
        .map(|r| r.to_string())
        .unwrap_or("?".into());

    println!(
        "start = {:.2}, end = {:.2}, speaker = {}",
        segment.start, segment.end, speaker
    );

    Ok(())
}

fn main() -> Result<()> {
    let audio_path = std::env::args().nth(1).expect("Please specify audio file");
    let search_threshold = 0.5;

    let embedding_model_path = "src/nn/speaker_identification/model.bpk";
    let segmentation_model_path = "src/nn/segmentation/model.bpk";

    let (samples, sample_rate) = pyannote_rs::read_wav(&audio_path)?;
    let mut embedding_extractor = EmbeddingExtractor::new(embedding_model_path)?;
    let mut embedding_manager = EmbeddingManager::new(usize::MAX);
    let mut segmenter = Segmenter::new(segmentation_model_path)?;

    for segment in segmenter.iter_segments(&samples, sample_rate)? {
        match segment {
            Ok(segment) => {
                if let Err(error) = process_segment(
                    segment,
                    &mut embedding_extractor,
                    &mut embedding_manager,
                    search_threshold,
                    sample_rate,
                ) {
                    eprintln!("Error processing segment: {:?}", error);
                }
            }
            Err(error) => eprintln!("Failed to process segment: {:?}", error),
        }
    }

    Ok(())
}
