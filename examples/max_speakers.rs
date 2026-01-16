use anyhow::Result;
use pyannote_rs::{EmbeddingExtractor, EmbeddingManager, Segmenter, read_wav};

fn main() -> Result<()> {
    let audio_path = std::env::args().nth(1).expect("Please specify audio file");
    let (samples, sample_rate) = read_wav(&audio_path)?;
    let max_speakers = 6;

    let mut extractor =
        EmbeddingExtractor::new("src/nn/speaker_identification/model.bpk")?;
    let mut manager = EmbeddingManager::new(max_speakers);
    let mut segmenter = Segmenter::new("src/nn/segmentation/model.bpk")?;

    for segment in segmenter.iter_segments(&samples, sample_rate)? {
        let segment = segment?;
        let embedding = extractor.extract(&segment.samples, sample_rate)?;

        let speaker = if manager.is_full() {
            manager.best_match(&embedding)
        } else {
            manager.upsert(&embedding, 0.5)
        }
        .map(|s| s.to_string())
        .unwrap_or_else(|| "?".into());

        println!(
            "start = {:.2}, end = {:.2}, speaker = {}",
            segment.start, segment.end, speaker
        );
    }

    Ok(())
}
