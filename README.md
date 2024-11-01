# Speech-to-Text Component with Whisper

This repository contains the implementation of a speech-to-text transcription system using OpenAI's Whisper model, specifically designed to handle long-form audio from the TEDLIUM dataset.

## Project Overview

The system processes long-form audio by:
1. Using Whisper base model for transcription
2. Implementing chunking strategy to handle Whisper's 30-second input limitation
3. Evaluating transcription quality using Word Error Rate (WER)

## Repository Structure
```
mtps/
├── whisper_transcriber.py     # Main transcription script
├── wer_calculator.py          # WER calculation script
└── transcriptions_*/          # Generated transcriptions and reports
    ├── transcription_*.txt    # Individual transcription files
    ├── summary_report.txt     # Processing summary
    └── wer_report.txt         # WER analysis report
```

## Key Features

### 1. Audio Processing
- Chunks long audio into 30-second segments with 2-second overlap
- Handles both validation and test splits from TEDLIUM dataset
- Processes audio using Whisper base model

### 2. Chunking Strategy
```python
CHUNK_LENGTH_SEC = 30  # Standard Whisper input size
OVERLAP_SEC = 2       # Overlap for context continuity
```
- Maintains context between chunks
- Prevents word splitting at boundaries
- Ensures smooth transcription flow

### Chunking Overlap Rationale
The 2-second overlap is designed to capture complete phrases (4-6 words) at chunk boundaries, based on average English speaking rates of 2-3 words per second. This duration aligns with natural speech patterns including sentence transitions and pauses (0.5-3 seconds), ensuring smooth context preservation between consecutive 30-second chunks.

### 3. WER Calculation
- Implemented both using jiwer library and from scratch
- Current average WER: 0.3240 (32.4%)
- Best performing sample: 0.2644 (26.44%)
- Worst performing sample: 0.4016 (40.16%)

## Dataset

Using `distil-whisper/tedlium-long-form` dataset:
- Validation split: 8 samples
- Test split: 11 samples
- Total processed: 19 samples
- Average duration: ~11.5 minutes per sample

## Requirements

```bash
# Core dependencies
pip install torch transformers datasets

# For WER calculation
pip install jiwer
```

## Usage


1. Activate Anaconda environment:
```bash
conda activate whisper_env
```
   
2. Run transcription:
```bash
python whisper_transcriber.py
```

3. Calculate WER:
```bash
python wer_calculator.py
```

## Results Summary

Performance metrics:
- Average WER: 0.3240
- Processing both validation and test splits
- Total samples processed: 19
- Consistent performance across different speakers
- WER range: 0.2644 - 0.4016

## Technical Details

### Whisper Configuration
```python
# Model settings
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")      # Load base model (~244MB)
processor = WhisperProcessor.from_pretrained("openai/whisper-base")                 # Load processor which:
                                                                                    # 1. Feature extractor: Converts audio to spectrograms (audio fingerprints)
                                                                                    # 2. Tokenizer: Converts between text and token IDs (e.g., "hello" → [123, 456])
                                                                                    # Both are needed to translate audio into text the model can understand

# Generation parameters
predicted_ids = model.generate(
    input_features,                # Processed audio input
    language="en",                # Force English language output
    num_beams=5,                  # Beam search with 5 parallel beams for better transcription quality 
    no_repeat_ngram_size=3        # Prevent repetition of 3-word phrases (e.g., "the the the" or "going to going to")
)
```
### Hardware
- Running on M2 MacBook Air
- Using MPS (Metal Performance Shaders) for acceleration
  - MPS (Metal Performance Shaders) provides native GPU acceleration for Apple Silicon (M1/M2) chips, serving as an alternative to NVIDIA's CUDA for deep learning tasks. The implementation leverages Apple's Neural Engine to accelerate neural network operations, resulting in faster processing compared to CPU-only execution while maintaining energy efficiency on MacBooks.

## Current Status and Next Steps

Completed:
- ✅ Basic transcription pipeline
- ✅ Chunking implementation
- ✅ WER calculation
- ✅ Initial performance evaluation


## References
1. TEDLIUM Dataset: [distil-whisper/tedlium-long-form](https://huggingface.co/datasets/distil-whisper/tedlium-long-form)
2. Whisper Model: [openai/whisper-base](https://huggingface.co/openai/whisper-base)