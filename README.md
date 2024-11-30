# Multimodal Text Processing System: An Integrated Approach

This repository implements a pipeline for processing long-form audio and integrating multiple NLP tasks, including speech-to-text transcription, text summarization, and text classification. The system utilizes state-of-the-art transformer models and deep learning techniques for efficient handling of multimodal data.

## Project Overview

The pipeline consists of the following components:

1. **Speech-to-Text Transcription**:
   - Uses OpenAI's Whisper base model for transcription.
   - Processes long-form audio by splitting it into 30-second chunks with a 2-second overlap to ensure continuity in transcription.
   - Supports long audio datasets like the TEDLIUM long-form dataset.
   - Outputs transcriptions for each audio sample in text format.

2. **Text Summarization**:
   - Summarizes transcriptions using Facebook's BART large model (`facebook/bart-large-cnn`).
   - Preprocesses transcription text by:
     - Removing filler words and repetitions.
     - Restoring punctuation using the `DeepMultilingualPunctuation` library.
     - Correcting grammar with `LanguageTool`.
   - Splits long text into manageable chunks for summarization.
   - Outputs summarized text for each transcription.

3. **Pipeline Orchestration**:
   - Automates the transcription and summarization process.
   - Ensures the creation of all intermediate files (e.g., transcription logs, summarized texts).
   - Verifies output integrity at each step to ensure pipeline robustness.

4. **Post-Processing with Notebook**:
   - Use the combined summaries file (`combined_summaries.txt`) in `demo` folder as input for the `Final_LSTMCRF.ipynb` notebook.
   - Perform downstream NLP tasks such as NER or sequence tagging.

## Features

- **Chunking Strategy**: Handles Whisper's input size limitation with a configurable chunking mechanism.
- **Summarization Enhancements**:
  - Handles long-form transcriptions efficiently.
  - Includes post-processing to improve summary readability.
- **Error Handling**: Comprehensive error tracking and reporting at every pipeline stage.

## Workflow

1. **Transcription**:
   - Process audio files using `whisper_transcriber_demo.py`.
   - Outputs transcriptions to a designated directory.

2. **Summarization**:
   - Process transcription files using `batch_text_summarizer_demo.py`.
   - Outputs summarized texts into a separate directory.

3. **Pipeline Execution**:
   - Run the complete pipeline with `run_pipeline.py`.
   - Combines transcription and summarization steps with logging and verification.

## Dataset

Using `distil-whisper/tedlium-long-form` dataset:
- Validation split: 8 samples
- Test split: 11 samples
- Total processed: 19 samples
- Average duration: ~11.5 minutes per sample

## Requirements

```bash
# Core dependencies
pip install librosa soundfile datasets torch transformers numpy tqdm nltk==3.5 deepmultilingualpunctuation language-tool-python

# For WER calculation
pip install jiwer
```

## Usage


1. Create and Activate Anaconda environment:
```bash
conda create -n pipeline python=3.9 -y
conda activate pipeline
```

2. Install 
```bash
conda install -c huggingface datasets transformers -y
conda install pytorch torchvision torchaudio -c pytorch
pip install nltk==3.5 numpy tqdm soundfile librosa deepmultilingualpunctuation language-tool-python
conda install -c conda-forge python-dotenv
```

3. Run pipeline to transcribe speech to text and summarize transcriptions:
```bash
python run_pipeline.py
```

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

### Chunking Strategy for Speech to Text
```python
CHUNK_LENGTH_SEC = 30  # Standard Whisper input size
OVERLAP_SEC = 2       # Overlap for context continuity
```
- Maintains context between chunks
- Prevents word splitting at boundaries
- Ensures smooth transcription flow

### Chunking Overlap Rationale
The 2-second overlap is designed to capture complete phrases (4-6 words) at chunk boundaries, based on average English speaking rates of 2-3 words per second. This duration aligns with natural speech patterns including sentence transitions and pauses (0.5-3 seconds), ensuring smooth context preservation between consecutive 30-second chunks.

### Optional

Calculate WER:
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

### 3. WER Calculation
- Implemented both using jiwer library and from scratch
- Current average WER: 0.3240 (32.4%)
- Best performing sample: 0.2644 (26.44%)
- Worst performing sample: 0.4016 (40.16%)


## References
1. TEDLIUM Dataset: [distil-whisper/tedlium-long-form](https://huggingface.co/datasets/distil-whisper/tedlium-long-form)
2. Whisper Model: [openai/whisper-base](https://huggingface.co/openai/whisper-base)