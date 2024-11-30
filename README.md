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

# For BERTScore calculation
pip install bert_score
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

### Text Summarization Configuration
```python
# Model settings
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")        # Load BART tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")    # Load BART model
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)     # Initialize summarization pipeline

# Generation parameters
summary = summarizer(
    text,                          # Input text chunk
    max_length=150,                # Maximum length of summary
    min_length=50,                 # Minimum length of summary
    do_sample=True,                # Enable sampling for diverse outputs
    num_beams=4,                   # Beam search with 4 parallel beams
    length_penalty=2.0,            # Favor longer summaries
    early_stopping=True            # Stop when valid output is found
)
```

### Chunking Strategy for Summarization
```python
MAX_TOKENS = 900     # Reduced from BART's 1024 limit for safety
MIN_LENGTH = 50      # Minimum summary length per chunk
```
- Splits text using sentence boundaries to maintain coherence
- Tracks token count to prevent exceeding model limits
- Preserves complete sentences within chunks
- Performs two-pass summarization for very long texts

### Preprocessing Pipeline
```python
# Key preprocessing steps
text = remove_non_speech_artifacts(text)      # Remove [brackets], (parentheses)
text = remove_filler_words(text)              # Remove um, uh, you know, etc.
text = remove_repetitions(text)               # Remove immediate word/phrase repetitions
text = restore_punctuation(text)              # Add missing punctuation using deep learning
text = correct_grammar(text)                  # Fix grammatical errors with LanguageTool
```
- Cleans transcription artifacts and noise
- Enhances readability through punctuation restoration
- Improves grammar and removes redundancies
- Prepares text for optimal summarization quality

### Two-Pass Summarization Rationale
The system implements a two-pass approach where long texts are first summarized in chunks, then combined and summarized again if needed. This ensures that even very long transcriptions (e.g., hour-long talks) can be effectively condensed while maintaining coherence and capturing key information across the entire text. The chunk size of 900 tokens provides a safety margin below BART's 1024 token limit while maximizing context available for each summary.

### Optional

Calculate WER:
```bash
python wer_calculator.py
```

## Results Summary for WER

Word Error Rate (WER) is a metric that measures the accuracy of speech recognition systems by calculating the minimum number of word insertions, deletions, and substitutions needed to transform the predicted transcript into the reference transcript, divided by the number of words in the reference. A WER of 0.32 means that approximately 32% of words contain errors compared to the reference transcript, where a lower score indicates better performance (e.g., a WER of 0 would mean perfect transcription).

### Performance metrics:
- Average WER: 0.3240
- Processing both validation and test splits
- Total samples processed: 19
- Consistent performance across different speakers
- WER range: 0.2644 - 0.4016

### WER Calculation
- Implemented both using jiwer library and from scratch
- Current average WER: 0.3240 (32.4%)
- Best performing sample: 0.2644 (26.44%)
- Worst performing sample: 0.4016 (40.16%)


## Results Summary BERTScore

`BERTScore` is a text evaluation metric that leverages BERT's contextual embeddings to measure semantic similarity between generated text and a reference, allowing it to recognize synonyms and paraphrases rather than requiring exact word matches. Unlike traditional metrics, BERTScore provides more nuanced evaluation scores that better align with human judgment, making it particularly valuable for assessing the quality of text generation tasks like summarization and translation.

### Performance metrics:
- Average Precision: 0.8394 (83.94%)
- Average Recall: 0.8020 (80.20%)
- Average F1 Score: 0.8201 (82.01%)
- Average Compression Ratio: 16.34x
- Total samples processed: 19 (8 validation, 11 test)

### BERTScore Analysis
- Implemented using bert-score library
- Strong semantic similarity maintained despite high compression
- Best performing sample: Brian Cox (P: 0.9126, F1: 0.8546)
- Worst performing sample: James Cameron (P: 0.8015, F1: 0.7882)
- Compression ratios range from 2.91x to 24.27x, with most samples between 15-18x

### Key Findings
- Consistent F1 scores above 0.80 across most speakers
- Higher precision than recall suggests accurate but concise summaries
- Short talks (<5 minutes) tend to achieve higher similarity scores
- Longest compression achieved on James Cameron talk (24.27x) while maintaining 0.79 F1 score
- Exceptional performance on both technical and narrative content types


## References
1. TEDLIUM Dataset: [distil-whisper/tedlium-long-form](https://huggingface.co/datasets/distil-whisper/tedlium-long-form)
2. Whisper Model: [openai/whisper-base](https://huggingface.co/openai/whisper-base)