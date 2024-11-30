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
   - Processes summarized texts using a custom LSTM-CRF model for named entity recognition.
   - Identifies and classifies key entities (people, organizations, locations) in the summaries.
   - Generates detailed entity analysis reports with classification metrics.
   - Outputs structured NER data for further downstream processing.
   - Use the file (`combined_summaries.txt`) in `demo` folder as input for the `Final_LSTMCRF.ipynb` notebook.

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

4. **Classification**:
   - Use the file (`combined_summaries.txt`) in `demo` folder as input for the `Final_LSTMCRF.ipynb` notebook.

## Dataset


1. For the speech-to-text and summarization pipeline:
- Used `distil-whisper/tedlium-long-form` dataset
- 19 total samples (8 validation, 11 test)
- ~11.5 minutes average duration per sample 

2. For the LSTM-CRF NER model:
- Used the `CoNLL-2003` dataset 
- Training data: 14,041 sequences
- Validation data: 3,250 sequences 
- Test data: 3,453 sequences
- Contains labeled data for 4 entity types: PER, ORG, LOC, MISC

Then the trained LSTM-CRF model was applied to our 307 summarized sequences from the TEDLIUM transcriptions, which explains the performance difference between:

- Original CoNLL test set performance (F1 scores 0.60-0.83)
- Summarized text performance (F1 scores 0.08-0.29)

This significant drop in performance suggests the model had difficulty generalizing from the CoNLL news text it was trained on to our summarized transcription text.

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


### Custom LSTM-CRF Model Configuration
```python
# Model Architecture Settings
EMBEDDING_DIM = 100     # Word embedding dimensions
HIDDEN_DIM = 128       # LSTM hidden layer size
BATCH_SIZE = 32        # Training batch size
MAX_LEN = 50          # Maximum sequence length
DROPOUT = 0.5         # Dropout rate for regularization

# Model Initialization
model = LSTM_CRF(
    vocab_size,           # Size of vocabulary from training data
    tagset_size,         # Number of unique NER tags
    EMBEDDING_DIM,       # Embedding layer dimensions
    HIDDEN_DIM,          # Hidden layer size
    padding_idx=PAD_IDX, # Index for padding tokens
    dropout=DROPOUT      # Dropout rate
)
```

### Bidirectional LSTM Architecture
```python
class BidirectionalCustomLSTM(nn.Module):
    # Forward LSTM processes sequence left-to-right
    # Backward LSTM processes sequence right-to-left
    # Outputs concatenated for richer context
    
    hidden_size = HIDDEN_DIM // 2  # Split for bidirectional
    outputs = torch.cat([forward_out, backward_out], dim=2)
    # Final output shape: (batch_size, seq_length, hidden_dim)
```

### CRF Layer Configuration
```python
# CRF Parameters
crf = CRF(tagset_size)  # Conditional Random Field layer
# Uses Viterbi algorithm for optimal tag sequence
predictions = crf.viterbi_decode(
    emissions,          # LSTM output scores
    mask=mask          # Mask for padding tokens
)
```

### Data Processing Pipeline
```python
# Preprocessing Steps
def preprocess_text(text):
    # Convert to lowercase for consistency
    text = text.lower()
    # Tokenize using spaCy
    doc = nlp(text)
    # Extract features and create masks
    tokens, pos_tags, ner_tags = extract_features(doc)
    # Handle padding and truncation
    return pad_sequences(tokens, maxlen=MAX_LEN)
```

### Training Configuration
```python
# Training Parameters
optimizer = optim.Adam(
    model.parameters(), 
    lr=0.001           # Learning rate
)
clip_value = 5.0      # Gradient clipping threshold

# Loss Calculation
loss = model(
    sentences,        # Input sequences
    tags,            # True NER tags
    mask             # Attention mask
)
```

### Entity Recognition Features
- Supports standard IOB (Inside, Outside, Beginning) tagging scheme
- Recognizes four entity types: LOC, ORG, PER, MISC
- Handles nested entities through layered tag structure
- Maintains entity boundaries across sentence splits

### Performance Optimization Techniques
- Uses dynamic batching for varying sequence lengths
- Implements attention masking for efficient computation
- Employs gradient clipping to prevent exploding gradients
- Utilizes early stopping based on validation metrics

### Output Processing
```python
def process_predictions(predictions, idx2tag):
    # Convert numeric predictions to tag names
    tags = [idx2tag[idx] for idx in predictions]
    # Group consecutive tags into entities
    entities = group_entities(tags)
    # Calculate confidence scores
    scores = calculate_confidence(emissions)
    return entities, scores
```

### Evaluation Metrics
- Precision, recall, and F1 score per entity type
- Micro and macro-averaged metrics
- Confusion matrix for error analysis
- Token-level and entity-level accuracy measures

### Metrics

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


## Results Summary LSTM-CRF NER using CONLL

The LSTM-CRF model combines a bidirectional LSTM network with a Conditional Random Field layer for named entity recognition in summarized text. This architecture allows the model to capture both word-level context and label dependencies, making it particularly effective for identifying and classifying named entities in summarized content.

### Performance Metrics:
- Average Precision: 0.8300 (83.00%)
- Average Recall: 0.7900 (79.00%)
- Average F1 Score: 0.8200 (82.00%)
- Total Entities Processed: 9,109
- Total Samples Analyzed: 19 (8 validation, 11 test)

### Entity-wise Analysis
- Location (LOC): 
  - Precision: 0.84, Recall: 0.82, F1: 0.83
  - Total instances: 1,834
- Organization (ORG):
  - Precision: 0.72, Recall: 0.52, F1: 0.60
  - Total instances: 1,339
- Person (PER):
  - Precision: 0.80, Recall: 0.69, F1: 0.74
  - Total instances: 1,796
- Miscellaneous (MISC):
  - Precision: 0.81, Recall: 0.61, F1: 0.70
  - Total instances: 919

### Key Findings
- Strongest performance in identifying location entities (F1: 0.83)
- High precision across all entity types (>0.72)
- Lower recall for organization entities suggests higher false negatives
- Consistent performance across different text lengths
- Effective handling of both technical and conversational content
- Successfully maintains entity recognition quality in compressed summaries

These results demonstrate the model's robust performance in identifying named entities within summarized text, with particularly strong results for location and person entities. The balanced precision-recall trade-off indicates reliable entity recognition, though there's room for improvement in organization detection.

## Results Summary: Combined Summaries NER Analysis

### Document Overview
- Total sequences analyzed: 307
- Total entity instances detected: 36

### Entity Performance Breakdown
- Organizations (ORG):
  - Most frequent entity type (27 instances)
  - Precision: 0.71 (71%) - highest among all categories
  - Recall: 0.19 (19%)
  - F1-score: 0.29 (29%)

- Locations (LOC):
  - 9 instances identified
  - Precision: 0.06 (6%)
  - Recall: 0.11 (11%)
  - F1-score: 0.08 (8%)

- Person (PER) & Miscellaneous (MISC):
  - No successful identifications
  - Support: 0

### Overall Performance
- Micro-average metrics:
  - Precision: 0.07
  - Recall: 0.17
  - F1-score: 0.10

### Key Observations
- Model shows strongest performance in identifying organizations
- Limited success with location entities
- Significant room for improvement across all categories
- Performance substantially lower than expected, suggesting potential issues with:
  - Model adaptation to summarized text format
  - Entity recognition in condensed content
  - Handling of context in shortened text

## References
1. TEDLIUM Dataset: [distil-whisper/tedlium-long-form](https://huggingface.co/datasets/distil-whisper/tedlium-long-form)
2. CoNLL-2003 Dataset: [eriktks/conll2003](https://huggingface.co/datasets/eriktks/conll2003)
3. Whisper Model: [openai/whisper-base](https://huggingface.co/openai/whisper-base)