import language_tool_python
from deepmultilingualpunctuation import PunctuationModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
import os
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

punctuation_model = PunctuationModel()
tool = language_tool_python.LanguageTool('en-US')


def extract_whisper_transcription(text):
    marker = "Whisper transcription:"
    marker_pos = text.find(marker)
    return text[marker_pos + len(marker):].strip() if marker_pos != -1 else text.strip()


def remove_repetitions(text):
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\b(\w+ \w+)( \1\b)+', r'\1', text)
    return text


def correct_grammar(text):
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)


def preprocess_transcription(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = text.replace('’', "'")

    filler_words = [
        'um', 'uh', 'you know', 'like', 'i mean', 'hmm', 'ah', 'er', 'uhm',
        'you see', 'basically', 'actually', 'sort of', 'kind of', 'you know what i mean',
        'you know what i\'m saying', 'well', 'so', 'let me see', 'i guess', 'i think'
    ]
    filler_pattern = r'(?<!\w)(' + '|'.join(filler_words) + r')(?!\w)'
    text = re.sub(filler_pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()

    custom_corrections = {
        'rect, stalled': 'wrecked, stalled',
        'done up, done for, done in': 'done up, done for, done in',
        'he came in to my he came in to my session': 'he came into my session',
        'back then back then': 'back then',
    }
    for wrong, right in custom_corrections.items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)

    text = remove_repetitions(text)
    text = punctuation_model.restore_punctuation(text)
    text = correct_grammar(text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def split_text_into_chunks(text, tokenizer, max_tokens=1024):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if current_length + len(sentence_tokens) > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = len(sentence_tokens)
        else:
            current_chunk += " " + sentence
            current_length += len(sentence_tokens)

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def create_structured_prompt(text):
    """Enhanced prompt for narrative coherence"""
    return (
        "Synthesize the key points into a flowing narrative that: "
        "1) Shows how events and experiences connect "
        "2) Explains how different interests influenced each other "
        "3) Presents ideas in a logical sequence "
        "Focus on creating clear connections between ideas and avoid fragmented statements. "
        "Text: " + text
    )


def summarize_text(processed_text, summarizer, tokenizer, max_tokens=1024):
    try:
        prompted_text = create_structured_prompt(processed_text)
        chunks = split_text_into_chunks(prompted_text, tokenizer, max_tokens)

        if not chunks:
            raise ValueError("No valid chunks generated")

        # First pass: Extract key themes
        chunk_summaries = []
        for chunk in chunks:
            if len(chunk.split()) < 30:
                continue

            try:
                summary = summarizer(
                    chunk,
                    max_length=120,          # Shorter for initial pass
                    min_length=40,
                    do_sample=False,         # Disable sampling for more consistent output
                    num_beams=8,             # Increased for better selection
                    length_penalty=5.0,      # Stronger penalty
                    early_stopping=True,
                    repetition_penalty=3.0    # Increased to avoid repetition
                )
                if summary and summary[0]['summary_text'].strip():
                    chunk_summaries.append(summary[0]['summary_text'].strip())
            except Exception as e:
                print(f"Chunk processing error: {str(e)}")
                continue

        if not chunk_summaries:
            raise ValueError("No valid summaries generated")

        intermediate = " ".join(chunk_summaries)

        # Final pass: Create coherent narrative
        final_summary = summarizer(
            intermediate,
            max_length=180,
            min_length=120,
            do_sample=False,
            num_beams=8,
            length_penalty=5.0,
            early_stopping=True,
            repetition_penalty=3.0
        )

        return final_summary[0]['summary_text'].strip() if final_summary else intermediate

    except Exception as e:
        print(f"Summarization error: {str(e)}")
        return ""


def post_process_summary(summary_text):
    """Enhanced post-processing for better flow"""
    # Remove quote marks while preserving content
    summary_text = re.sub(r'"([^"]*)"', r'\1', summary_text)
    summary_text = re.sub(r"'([^']*)'", r'\1', summary_text)

    # Add connecting words
    summary_text = re.sub(r'(?<=[.!?])\s+(?=[A-Z])',
                          '. This led to ', summary_text)

    # Fix common issues
    custom_corrections = {
        # Remove repeated transitions
        r"\b(this|that|these|those)\s+led\s+to\s+(?=(?:this|that|these|those)\s+led\s+to)": "",
        r"\bi\b": "I",
        r"(?<=\.\s)i\b": "I",
        r"\s+,\s*": ", ",
        r"\s+\.\s*": ". "
    }

    for pattern, replacement in custom_corrections.items():
        summary_text = re.sub(pattern, replacement, summary_text)

    # Fix grammar
    matches = tool.check(summary_text)
    summary_text = language_tool_python.utils.correct(summary_text, matches)

    # Clean up
    summary_text = re.sub(r'\s+', ' ', summary_text).strip()
    summary_text = re.sub(r'(?<=[.!?])\s+', ' ', summary_text)

    return summary_text


def post_process_summary(summary_text):
    """Enhanced post-processing to reduce direct quotes"""
    # Remove quote marks while preserving content
    summary_text = re.sub(r'"([^"]*)"', r'\1', summary_text)
    summary_text = re.sub(r"'([^']*)'", r'\1', summary_text)

    # Fix common issues
    custom_corrections = {
        "DJ VU": "déjà vu",
        "Danny Boyle": "James Cameron",
        "Danny": "James Cameron",
        "Boyle": "Cameron",
        r"\bi\b": "I",  # Fix standalone 'i'
        r"(?<=\.\s)i\b": "I",  # Fix 'i' at start of sentence
    }

    for wrong, correct in custom_corrections.items():
        summary_text = re.sub(re.escape(wrong) if not wrong.startswith(r"\b") else wrong,
                              correct, summary_text, flags=re.IGNORECASE)

    # Fix grammar and punctuation
    matches = tool.check(summary_text)
    summary_text = language_tool_python.utils.correct(summary_text, matches)

    # Clean up spacing and format
    summary_text = re.sub(r'\s+', ' ', summary_text).strip()

    return summary_text


def save_summary(summary, file_path):
    output_dir = os.path.join(os.path.dirname(
        file_path), 'summarized_texts_v3')
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_summarized.txt")

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(summary)
    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    try:
        file_path = 'data/transcription_test_JamesCameron_982s.txt'

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        whisper_text = extract_whisper_transcription(text)
        if not whisper_text:
            raise ValueError("No valid text found")

        processed_text = preprocess_transcription(whisper_text)
        if not processed_text:
            raise ValueError("Preprocessing resulted in empty text")

        print("\nLoading BART model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn")
        summarizer = pipeline(
            "summarization", model=model, tokenizer=tokenizer)
        print("Models loaded successfully")

        print("\nGenerating summary...")
        summary = summarize_text(processed_text, summarizer, tokenizer)
        if not summary:
            raise ValueError("Summary generation failed")

        processed_summary = post_process_summary(summary)
        print("\nFinal Summary:\n", processed_summary)
        save_summary(processed_summary, file_path)

    except Exception as e:
        print(f"Error: {str(e)}")
