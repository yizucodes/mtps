import language_tool_python
from deepmultilingualpunctuation import PunctuationModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
import os
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
import glob

# Install necessary packages and download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize models and tools
punctuation_model = PunctuationModel()
tool = language_tool_python.LanguageTool('en-US')


def extract_whisper_transcription(text):
    """
    Extracts the Whisper transcription text from a document.
    """
    marker = "Whisper transcription:"
    marker_pos = text.find(marker)

    if marker_pos == -1:
        return ""

    transcription = text[marker_pos + len(marker):].strip()
    return transcription


def remove_repetitions(text):
    """
    Removes immediate word and phrase repetitions in the text.
    """
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\b(\w+ \w+)( \1\b)+', r'\1', text)
    return text


def correct_grammar(text):
    """
    Corrects grammatical errors in the text using LanguageTool.
    """
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text


def preprocess_transcription(text):
    """Preprocess transcription text for summarization."""
    # Remove non-speech artifacts
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)

    # Replace typographic apostrophes
    text = text.replace('’', "'")

    # Remove filler words
    filler_words = [
        'um', 'uh', 'you know', 'like', 'i mean', 'hmm', 'ah', 'er', 'uhm',
        'you see', 'basically', 'actually', 'sort of', 'kind of', 'you know what i mean',
        'you know what i\'m saying', 'well', 'so', 'let me see', 'i guess', 'i think'
    ]
    filler_pattern = r'(?<!\w)(' + '|'.join(filler_words) + r')(?!\w)'
    text = re.sub(filler_pattern, '', text, flags=re.IGNORECASE)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Apply custom corrections
    text = remove_repetitions(text)
    text = punctuation_model.restore_punctuation(text)
    text = correct_grammar(text)

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def split_text_into_chunks(text, tokenizer, max_tokens=1024):
    """
    Splits the text into chunks that do not exceed the maximum token limit.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.encode(
            sentence, add_special_tokens=False))
        if current_length + sentence_length <= max_tokens:
            current_chunk += " " + sentence
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = sentence_length

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def summarize_text(processed_text, summarizer, tokenizer, max_tokens=1024, default_max_length=350, default_min_length=100):
    """
    Summarizes the processed text using the BART summarization model.
    """
    chunks = split_text_into_chunks(processed_text, tokenizer, max_tokens)
    summaries = []

    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")

        input_length = len(tokenizer.encode(chunk, add_special_tokens=False))
        max_length = max(30, min(default_max_length, int(input_length * 0.5)))
        min_length = max(10, min(default_min_length, int(input_length * 0.2)))

        summary = summarizer(chunk, max_length=max_length,
                             min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    combined_summary = " ".join(summaries)

    if len(tokenizer.encode(combined_summary, add_special_tokens=False)) > max_tokens:
        print("Performing second round of summarization...")
        combined_chunks = split_text_into_chunks(
            combined_summary, tokenizer, max_tokens)
        final_summaries = []

        for i, chunk in enumerate(combined_chunks):
            print(
                f"Summarizing combined chunk {i+1}/{len(combined_chunks)}...")
            input_length = len(tokenizer.encode(
                chunk, add_special_tokens=False))
            max_length = max(
                30, min(default_max_length, int(input_length * 0.5)))
            min_length = max(
                10, min(default_min_length, int(input_length * 0.2)))

            summary = summarizer(chunk, max_length=max_length,
                                 min_length=min_length, do_sample=False)
            final_summaries.append(summary[0]['summary_text'])

        final_summary = " ".join(final_summaries)
    else:
        final_summary = combined_summary

    return final_summary


def post_process_summary(summary_text):
    """
    Post-processes the summarized text.
    """
    custom_corrections = {
        "Sara Pescara": "Amy Mullins",
        "Cable of Bones": "fibula bones",
        "someone-me-": "someone like me",
        "It no longer has our natural childlike curiosity": "It no longer fosters natural childlike curiosity",
    }

    for wrong, correct in custom_corrections.items():
        summary_text = re.sub(re.escape(wrong), correct,
                              summary_text, flags=re.IGNORECASE)

    matches = tool.check(summary_text)
    summary_text = language_tool_python.utils.correct(summary_text, matches)
    summary_text = re.sub(r'\s+', ' ', summary_text).strip()

    return summary_text


def process_file(file_path, summarizer, tokenizer):
    """
    Processes a single file through the entire pipeline.
    """
    print(f"\nProcessing file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        whisper_text = extract_whisper_transcription(text)

        if not whisper_text:
            print(f"Warning: Whisper transcription not found in {file_path}")
            return

        processed_text = preprocess_transcription(whisper_text)
        print(f"Text preprocessed successfully for {file_path}")

        summary = summarize_text(processed_text, summarizer, tokenizer)
        processed_summary = post_process_summary(summary)

        # Create output directory and save summary
        output_dir = os.path.join(os.path.dirname(
            file_path), 'summarized_texts_v2')
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file_path = os.path.join(
            output_dir, f"{base_name}_summarized.txt")

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(processed_summary)

        print(f"Summary saved to: {output_file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")


def main():
    # Initialize BART model and tokenizer
    print("Loading BART model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    print("Models loaded successfully")

    # Process all .txt files in the specified directory
    directory = "/Users/yizu/Desktop/CS5100/FinalProject/mtps/data"
    txt_files = glob.glob(os.path.join(directory, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in {directory}")
        return

    print(f"Found {len(txt_files)} .txt files to process")

    for file_path in txt_files:
        process_file(file_path, summarizer, tokenizer)

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
