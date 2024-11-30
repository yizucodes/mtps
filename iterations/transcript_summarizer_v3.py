import language_tool_python
from deepmultilingualpunctuation import PunctuationModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
import os
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize

# Install necessary packages if not already installed
# You might need to run these commands in your environment:
# pip install nltk
# pip install transformers
# pip install language-tool-python
# pip install deepmultilingualpunctuation

# Uncomment the lines below if you need to download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Import necessary modules from transformers and other libraries

# Initialize the punctuation restoration model
punctuation_model = PunctuationModel()

# Initialize the grammar correction tool
tool = language_tool_python.LanguageTool('en-US')

# Define helper functions


def extract_whisper_transcription(text):
    """
    Extracts the Whisper transcription text from a document.

    Args:
        text (str): The input text containing the transcription

    Returns:
        str: The extracted Whisper transcription text, or empty string if not found
    """
    # Look for the marker
    marker = "Whisper transcription:"

    # Find the position of the marker
    marker_pos = text.find(marker)

    # If marker is not found, return empty string
    if marker_pos == -1:
        return ""

    # Get the text after the marker
    # Add len(marker) to start from the end of the marker
    transcription = text[marker_pos + len(marker):].strip()

    return transcription


def remove_repetitions(text):
    """
    Removes immediate word and phrase repetitions in the text.
    """
    # Remove immediate word repetitions (e.g., "the the")
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    # Remove immediate phrase repetitions (e.g., "back then back then")
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
    import unicodedata

    # 1. Remove non-speech artifacts (e.g., [laughter], (applause))
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)

    # 2. Replace typographic apostrophes with standard apostrophes
    text = text.replace('’', "'")

    # 3. Remove filler words and phrases
    filler_words = [
        'um', 'uh', 'you know', 'like', 'i mean', 'hmm', 'ah', 'er', 'uhm',
        'you see', 'basically', 'actually', 'sort of', 'kind of', 'you know what i mean',
        'you know what i\'m saying', 'well', 'so', 'let me see', 'i guess', 'i think'
    ]
    # Use negative lookbehind and lookahead to avoid matching parts of words
    filler_pattern = r'(?<!\w)(' + '|'.join(filler_words) + r')(?!\w)'
    text = re.sub(filler_pattern, '', text, flags=re.IGNORECASE)

    # 4. Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # 5. Apply custom corrections for known transcription errors
    custom_corrections = {
        'rect, stalled': 'wrecked, stalled',
        'done up, done for, done in': 'done up, done for, done in',
        'he came in to my he came in to my session': 'he came into my session',
        'back then back then': 'back then',
        'done up, done for, done in, cracked up, counted out': 'done up, done for, done in, cracked up, counted out',
        'see also hurt, useless, hurt, useless, and weak': 'see also hurt, useless, and weak',
        # Add more as identified
    }
    for wrong, right in custom_corrections.items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)

    # 6. Remove repetitions
    text = remove_repetitions(text)

    # 7. Restore punctuation
    text = punctuation_model.restore_punctuation(text)

    # 8. Correct grammar
    text = correct_grammar(text)

    # 9. Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # 10. Final cleanup
    # Ensures that the text does not contain irregular spacing, which can occur after removing words or characters in previous preprocessing steps
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def split_text_into_chunks(text, tokenizer, max_tokens=1024):
    """
    Splits the text into chunks that do not exceed the maximum token limit.

    Args:
        text (str): The input text to split.
        tokenizer (Tokenizer): The tokenizer used to count tokens.
        max_tokens (int): The maximum number of tokens per chunk.

    Returns:
        List[str]: A list of text chunks.
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


def summarize_text(processed_text, summarizer, tokenizer, max_tokens=1024, default_max_length=150, default_min_length=50):
    """
    Summarizes text with improved summarization logic and length control.
    """
    chunks = split_text_into_chunks(processed_text, tokenizer, max_tokens)
    if not chunks:
        raise ValueError("No valid chunks generated from input text")

    # First pass: Summarize each chunk independently
    chunk_summaries = []
    errors = []

    for i, chunk in enumerate(chunks, 1):
        try:
            # Use more aggressive summarization parameters
            summary = summarizer(
                chunk,
                max_length=default_max_length,  # Reduced from 350
                min_length=default_min_length,  # Reduced from 100
                do_sample=True,  # Enable sampling for more varied output
                num_beams=4,
                length_penalty=2.0,  # Encourage shorter summaries
                early_stopping=True
            )
            if summary and len(summary) > 0:
                chunk_summaries.append(summary[0]['summary_text'].strip())
        except Exception as e:
            errors.append(f"Error processing chunk {i}: {str(e)}")
            continue

    if not chunk_summaries:
        error_log = "\n".join(errors)
        raise ValueError(
            f"Failed to generate any valid summaries. Errors:\n{error_log}")

    # Combine chunk summaries
    intermediate_summary = " ".join(chunk_summaries)

    # Second pass: Further summarize if still too long
    if len(tokenizer.encode(intermediate_summary)) > max_tokens:
        try:
            final_summary = summarizer(
                intermediate_summary,
                max_length=default_max_length,
                min_length=default_min_length,
                do_sample=True,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            if final_summary and len(final_summary) > 0:
                return final_summary[0]['summary_text'].strip()
        except Exception as e:
            print(f"Warning: Second pass failed: {str(e)}")
            # Fall back to first pass summary
            return intermediate_summary

    return intermediate_summary


def process_chunk(chunk, summarizer, tokenizer, chunk_num, total_chunks, max_length=150, min_length=50):
    """
    Process a single chunk with improved summarization parameters.
    """
    try:
        # Validate chunk
        if not chunk or not chunk.strip():
            return False, None, "Empty chunk received"

        # Get token count for dynamic length adjustment
        tokens = tokenizer.encode(chunk, add_special_tokens=False)
        input_length = len(tokens)

        # More aggressive length adjustments
        adjusted_max_length = min(max_length, max(
            30, input_length // 3))  # More reduction
        adjusted_min_length = min(min_length, max(
            10, input_length // 6))  # More reduction

        summary = summarizer(
            chunk,
            max_length=adjusted_max_length,
            min_length=adjusted_min_length,
            do_sample=True,  # Enable sampling
            num_beams=4,
            length_penalty=2.0,  # Encourage shorter summaries
            early_stopping=True,
            repetition_penalty=1.5  # Reduce repetition
        )

        if summary and len(summary) > 0:
            return True, summary[0]['summary_text'].strip(), None

        return False, None, f"Failed to generate valid summary for chunk {chunk_num}"

    except Exception as e:
        return False, None, f"Error processing chunk {chunk_num}: {str(e)}"


def save_summary(summary, original_file_path):
    """
    Saves the summary to a new file in the 'summarized_texts_v2' directory.

    Args:
        summary (str): The summary text to save.
        original_file_path (str): The path to the original transcription file.
    """
    # Create the output directory "summarized_texts" if it doesn't exist
    output_dir = os.path.join(os.path.dirname(
        original_file_path), 'summarized_texts_v3')
    os.makedirs(output_dir, exist_ok=True)

    # Generate the output file name
    base_name = os.path.splitext(os.path.basename(original_file_path))[0]
    output_file_name = f"{base_name}_summarized.txt"
    output_file_path = os.path.join(output_dir, output_file_name)

    # Write the summary to the file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(summary)

    print(f"Summary saved to: {output_file_path}")


def post_process_summary(summary_text):
    """
    Postprocesses the summarized text by fixing known errors, correcting grammar,
    and performing general cleanup.

    Args:
        summary_text (str): The input summarized text.

    Returns:
        str: The cleaned and corrected text.
    """
    # Step 1: Apply custom corrections
    custom_corrections = {
        "Sara Pescara": "Amy Mullins",
        "Cable of Bones": "fibula bones",
        "someone-me-": "someone like me",
        "It no longer has our natural childlike curiosity": "It no longer fosters natural childlike curiosity",
        # Add more corrections as needed
    }

    for wrong, correct in custom_corrections.items():
        summary_text = re.sub(re.escape(wrong), correct,
                              summary_text, flags=re.IGNORECASE)

    # Step 2: Correct grammar and punctuation
    matches = tool.check(summary_text)
    summary_text = language_tool_python.utils.correct(summary_text, matches)

    # Step 3: Final cleanup (e.g., removing extra spaces, fixing typos)
    summary_text = re.sub(r'\s+', ' ', summary_text).strip()

    return summary_text


# Main execution flow
if __name__ == "__main__":
    # Load the transcription file
    file_path = 'data/transcription_test_JamesCameron_982s.txt'

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Extract the Whisper transcription
    whisper_text = extract_whisper_transcription(text)

    # Check if transcription was found
    if not whisper_text:
        print("Whisper transcription not found in the file.")
    else:
        # Apply preprocessing to clean the text
        processed_text = preprocess_transcription(whisper_text)

        # At this point, 'processed_text' is ready for summarization
        # For testing, you can print the processed text
        print("Processed Text:\n", processed_text)

        # Load BART tokenizer and model
        print("\nLoading BART model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn")
        summarizer = pipeline(
            "summarization", model=model, tokenizer=tokenizer)
        print("BART model and tokenizer loaded successfully.")

        # Summarize the processed text
        print("\nStarting summarization...")
        try:
            summary = summarize_text(processed_text, summarizer, tokenizer)
        except Exception as e:
            print(f"Error during summarization: {e}")
            print("Skipping summarization and moving to the next step.")
            summary = ""

        print("\nRaw Summary:\n", summary)

        # Post-process the summary
        processed_summary = post_process_summary(summary)
        print("\nPost-Processed Summary:\n", processed_summary)

        # Save the post-processed summary to a file
        save_summary(processed_summary, file_path)
