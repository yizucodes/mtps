
import sys
import language_tool_python
from deepmultilingualpunctuation import PunctuationModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
import os
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
import glob
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize models and tools
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
punctuation_model = PunctuationModel()
tool = language_tool_python.LanguageTool('en-US')


# Enhanced logging setup
def setup_logging():
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Setup console handler with a higher logging level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Setup file handler
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(
            log_dir, f'summarizer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels of logs

    # Remove any existing handlers to avoid duplication
    logger.handlers = []

    # Add our handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


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

# Reduced from 1024 for safety to avoid index out of range errors


def split_text_into_chunks(text, tokenizer, max_tokens=900):
    """
    Splits the text into chunks with a safety margin for token limits.
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


def summarize_text(processed_text, summarizer, tokenizer, max_tokens=900, default_max_length=150, default_min_length=50):
    """
    Summarizes text with improved summarization logic and length control and detailed logging.
    """
    chunks = split_text_into_chunks(processed_text, tokenizer, max_tokens)
    if not chunks:
        raise ValueError("No valid chunks generated from input text")

    logger.info(f"Total number of chunks: {len(chunks)}")
    chunk_summaries = []
    errors = []

    for i, chunk in enumerate(chunks, 1):
        try:
            # Log chunk details
            logger.info(f"\nProcessing chunk {i}/{len(chunks)}")
            logger.info(f"Chunk {i} length: {len(chunk)} characters")
            logger.info(f"First 100 chars of chunk {i}: {chunk[:100]}...")

            # Log tokenization details
            token_length = len(tokenizer.encode(
                chunk, add_special_tokens=False))
            logger.info(f"Token length for chunk {i}: {token_length}")
            logger.info(
                f"Using max_length: {default_max_length}, min_length: {default_min_length}")

            # Attempt summarization
            logger.info(f"Attempting summarization of chunk {i}...")
            summary = summarizer(
                chunk,
                max_length=default_max_length,
                min_length=default_min_length,
                do_sample=True,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

            # Log summarizer output structure
            logger.info(f"Summarizer output type: {type(summary)}")
            logger.info(f"Summarizer output: {str(summary)[:200]}...")

            if summary and len(summary) > 0:
                if isinstance(summary[0], dict) and 'summary_text' in summary[0]:
                    chunk_summaries.append(summary[0]['summary_text'].strip())
                    logger.info(f"Successfully summarized chunk {i}")
                else:
                    logger.warning(f"Invalid summary structure for chunk {i}")
                    logger.warning(
                        f"Summary structure: {summary[0].keys() if isinstance(summary[0], dict) else 'not a dict'}")
            else:
                logger.warning(f"Empty or invalid summary for chunk {i}")

        except IndexError as ie:
            error_msg = f"Index error processing chunk {i}: {str(ie)}"
            logger.error(error_msg)
            logger.error(
                f"Summary content at error: {summary if 'summary' in locals() else 'No summary generated'}")
            errors.append(error_msg)
            continue
        except Exception as e:
            error_msg = f"Error processing chunk {i}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            continue

    if not chunk_summaries:
        error_log = "\n".join(errors)
        raise ValueError(
            f"Failed to generate any valid summaries. Errors:\n{error_log}")

    # Log intermediate results
    logger.info(f"\nSuccessfully generated {len(chunk_summaries)} summaries")
    intermediate_summary = " ".join(chunk_summaries)
    logger.info(
        f"Combined summary length: {len(intermediate_summary)} characters")

    # Second pass if needed
    if len(tokenizer.encode(intermediate_summary)) > max_tokens:
        try:
            logger.info("Performing second round of summarization...")
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
                logger.info("Second pass successful")
                return final_summary[0]['summary_text'].strip()
            else:
                logger.warning(
                    "Invalid final summary output, using first pass summary")
                return intermediate_summary
        except Exception as e:
            logger.error(f"Error in second pass summarization: {str(e)}")
            return intermediate_summary

    return intermediate_summary


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
    Processes a single file through the entire pipeline with enhanced error handling.
    """
    logger.info(f"\nProcessing file: {file_path}")
    errors = []

    try:
        # Read and extract Whisper transcription
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        whisper_text = extract_whisper_transcription(text)
        if not whisper_text:
            logger.warning(f"Whisper transcription not found in {file_path}")
            return False, None, ["No Whisper transcription found"]

        # Preprocess text
        try:
            processed_text = preprocess_transcription(whisper_text)
            logger.info(f"Text preprocessed successfully for {file_path}")
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return False, None, [f"Preprocessing error: {str(e)}"]

        # Summarize text
        try:
            summary = summarize_text(processed_text, summarizer, tokenizer)
            if not summary:
                return False, None, ["Summarization produced empty result"]
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return False, None, [f"Summarization error: {str(e)}"]

        # Post-process summary
        try:
            processed_summary = post_process_summary(summary)
            if not processed_summary:
                return False, None, ["Post-processing produced empty result"]
        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}")
            return False, None, [f"Post-processing error: {str(e)}"]

        # Save summary
        try:
            output_dir = os.path.join(os.path.dirname(
                file_path), 'summarized_texts_demo')
            os.makedirs(output_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file_path = os.path.join(
                output_dir, f"{base_name}_summarized.txt")

            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(processed_summary)

            logger.info(f"Summary saved to: {output_file_path}")
            return True, processed_summary, errors if errors else None

        except Exception as e:
            logger.error(f"Failed to save summary: {str(e)}")
            return False, processed_summary, [f"Save error: {str(e)}"]

    except Exception as e:
        logger.error(f"Critical error processing {file_path}: {str(e)}")
        return False, None, [f"Critical error: {str(e)}"]


def main():
    """
    Enhanced main function with better error handling and reporting.
    """
    try:
        # Initialize logging
        logger = setup_logging()
        logger.info("Starting text summarization process...")

        # Initialize models
        logger.info("Loading BART model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn")
        summarizer = pipeline(
            "summarization", model=model, tokenizer=tokenizer)
        logger.info("✓ Models loaded successfully")

        # Process files
        directory = "./demo"  # Update this path
        txt_files = glob.glob(os.path.join(directory, "*.txt"))

        if not txt_files:
            logger.error(f"✗ No .txt files found in {directory}")
            return

        logger.info(f"Found {len(txt_files)} .txt files to process")

        # Add a visual separator
        logger.info("=" * 50)

        # Initialize results tracking
        results = {
            'successful': [],
            'failed': [],
            'partial': []
        }

        # Process each file with progress indication
        for idx, file_path in enumerate(txt_files, 1):
            logger.info(
                f"\nProcessing file {idx}/{len(txt_files)}: {os.path.basename(file_path)}")
            logger.info("-" * 30)

            success, summary, errors = process_file(
                file_path, summarizer, tokenizer)

            if success and not errors:
                results['successful'].append(file_path)
                logger.info(
                    f"✓ Successfully processed: {os.path.basename(file_path)}")
            elif success and errors:
                results['partial'].append((file_path, errors))
                logger.warning(
                    f"⚠ Partially processed with warnings: {os.path.basename(file_path)}")
            else:
                results['failed'].append((file_path, errors))
                logger.error(
                    f"✗ Failed to process: {os.path.basename(file_path)}")

        # Generate processing report with visual separators
        logger.info("\n" + "=" * 50)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total files processed: {len(txt_files)}")
        logger.info(f"✓ Successful: {len(results['successful'])} files")
        logger.info(f"⚠ Partial: {len(results['partial'])} files")
        logger.info(f"✗ Failed: {len(results['failed'])} files")

        # Save detailed report
        report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(
            directory, f"processing_report_{report_time}.log")

        with open(report_path, 'w', encoding='utf-8') as report:
            report.write("Text Summarization Processing Report\n")
            report.write("=" * 40 + "\n\n")
            report.write(f"Processing Time: {datetime.now()}\n")
            report.write(f"Total Files Processed: {len(txt_files)}\n\n")

            report.write("Successful Files:\n")
            for file in results['successful']:
                report.write(f"✓ {os.path.basename(file)}\n")

            report.write("\nPartially Successful Files:\n")
            for file, errors in results['partial']:
                report.write(f"⚠ {os.path.basename(file)}:\n")
                for error in errors:
                    report.write(f"  - {error}\n")

            report.write("\nFailed Files:\n")
            for file, errors in results['failed']:
                report.write(f"✗ {os.path.basename(file)}:\n")
                for error in errors:
                    report.write(f"  - {error}\n")

        logger.info(f"\nDetailed report saved to: {report_path}")

    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
