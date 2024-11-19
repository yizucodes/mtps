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
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# Import necessary modules from transformers and other libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from deepmultilingualpunctuation import PunctuationModel
import language_tool_python

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
    # 1. Remove non-speech artifacts (e.g., [laughter], (applause))
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    
    # 2. Remove filler words and phrases
    filler_words = [
        'um', 'uh', 'you know', 'like', 'i mean', 'hmm', 'ah', 'er', 'uhm',
        'you see', 'basically', 'actually', 'sort of', 'kind of', 'you know what i mean',
        'you know what i\'m saying', 'well', 'so', 'let me see', 'i guess', 'i think'
    ]
    # Create a regex pattern to match filler words
    filler_pattern = r'\b(' + '|'.join(filler_words) + r')\b'
    text = re.sub(filler_pattern, '', text, flags=re.IGNORECASE)
    
    # 3. Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Apply custom corrections for known transcription errors
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
    
    # 5. Remove repetitions
    text = remove_repetitions(text)
    
    # 6. Restore punctuation
    text = punctuation_model.restore_punctuation(text)
    
    # 7. Correct grammar
    text = correct_grammar(text)
    
    # 8. Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # 9. Final cleanup
    # Ensures that the text does not contain irregular spacing, which can occur after removing words or characters in previous preprocessing steps
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Load the transcription file
file_path = 'data/transcription_test_DanBarber_2010_S103_6s.txt'
# 'data/transcription_test_AimeeMullins_1249s.txt'


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
