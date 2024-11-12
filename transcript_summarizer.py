import re
import nltk
from spellchecker import SpellChecker
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load BART model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Uncomment the line below if you need to download nltk punkt tokenizer
# nltk.download('punkt')
# nltk.download('punkt_tab')


def preprocess_transcription(text):
    """Preprocess transcription text for summarization."""
    # 1. Remove non-speech artifacts
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    # 2. Convert to lowercase
    text = text.lower()
    # 3. Remove filler words and phrases
    filler_words = [
        'um', 'uh', 'you know', 'like', 'i mean', 'so', 'well', 'hmm', 'ah', 'er', 'uhm',
        'you see', 'basically', 'actually', 'sort of', 'kind of', 'you know what i mean',
        'you know what i\'m saying', 'and', 'but'
    ]
    filler_pattern = r'\b(' + '|'.join(filler_words) + r')\b'
    text = re.sub(filler_pattern, '', text)
    # 4. Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    # 5. Spell correction
    spell = SpellChecker()
    corrected_text = []
    words = word_tokenize(text)
    for word in words:
        if word not in string.punctuation:
            corrected_word = spell.correction(word)
            corrected_text.append(corrected_word if corrected_word else word)
        else:
            corrected_text.append(word)
    text = ' '.join(corrected_text)
    # 6. Fix common abbreviations and technical terms
    corrections = {
        'co2': 'CO2',
        'u.s.': 'U.S.',
        'ipcc': 'IPCC',
        'geoengineering': 'geoengineering',
        'terrapower': 'TerraPower',
        'nuclear': 'nuclear',
        'manhattan': 'Manhattan',
        'china': 'China',
        'r&d': 'R&D',
        'bjorn lomborg': 'Bjorn Lomborg'
    }
    for wrong, right in corrections.items():
        text = re.sub(r'\b{}\b'.format(wrong), right, text, flags=re.IGNORECASE)
    # 7. Add proper punctuation and capitalization
    try:
        from deepmultilingualpunctuation import PunctuationModel
        model = PunctuationModel()
        text = model.restore_punctuation(text)
    except ImportError:
        sentences = sent_tokenize(text)
        sentences = [sentence.capitalize() for sentence in sentences]
        text = ' '.join(sentences)
    # 8. Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # 9. Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Test file for summarization
file_path = 'data/transcription_test_AimeeMullins_1249s.txt'
with open(file_path, 'r') as file:
    text = file.read()

# Apply preprocessing to clean the text
processed_text = preprocess_transcription(text)

# Extract main content
main_text_start = processed_text.find("original text:") + len("original text:")
main_content = processed_text[main_text_start:].strip()

# Increase chunk size to allow BART more context
max_chunk = 512  # BART max token size: 1024
chunks = [main_content[i:i+max_chunk] for i in range(0, len(main_content), max_chunk)]

# Summarize each chunk with length settings adjusted for BART
summaries = summarizer(chunks, max_length=50, min_length=30, do_sample=False)
full_summary = " ".join([summ['summary_text'] for summ in summaries])

# Create the output directory "summarized_texts" if it doesn't exist
output_dir = os.path.join(os.path.dirname(file_path), 'summarized_texts')
os.makedirs(output_dir, exist_ok=True)

# Generate the output file path within "summarized_texts"
output_file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_summarized.txt"
output_file_path = os.path.join(output_dir, output_file_name)

# Write the summary to the new file in "summarized_texts"
with open(output_file_path, 'w') as output_file:
    output_file.write(full_summary)

print(f"Summary saved to: {output_file_path}")
