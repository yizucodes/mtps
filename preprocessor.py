from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
from spellchecker import SpellChecker

def preprocess_text_for_t5(text):
    """
    Preprocess text for T5 summarization based on common patterns in transcriptions.
    Args:
        text (str): Raw text from transcription file
    Returns:
        str: Preprocessed text ready for T5
    """
    # Extract the Whisper transcription if present
    if "Whisper transcription:" in text:
        text = text.split("Whisper transcription:")[1].strip()

    # Remove metadata headers if present
    if "Split:" in text and "Speaker ID:" in text:
        text = text.split("Original text:")[1].strip()

    # Correct known transcription errors
    corrections = {
        'Steve Lopez Junior Acting as Steve Lopez': 'Robert Downey Jr. acting as Steve Lopez',
        'Chakowsky': 'Tchaikovsky',
        'Aldebes': 'Albeniz',
        'S. Pekka Salmonen': 'Esa-Pekka Salonen',
        'air you die it': 'erudite',
        'air your dith': 'erudite',
        'Julliard train musician': 'Juilliard-trained musician',
        'Jullyard': 'Juilliard',
        'Nathaniels\'': 'Nathaniel\'s',
        'Walt Disney concert hall': 'Walt Disney Concert Hall',
        # Add other corrections as needed
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)

    # Initialize spell checker
    spell = SpellChecker()

    # Tokenize text into words
    words = text.split()
    corrected_words = []
    for word in words:
        # Check if the word is misspelled
        if word.lower() in spell.unknown([word]):
            # Correct the word
            corrected_word = spell.correction(word)
            # If correction is None, keep the original word
            if corrected_word is not None:
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    text = ' '.join(corrected_words)

    # Remove filler words and phrases
    filler_patterns = [
        r'\buh\b', r'\bum\b', r'\ber\b', r'\bhmm\b', r'\byou know\b',
        r'\band stuff\b', r'\bkind of\b', r'\bsort of\b',
        r'\bI mean\b', r'\blike\b', r'\bso\b', r'\bwell\b', r'\bokay\b'
    ]
    for pattern in filler_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Remove repeated phrases
    text = re.sub(r'\b(\w+\s+)\1{1,}\b', r'\1', text, flags=re.IGNORECASE)

    # Fix common transcription errors
    text = text.replace(" i ", " I ")
    text = text.replace(" i'm ", " I'm ")
    text = text.replace(" i've ", " I've ")
    text = text.replace(" i'll ", " I'll ")

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove repeated punctuation
    text = re.sub(r'([.,!?])\1+', r'\1', text)

    # Fix spaces around punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'([.,!?])(\w)', r'\1 \2', text)

    # Remove any remaining transcription artifacts like "[laughter]" or "(applause)"
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\([^)]*\)', '', text)

    # Remove excessive line breaks
    text = re.sub(r'\n+', ' ', text)

    # Final cleanup of extra spaces
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    # Add T5 prefix
    text = "summarize: " + text

    return text

def check_text_length(text, tokenizer):
    """
    Check if text length is within T5's limits
    """
    tokens = tokenizer.encode(text, truncation=False)
    return len(tokens)

def chunk_text(text, tokenizer, max_tokens=500):
    """
    Split text into chunks that are within the model's max token limit.
    Args:
        text (str): The text to split.
        tokenizer: The tokenizer used to tokenize the text.
        max_tokens (int): The maximum number of tokens per chunk.
    Returns:
        List[str]: A list of text chunks.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ''
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        sentence_length = len(tokenizer.encode(sentence, truncation=False))

        if current_length + sentence_length <= max_tokens:
            current_chunk += ' ' + sentence
            current_length += sentence_length
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = sentence_length

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Preprocess text
raw_text = """One day Los Angeles Times columnist Steve Lopez was walking along the streets of downtown Los Angeles when he heard beautiful music. And the source was a man, an African-American man, charming, rugged, homeless, playing a violin that only had two strings. And I'm telling a story that many of you know, because Steve's columns became the basis for a book which was turned into a movie with Robert Downey Jr. acting as Steve Lopez and Jamie Foxx as Nathaniel Anthony Ayers. The Juilliard-trained double bassist whose promising career was cut short by a tragic affliction with paranoid schizophrenia. Nathaniel dropped out of Juilliard. He suffered a complete breakdown and 30 years later he was living homeless on the streets of Skid Row in downtown Los Angeles. I encourage all of you to read Steve's book or to watch the movie to understand not only the beautiful bond that formed between these two men but how music helped shape that bond and ultimately was instrumental, if you'll pardon the pun, in helping Nathaniel get off the streets. I met Mr. Ayers in 2008, two years ago at Walt Disney Concert Hall. He had just heard a performance of Beethoven's first and fourth symphonies and came backstage and introduced himself, speaking in a very jovial and gregarious way about Yo-Yo Ma and Hillary Clinton and how the Dodgers were never gonna make the World Series all because of the treacherous first violin passage work in the last movement of Beethoven's fourth symphony. And we got talking about music, and I got an email from Steve a few days later saying that Nathaniel was interested in a violin lesson with me. Now, I should mention Nathaniel refuses treatment, because when he was treated, it was with shock therapy and Thorazine and handcuffs, and that scar has stayed with him for his entire life. But as a result, now he is prone to these schizophrenic episodes, the worst of which can manifest themselves as him exploding and then disappearing for days, wandering the streets of Skid Row exposed to its horrors with the torment of his own mind unleashed upon him. And Nathaniel was in such a state of agitation when we started our first lesson at Walt Disney Concert Hall. He had a manic glint in his eyes. He was lost, and he was talking about invisible demons and smoke and how someone was poisoning him in his sleep. And I was afraid, not for myself, but that I was gonna lose him. That he was gonna sink into one of his states and that I would ruin his relationship with the violin if I started talking about scales and arpeggios and other exciting forms of didactic violin pedagogy. So I just started playing. And I played the first movement of the Beethoven violin concerto. And as I played, I understood that there was a profound change occurring in Nathaniel's eyes. It was as if he was in the grip of some invisible pharmaceutical, a chemical reaction for which my playing, the music, was its catalyst. And Nathaniel's manic rage was transformed into understanding, acquired curiosity, and grace. And in a miracle, he lifted his own violin, and he started playing by ear, certain snippets of violin concertos, which he then asked me to complete: Mendelssohn, Tchaikovsky, Sibelius. And we started talking about music from Bach to Beethoven and Brahms, Bruckner, Albeniz, from Bartok all the way up to Esa-Pekka Salonen. And I understood that he not only had an encyclopedic knowledge of music, but he related to this music at a personal level. He spoke about it with a kind of passion and understanding that I share with my colleagues in the Los Angeles Philharmonic. And through playing music and talking about music, this man had transformed. From the paranoid, disturbed man that had just come from walking the streets of downtown Los Angeles to the charming, erudite, brilliant, Juilliard-trained musician. Music is medicine. Music changes us. And for Nathaniel, music is sanity. Because music allows him to take his thoughts and delusions and shape them through his imagination and his creativity into reality. And that is an escape from his tormented state. And I understood that this was the very essence of art. This is the very reason why we make music: that we take something that exists within all of us at our very fundamental core, our emotions. And through our artistic lens, through our creativity, we're able to shape those emotions into reality. And the reality of that expression reaches all of us and moves us, inspires and unites us. And for Nathaniel, music brought him back into a fold of friends. And I will always make music with Nathaniel, whether we're at Walt Disney Concert Hall or on Skid Row, because he reminds me why I became a musician. Thank you."""
processed_text = preprocess_text_for_t5(raw_text)

# Print the processed text for verification
print("Processed Text:\n", processed_text)

# Check length
token_count = check_text_length(processed_text, tokenizer)
print(f"Token count: {token_count}")

# Split the text into manageable chunks
chunks = chunk_text(processed_text, tokenizer, max_tokens=500)
print(f"Number of chunks: {len(chunks)}")

# Generate summaries for each chunk
summaries = []
for i, chunk in enumerate(chunks):
    print(f"\nProcessing chunk {i+1}/{len(chunks)}")
    # Tokenize input with truncation to fit model's input limit
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)

    # Generate summary with adjusted parameters for better quality
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=150,       # Adjust as needed
        min_length=60,        # Adjust as needed
        num_beams=5,          # Use more beams for better quality
        length_penalty=1.0,   # Adjust length penalty to favor longer summaries
        no_repeat_ngram_size=3,  # Prevent repeating phrases
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summaries.append(summary)
    print(f"Summary for chunk {i+1}:\n{summary}")

# Combine summaries
combined_summary = ' '.join(summaries)

# Optionally, summarize the combined summary to further condense it
inputs = tokenizer("summarize: " + combined_summary, return_tensors="pt", truncation=True, max_length=512)
summary_ids = model.generate(
    inputs.input_ids,
    max_length=150,
    min_length=60,
    num_beams=5,
    length_penalty=1.0,
    no_repeat_ngram_size=3,
    early_stopping=True
)
final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Save the summarized text
with open("summarized_text.txt", "w") as file:
    file.write(final_summary)

print("\nFinal Summary saved to summarized_text.txt")
print("Generated Final Summary:\n", final_summary)
