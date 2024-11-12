from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load BART model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Initialize summarization pipeline with BART
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Load your text
file_path = 'data/transcription_test_RobertGupta_340s.txt'
with open(file_path, 'r') as file:
    text = file.read()

# Extract main content (if necessary)
main_text_start = text.find("Original text:") + len("Original text:")
main_content = text[main_text_start:].strip().split("Whisper transcription:")[0]

# Increase chunk size to allow BART more context
max_chunk = 1024  # BART max token size
chunks = [main_content[i:i+max_chunk] for i in range(0, len(main_content), max_chunk)]

# Summarize each chunk with length settings adjusted for BART
summaries = summarizer(chunks, max_length=50, min_length=30, do_sample=False)
full_summary = " ".join([summ['summary_text'] for summ in summaries])

print("Summary:\n", full_summary)
