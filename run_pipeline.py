import os
import subprocess
import time
from datetime import datetime
import torch
from lstm_crf import NERModel
import spacy

def run_pipeline():
    print("\n=== Starting Pipeline Execution ===")
    start_time = time.time()
    
    # Create the demo directory if it doesn't exist
    os.makedirs("demo", exist_ok=True)
    
    try:

        # Check if trained model exists
        if not os.path.exists('ner_model.pt'):
            print("\nPre-trained NER model not found. Starting training...")
            subprocess.run(["python", "train.py"], check=True)
            print("✓ Model training completed successfully")
        else:
            print("\nFound pre-trained NER model")

        # Step 1: Run Whisper Transcriber
        print("\n1. Starting Speech-to-Text Transcription...")
        subprocess.run(["python", "whisper_transcriber_demo.py"], check=True)
        print("✓ Transcription completed successfully")
        
        # Check if transcription files were created
        transcription_files = [f for f in os.listdir("demo") if f.endswith(".txt")]
        if not transcription_files:
            raise Exception("No transcription files were generated!")
        print(f"Found {len(transcription_files)} transcription files")
        
        # Step 2: Run Text Summarizer
        print("\n2. Starting Text Summarization...")
        subprocess.run(["python", "batch_text_summarizer_demo.py"], check=True)
        print("✓ Summarization completed successfully")
        
        # Check if summary files were created
        summary_dir = os.path.join("demo", "summarized_texts_demo")
        if not os.path.exists(summary_dir):
            raise Exception("Summarized texts directory was not created!")
        
        summary_files = [f for f in os.listdir(summary_dir) if f.endswith("_summarized.txt")]
        if not summary_files:
            raise Exception("No summary files were generated!")
        print(f"Found {len(summary_files)} summary files")
        
        # Step 3: Concatenate Summaries
        print("\n3. Concatenating Summaries...")
        subprocess.run(["python", "concat_summaries_demo.py"], check=True)
        print("✓ Concatenation completed successfully")
        
        # Verify final concatenated file
        # final_file = os.path.join(summary_dir, "combined_summaries_demo.txt")
        # if not os.path.exists(final_file):
        #     raise Exception("Final combined file was not created!")
        final_file = "./demo/combined_summaries.txt"

        # Step 4: NER Classification
        print("\n4. Starting NER Classification...")
        
        # Load the pre-trained model
        try:
            model = NERModel.load('ner_model.pt')
            print("Loaded pre-trained NER model successfully")
        except FileNotFoundError:
            print("Pre-trained model not found. Please run train.py first.")
            raise
        
        # Load spaCy for text preprocessing
        nlp = spacy.load("en_core_web_sm")
        
        # Read the combined summary
        with open(final_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Prepare output file for NER results
        ner_results_file = os.path.join("demo", "ner_results.txt")
        with open(ner_results_file, 'w', encoding='utf-8') as f:
            f.write("Named Entity Recognition Results\n")
            f.write("=" * 50 + "\n\n")
            
            # Process each sentence
            for sent in doc.sents:
                # Get tokens and prepare for model
                tokens = [token.text for token in sent]
                
                # Encode tokens (you'll need to use the same word2idx from training)
                # This should be loaded from the model checkpoint
                word2idx = model.config['word2idx']
                encoded_tokens = [word2idx.get(t.lower(), word2idx["<UNK>"]) for t in tokens]
                token_tensor = torch.tensor([encoded_tokens], device=model.device)
                
                # Create mask for the sequence
                mask = torch.ones(1, len(tokens), dtype=torch.bool, device=model.device)
                
                # Get predictions
                predictions = model.model.predict(token_tensor, mask)[0]
                
                # Convert predictions to tags
                idx2tag = {v: k for k, v in model.config['tag2idx'].items()}
                predicted_tags = [idx2tag[p] for p in predictions]
                
                # Write results
                f.write(f"Sentence: {sent.text}\n")
                f.write("Entities:\n")
                for token, tag in zip(tokens, predicted_tags):
                    if tag != "O":
                        f.write(f"- {token}: {tag}\n")
                f.write("\n")
        
        print("✓ NER Classification completed successfully")
        print(f"NER results saved to: {ner_results_file}")
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        
        print("\n=== Pipeline Execution Complete ===")
        print(f"Total execution time: {execution_time/60:.2f} minutes")
        print(f"Final output files:")
        print(f"- Summary: {final_file}")
        print(f"- NER Results: {ner_results_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: Pipeline failed at step {e.cmd[1]}")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.output if hasattr(e, 'output') else 'No error output available'}")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    run_pipeline()