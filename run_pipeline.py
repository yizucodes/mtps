import os
import subprocess
import time
from datetime import datetime

def run_pipeline():
    print("\n=== Starting Pipeline Execution ===")
    start_time = time.time()
    
    # Create the demo directory if it doesn't exist
    os.makedirs("demo", exist_ok=True)
    
    try:
        # Step 1: Running the Whisper Transcriber
        print("\n1. Starting Speech-to-Text Transcription...")
        subprocess.run(["python", "whisper_transcriber.py"], check=True)
        print("✓ Transcription completed successfully")
        
        # Check if transcription files were created
        transcription_files = [f for f in os.listdir("demo") if f.endswith(".txt")]
        if not transcription_files:
            raise Exception("No transcription files were generated!")
        print(f"Found {len(transcription_files)} transcription files")
        
        # Step 2: Run Text Summarizer
        print("\n2. Starting Text Summarization...")
        subprocess.run(["python", "batch_text_summarizer.py"], check=True)
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
        subprocess.run(["python", "concatenate_summaries.py"], check=True)
        print("✓ Concatenation completed successfully")
        
        # Verify final concatenated file
        final_file = os.path.join(summary_dir, "combined_summaries_demo.txt")
        if not os.path.exists(final_file):
            raise Exception("Final combined file was not created!")
        
        # Step 4: Classification
        # Use the file (`combined_summaries.txt`) in `demo` folder as input for the `Final_LSTMCRF.ipynb` notebook.
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        
        print("\n=== Pipeline Execution Complete ===")
        print(f"Total execution time: {execution_time/60:.2f} minutes")
        print(f"Final output file: {final_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: Pipeline failed at step {e.cmd[1]}")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.output if hasattr(e, 'output') else 'No error output available'}")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    run_pipeline()