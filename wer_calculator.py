from jiwer import wer
import os
from typing import Dict

def calculate_wer_for_file(filepath: str) -> Dict:
    """
    Calculate WER for a single transcription file
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Split content to get original and transcribed text
        sections = content.split('\n\n')
        
        # Extract metadata and texts
        metadata = {}
        for line in sections[0].split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
        
        # Find the original and transcribed texts
        original_start = content.find("Original text:\n") + len("Original text:\n")
        original_end = content.find("\nWhisper transcription:")
        transcription_start = content.find("Whisper transcription:\n") + len("Whisper transcription:\n")
        
        original_text = content[original_start:original_end].strip()
        transcribed_text = content[transcription_start:].strip()
        
        # Calculate WER
        error_rate = wer(original_text, transcribed_text)
        
        result = {
            'speaker_id': metadata.get('Speaker ID', 'Unknown'),
            'duration': metadata.get('Duration', 'Unknown'),
            'wer': error_rate,
            'original_length': len(original_text.split()),
            'transcribed_length': len(transcribed_text.split()),
            'filepath': filepath
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing file {filepath}: {str(e)}")
        return None

def process_transcription_directory(directory_path: str) -> None:
    """
    Process all transcription files in a directory and generate WER report
    """
    try:
        # Get all txt files
        files = [f for f in os.listdir(directory_path) if f.endswith('.txt') and f != 'summary_report.txt']
        results = []
        
        print(f"\nProcessing {len(files)} transcription files...")
        
        # Process each file
        for file in files:
            filepath = os.path.join(directory_path, file)
            result = calculate_wer_for_file(filepath)
            if result:
                results.append(result)
                print(f"Processed {result['speaker_id']}: WER = {result['wer']:.4f}")
        
        # Generate report
        report_path = os.path.join(directory_path, 'wer_report.txt')
        with open(report_path, 'w') as f:
            f.write("=== Word Error Rate (WER) Analysis Report ===\n\n")
            f.write(f"Total files processed: {len(results)}\n")
            f.write(f"Average WER: {sum(r['wer'] for r in results)/len(results):.4f}\n\n")
            
            # Sort by WER
            results.sort(key=lambda x: x['wer'])
            
            # Individual results
            f.write("Individual Results:\n")
            for result in results:
                f.write(f"\nSpeaker: {result['speaker_id']}\n")
                f.write(f"Duration: {result['duration']}\n")
                f.write(f"WER: {result['wer']:.4f}\n")
                f.write(f"Word Counts - Original: {result['original_length']}, ")
                f.write(f"Transcribed: {result['transcribed_length']}\n")
                f.write(f"File: {os.path.basename(result['filepath'])}\n")
                f.write("-" * 50 + "\n")
            
            # Summary statistics
            f.write("\nSummary Statistics:\n")
            f.write(f"Best WER: {min(r['wer'] for r in results):.4f}\n")
            f.write(f"Worst WER: {max(r['wer'] for r in results):.4f}\n")
            f.write(f"Median WER: {sorted(r['wer'] for r in results)[len(results)//2]:.4f}\n")
        
        print(f"\nWER analysis complete! Report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error processing directory: {str(e)}")

if __name__ == "__main__":
    # Use it like this:
    directory_path = "TODO: REPLACE WITH YOUR DIRECTORY"  # Replace with your directory path
    process_transcription_directory(directory_path)