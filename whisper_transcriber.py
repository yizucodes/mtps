from datasets import load_dataset
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from datetime import datetime
import os

def process_audio_chunks(audio_array, sampling_rate, processor, model, device):
    """
    Process long audio in 30-second chunks with overlap
    """
    CHUNK_LENGTH_SEC = 30
    OVERLAP_SEC = 2
    
    # Calculate sizes in samples
    chunk_length = CHUNK_LENGTH_SEC * sampling_rate
    overlap_length = OVERLAP_SEC * sampling_rate
    stride_length = chunk_length - overlap_length
    
    # Initialize results
    chunks_transcription = []
    position = 0
    
    while position < len(audio_array):
        # Get chunk
        chunk_end = min(position + chunk_length, len(audio_array))
        chunk = audio_array[position:chunk_end]
        
        # Process chunk
        input_features = processor(
            chunk,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_features.to(device)
        
        # Generate transcription for chunk
        predicted_ids = model.generate(
            input_features,
            language="en",
            num_beams=5,
            no_repeat_ngram_size=3
        )
        
        # Decode chunk
        chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        chunks_transcription.append(chunk_text.strip())
        
        # Update progress
        progress = min(100, (position / len(audio_array)) * 100)
        print(f"Chunk progress: {progress:.1f}% - Processed {position/sampling_rate:.1f}s / {len(audio_array)/sampling_rate:.1f}s", end='\r')
        
        # Move to next chunk
        position += stride_length
    
    # Combine all chunks
    print("\nChunk processing complete!")
    return ' '.join(chunks_transcription)

def process_samples(num_transcriptions=5):
    """
    Process specified number of samples from the TEDLIUM long-form dataset
    Args:
        num_transcriptions (int): Number of transcriptions to process (default: 5)
    """
    print(f"Loading TEDLIUM long-form dataset (processing {num_transcriptions} samples)...")
    try:
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #output_dir = f"transcriptions_{timestamp}"
        output_dir = "demo"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load both validation and test splits
        splits = ['validation', 'test']
        all_results = []
        samples_processed = 0
        
        # Initialize Whisper (do this once outside the loop)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        print("\nLoading Whisper base model...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
        
        # Process each split until we reach desired number of transcriptions
        for split in splits:
            if samples_processed >= num_transcriptions:
                break
                
            dataset = load_dataset("distil-whisper/tedlium-long-form", split=split)
            print(f"\nProcessing {split} split")
            
            # Process each sample in the split
            for idx, sample in enumerate(dataset):
                if samples_processed >= num_transcriptions:
                    break
                    
                print(f"\nProcessing sample {samples_processed + 1}/{num_transcriptions}")
                print(f"Speaker ID: {sample['speaker_id']}")
                
                duration = len(sample['audio']['array']) / sample['audio']['sampling_rate']
                print(f"Duration: {duration/60:.2f} minutes")
                
                # Process audio in chunks
                print(f"Generating transcription...")
                transcription = process_audio_chunks(
                    sample["audio"]["array"],
                    sample["audio"]["sampling_rate"],
                    processor,
                    model,
                    device
                )
                
                # Save individual result
                filename = f"{output_dir}/transcription_{split}_{sample['speaker_id']}_{duration:.0f}s.txt"
                with open(filename, 'w') as f:
                    f.write(f"Split: {split}\n")
                    f.write(f"Speaker ID: {sample['speaker_id']}\n")
                    f.write(f"Duration: {duration/60:.2f} minutes\n")
                    f.write(f"Processing method: 30-second chunks with 2-second overlap\n\n")
                    f.write(f"Original text:\n{sample['text']}\n")
                    f.write(f"\nWhisper transcription:\n{transcription}\n")
                
                # Store results for summary
                result = {
                    'split': split,
                    'speaker_id': sample['speaker_id'],
                    'duration': duration,
                    'original_length': len(sample['text']),
                    'transcription_length': len(transcription),
                    'filename': filename
                }
                all_results.append(result)
                samples_processed += 1
                
                print(f"Results saved to: {filename}")
        
        # Save summary report
        # summary_file = f"{output_dir}/summary_report.txt"
        # with open(summary_file, 'w') as f:
        #     f.write("=== TEDLIUM Long Form Transcription Summary ===\n\n")
        #     f.write(f"Processing date: {timestamp}\n")
        #     f.write(f"Total samples processed: {len(all_results)}\n\n")
            
        #     for split in splits:
        #         split_results = [r for r in all_results if r['split'] == split]
        #         if split_results:
        #             f.write(f"\n{split.upper()} Split Summary:\n")
        #             f.write(f"Number of samples: {len(split_results)}\n")
        #             f.write(f"Total duration: {sum(r['duration'] for r in split_results)/60:.2f} minutes\n")
                    
        #             for result in split_results:
        #                 f.write(f"\n- Speaker: {result['speaker_id']}\n")
        #                 f.write(f"  Duration: {result['duration']/60:.2f} minutes\n")
        #                 f.write(f"  Original text length: {result['original_length']} chars\n")
        #                 f.write(f"  Transcription length: {result['transcription_length']} chars\n")
        
        # print(f"\nProcessing completed successfully!")
        # print(f"All results saved in: {output_dir}")
        # print(f"Summary report: {summary_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    process_samples(1)  # Process 3 transcriptions by default