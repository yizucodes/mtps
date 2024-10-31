from datasets import load_dataset
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np

def process_audio_chunks(audio_array, sampling_rate, processor, model, device):
    """
    Process long audio in 30-second chunks with overlap
    """
    # Constants for chunking
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
        print(f"Progress: {progress:.1f}% - Processed {position/sampling_rate:.1f}s / {len(audio_array)/sampling_rate:.1f}s", end='\r')
        
        # Move to next chunk
        position += stride_length
    
    print("\nProcessing complete!")
    
    # Combine all chunks
    full_transcription = ' '.join(chunks_transcription)
    return full_transcription

def test_whisper_with_tedlium():
    """
    Test Whisper with TEDLIUM long-form dataset using chunking
    """
    print("Loading TEDLIUM long-form dataset...")
    try:
        dataset = load_dataset("distil-whisper/tedlium-long-form", split="validation[:5]")
        print(f"\nLoaded dataset with {len(dataset)} samples")
        
        # Initialize Whisper
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        print("\nLoading Whisper base model...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
        
        # Process first sample
        print("\nProcessing first audio sample...")
        sample = dataset[0]
        
        print("\nTalk Information:")
        print(f"Speaker ID: {sample['speaker_id']}")
        print(f"Audio sample rate: {sample['audio']['sampling_rate']} Hz")
        print(f"Audio length: {len(sample['audio']['array'])} samples")
        duration = len(sample['audio']['array']) / sample['audio']['sampling_rate']
        print(f"Duration: {duration/60:.2f} minutes")
        
        # Process audio in chunks
        print("\nGenerating transcription (using 30-second chunks with 2-second overlap)...")
        transcription = process_audio_chunks(
            sample["audio"]["array"],
            sample["audio"]["sampling_rate"],
            processor,
            model,
            device
        )
        
        # Save results
        filename = f"transcription_{sample['speaker_id']}_{duration:.0f}s_chunked.txt"
        print(f"\nSaving results to {filename}")
        
        with open(filename, 'w') as f:
            f.write(f"Speaker ID: {sample['speaker_id']}\n")
            f.write(f"Duration: {duration/60:.2f} minutes\n")
            f.write(f"Processing method: 30-second chunks with 2-second overlap\n\n")
            f.write(f"Original text:\n{sample['text']}\n")
            f.write(f"\nWhisper transcription:\n{transcription}\n")
        
        # Print results
        print("\nResults:")
        print(f"Original text length: {len(sample['text'])} characters")
        print(f"Transcription length: {len(transcription)} characters")
        print(f"\nFirst 500 characters of transcription:\n{transcription[:500]}...")
        
        print(f"\nProcessing completed successfully!")
        print(f"Results saved to: {filename}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_whisper_with_tedlium()