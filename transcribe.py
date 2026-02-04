#!/usr/bin/env python3
"""
Audio transcription and summarization script
Transcribes Spanish audio to text and generates a summary
"""

import whisper
import os
from pathlib import Path
from transformers import pipeline
import torch

def transcribe_audio(audio_file, language="es", model_size="base"):
    """
    Transcribe audio file to text using Whisper
    
    Args:
        audio_file: Path to the audio file
        language: Language code (default: "es" for Spanish)
        model_size: Whisper model size (tiny, base, small, medium, large)
    
    Returns:
        Transcription result dictionary
    """
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    
    print(f"Transcribing {audio_file}...")
    result = model.transcribe(
        audio_file,
        language=language,
        verbose=True
    )
    
    return result

def summarize_text(text, max_length=200, min_length=50):
    """
    Summarize Spanish text using transformer model
    
    Args:
        text: Text to summarize
        max_length: Maximum summary length
        min_length: Minimum summary length
    
    Returns:
        Summary text
    """
    print(f"Loading summarization model...")
    
    # Use Spanish BERT model for summarization
    try:
        summarizer = pipeline(
            "summarization",
            model="mrm8488/bert2bert_shared-spanish-finetuned-summarization",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Split text into chunks if too long (models have max token limits)
        max_chunk_length = 1024
        if len(text.split()) > max_chunk_length:
            print("Text is long, processing in chunks...")
            words = text.split()
            chunks = [' '.join(words[i:i+max_chunk_length]) 
                     for i in range(0, len(words), max_chunk_length)]
            summaries = []
            for i, chunk in enumerate(chunks):
                print(f"Summarizing chunk {i+1}/{len(chunks)}...")
                result = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(result[0]['summary_text'])
            summary = ' '.join(summaries)
        else:
            result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            summary = result[0]['summary_text']
        
        return summary
    except Exception as e:
        print(f"Summarization error: {e}")
        return f"Error al generar resumen: {str(e)}"

def main():
    # Input audio file
    audio_file = "CG-20260202.mp3"
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found!")
        return
    
    # Transcribe
    result = transcribe_audio(audio_file, language="es", model_size="base")
    
    # Save transcription to text file
    output_dir = os.getenv("OUTPUT_DIR", ".")
    output_file = os.path.join(output_dir, Path(audio_file).stem + "_transcription.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    print(f"\n{'='*60}")
    print("TRANSCRIPTION:")
    print('='*60)
    print(result["text"])
    print(f"\n{'='*60}")
    print(f"Transcription saved to: {output_file}")
    
    # Generate summary
    print(f"\n{'='*60}")
    print("GENERATING SUMMARY...")
    print('='*60)
    summary = summarize_text(result["text"])
    
    # Save summary to file
    summary_file = os.path.join(output_dir, Path(audio_file).stem + "_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print('='*60)
    print(summary)
    print(f"\n{'='*60}")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
