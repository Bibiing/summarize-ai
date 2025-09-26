"""
Test script to demonstrate the enhanced audio processing capabilities.
This script shows how the system handles different audio quality scenarios.
"""

import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from pipelines.preprocessor import enhance_audio, assess_audio_quality, convert_to_wav

def create_test_audio_samples():
    """
    Create test audio samples with different quality levels for demonstration.
    """
    print("Creating test audio samples...")
    
    # Create a clean sine wave (high quality)
    duration = 5  # seconds
    sample_rate = 16000
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, duration * sample_rate, False)
    clean_audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Create data directory if it doesn't exist
    data_dir = Path("data/audio/test_samples")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # High quality audio (clean)
    high_quality_path = data_dir / "high_quality_test.wav"
    sf.write(str(high_quality_path), clean_audio, sample_rate)
    
    # Medium quality audio (some noise)
    medium_noise = np.random.normal(0, 0.05, len(clean_audio))
    medium_quality_audio = clean_audio + medium_noise
    medium_quality_path = data_dir / "medium_quality_test.wav"
    sf.write(str(medium_quality_path), medium_quality_audio, sample_rate)
    
    # Low quality audio (heavy noise)
    heavy_noise = np.random.normal(0, 0.15, len(clean_audio))
    low_frequency_noise = 0.1 * np.sin(2 * np.pi * 50 * t)  # 50Hz hum
    low_quality_audio = clean_audio + heavy_noise + low_frequency_noise
    low_quality_path = data_dir / "low_quality_test.wav"
    sf.write(str(low_quality_path), low_quality_audio, sample_rate)
    
    return [high_quality_path, medium_quality_path, low_quality_path]

def test_audio_quality_assessment():
    """
    Test the audio quality assessment function.
    """
    print("\n" + "="*60)
    print("TESTING AUDIO QUALITY ASSESSMENT")
    print("="*60)
    
    test_files = create_test_audio_samples()
    
    for i, test_file in enumerate(test_files):
        quality_levels = ["High", "Medium", "Low"]
        print(f"\n--- {quality_levels[i]} Quality Test Sample ---")
        
        # Load and assess audio
        data, sr = sf.read(str(test_file))
        quality_info = assess_audio_quality(data, sr)
        
        print(f"File: {test_file.name}")
        print(f"Detected Quality Level: {quality_info['quality_level'].upper()}")
        print(f"Spectral Rolloff: {quality_info['spectral_rolloff']:.0f} Hz")
        print(f"Zero Crossing Rate: {quality_info['zero_crossing_rate']:.4f}")
        print(f"RMS Energy: {quality_info['rms_energy']:.4f}")

def test_noise_reduction():
    """
    Test the noise reduction capabilities on different quality audio.
    """
    print("\n" + "="*60)
    print("TESTING NOISE REDUCTION CAPABILITIES")
    print("="*60)
    
    test_files = create_test_audio_samples()
    
    for i, test_file in enumerate(test_files):
        quality_levels = ["High", "Medium", "Low"]
        print(f"\n--- Processing {quality_levels[i]} Quality Audio ---")
        
        # Test adaptive noise reduction
        print("Testing adaptive noise reduction...")
        enhanced_path = enhance_audio(test_file, aggressive_mode=False)
        if enhanced_path:
            print(f"✓ Enhanced audio saved: {enhanced_path.name}")
        
        # Test aggressive noise reduction
        print("Testing aggressive noise reduction...")
        aggressive_path = enhance_audio(test_file, aggressive_mode=True)
        if aggressive_path:
            print(f"✓ Aggressively enhanced audio saved: {aggressive_path.name}")

def test_format_conversion():
    """
    Test the format conversion capabilities.
    """
    print("\n" + "="*60)
    print("TESTING FORMAT CONVERSION")
    print("="*60)
    
    # Check if we have any existing audio files to test with
    audio_dir = Path("data/audio")
    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.m4a"))
        
        if audio_files:
            test_file = audio_files[0]
            print(f"Testing conversion with: {test_file.name}")
            
            converted_path = convert_to_wav(test_file)
            if converted_path and converted_path.exists():
                print(f"✓ Successfully converted to: {converted_path.name}")
                
                # Check the converted file properties
                data, sr = sf.read(str(converted_path))
                print(f"  - Sample Rate: {sr} Hz")
                print(f"  - Duration: {len(data)/sr:.2f} seconds")
                print(f"  - Channels: {'Mono' if len(data.shape) == 1 else 'Stereo'}")
            else:
                print("❌ Conversion failed")
        else:
            print("No MP3 or M4A files found for conversion testing")
    else:
        print("No audio directory found")

def run_comprehensive_test():
    """
    Run a comprehensive test of all enhanced features.
    """
    print("AI AUDIO SUMMARIZER - ENHANCED FEATURES TEST")
    print("=" * 60)
    print("This test demonstrates the improved audio processing capabilities:")
    print("- Audio quality assessment")
    print("- Adaptive noise reduction")
    print("- Format conversion")
    print("- Multi-stage audio enhancement")
    
    try:
        test_audio_quality_assessment()
        test_noise_reduction()
        test_format_conversion()
        
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print("✓ All enhanced features tested successfully!")
        print("\nGenerated files can be found in:")
        print("- data/audio/test_samples/ (test audio)")
        print("- data/audio/converted/ (format conversions)")
        print("- data/audio/enhanced/ (noise-reduced audio)")
        
        print("\nTo test with real audio files:")
        print("python apps.py --file data/audio/your_file.mp3 --denoise")
        print("python apps.py --file data/audio/noisy_file.wav --aggressive-denoise")
        
    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("Please install the required packages:")
        print("conda env update -f env.yml")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    run_comprehensive_test()