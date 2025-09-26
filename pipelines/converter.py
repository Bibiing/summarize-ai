from moviepy import VideoFileClip
from pathlib import Path
import librosa
import soundfile as sf

def convert_video_to_audio(input_path, output_path, target_sr=16000):
    """
    Convert video to audio using moviepy, ensuring consistent WAV output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with VideoFileClip(str(input_path)) as video:
            if video.audio is None:
                print(f"Error: No audio track found in video '{input_path.name}'")
                return None
                
            # Extract audio to temporary file first
            temp_audio_path = output_path.with_suffix('.temp.wav')
            video.audio.write_audiofile(str(temp_audio_path), verbose=False, logger=None)
            
            # Load with librosa and resample to target sample rate for consistency
            print(f"Resampling audio to {target_sr} Hz for optimal processing...")
            data, sr = librosa.load(str(temp_audio_path), sr=target_sr, mono=True)
            
            # Save as high-quality WAV
            sf.write(str(output_path), data, target_sr, subtype='PCM_16')
            
            # Clean up temporary file
            temp_audio_path.unlink()
            
        print(f"Video converted to high-quality WAV: '{input_path.name}' -> '{output_path.name}'")
        print(f"Output format: 16-bit WAV, {target_sr} Hz, mono")
        return output_path
        
    except Exception as e:
        print(f"Error converting video: {e}")
        return None

def convert_audio_format(input_path: Path, output_path: Path, target_sr=16000):
    """
    Convert any audio format to standardized WAV format.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load audio with librosa for better format support
        print(f"Converting {input_path.name} to standardized WAV format...")
        data, sr = librosa.load(str(input_path), sr=target_sr, mono=True)
        
        # Save as high-quality WAV
        sf.write(str(output_path), data, target_sr, subtype='PCM_16')
        print(f"Audio converted: '{input_path.name}' -> '{output_path.name}'")
        print(f"Output format: 16-bit WAV, {target_sr} Hz, mono")
        return output_path
        
    except Exception as e:
        print(f"Error converting audio format: {e}")
        return None
