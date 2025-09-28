from moviepy import VideoFileClip # https://zulko.github.io/moviepy/reference/reference/moviepy.video.io.VideoFileClip.VideoFileClip.html#moviepy.video.io.VideoFileClip.VideoFileClip
from pathlib import Path
import librosa # https://librosa.org/doc/latest/generated/librosa.load.html#librosa.load, https://github.com/librosa/librosa,  python package for music and audio analysis
import soundfile as sf # https://python-soundfile.readthedocs.io/en/0.13.1/#read-write-functions 
import numpy as np

# Penjelasan dan Sumber
# kenapa menggunakan 16000Hz? 
# teorema sampling, prinsip reproduksi laju sampel, yaitu setidaknya 2kali frekuensi maksimum. fs = 2 * fmax. https://web.stanford.edu/class/engr76/lectures/lecture9-10.pdf
# frekuensi audio manusia umumnya hingga 20Hz - 20000Hz, tapi untuk ucapan biasanya hingga 8000Hz. Jadi 16000Hz sudah cukup. https://en.wikipedia.org/wiki/Intelligibility_(communication)

# mono? karena audio mono lebih sederhana dan mengurangi kompleksitas pemrosesan, terutama untuk tugas seperti pengenalan ucapan.
# jika stereo, ada dua saluran (kiri dan kanan, seperti telinga manusia) yang perlu diproses secara terpisah.

# Kenapa WAV? WAV adalah format audio tanpa kompresi yang mempertahankan kualitas asli rekaman.
# karena kompresi (seperti MP3) memperkecil ukuran file yang dapat menghilangkan detail penting dalam sinyal audio.
# explanation and example: https://www.macaulaylibrary.org/resources/why-wav/

# PCM_16 adalah format penyimpanan data audio yang menggunakan 16-bit per sampel. https://en.wikipedia.org/wiki/Audio_bit_depth

def convert_video_to_audio(input_path, output_path, target_sr=16000):
    """
    Convert video to audio using moviepy.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with VideoFileClip(str(input_path)) as video:
            if video.audio is None:
                print(f"Error: No audio track found in video '{input_path.name}'")
                return None
                
            # Extract audio directly to numpy array (in memory)
            print(f"Extracting audio from video...")
            audio_array = video.audio.to_soundarray()
            original_sr = video.audio.fps
            
            # Convert stereo to mono if necessary
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resample to target sample rate
            print(f"Resampling audio to {target_sr} Hz for optimal processing...")
            data = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
            
            # Save
            sf.write(str(output_path), data, target_sr, subtype='PCM_16')
            
        print(f"Video converted to WAV: '{input_path.name}' -> '{output_path.name}'")
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
        # Load audio
        print(f"Converting {input_path.name} to standardized WAV format...")
        data, sr = librosa.load(str(input_path), sr=target_sr, mono=True)
        
        # Ensure sample rate is correct
        assert sr == target_sr, f"Sample rate mismatch! Expected {target_sr}, got {sr}"

        # Save
        sf.write(str(output_path), data, target_sr, subtype='PCM_16')
        print(f"Audio converted: '{input_path.name}' -> '{output_path.name}'")
        print(f"Output format: 16-bit WAV, {target_sr} Hz, mono")
        return output_path
        
    except Exception as e:
        print(f"Error converting audio format: {e}")
        return None
