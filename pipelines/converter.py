from pathlib import Path
import numpy as np
import subprocess 

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

    
# command FFmpeg
# -i : file input
# -vn : ignore video (video no)
# -acodec pcm_s16le : format audio output (WAV 16-bit)
# -ar : audio sample rate (Hz)
# -ac : number of audio channels (1 untuk mono)
# -y : overwrite output file if exists

def convert_video_to_audio(input_path: Path, output_path: Path, target_sr=16000):
    """
    Convert video to a standardized mono WAV audio file using FFmpeg.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        'ffmpeg',
        '-i', str(input_path),
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', str(target_sr),
        '-ac', '1',
        '-y', str(output_path)
    ]
    
    try:
        print(f"Converting '{input_path.name}' using FFmpeg...")
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"FFmpeg output: {result.stdout.decode()}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting video with FFmpeg for file: {input_path.name}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
        return None

def convert_audio_format(input_path: Path, output_path: Path, target_sr=16000):
    """
    Convert any audio format to standardized mono WAV format.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        'ffmpeg',
        '-i', str(input_path),
        '-acodec', 'pcm_s16le',
        '-ar', str(target_sr),
        '-ac', '1',
        '-y', str(output_path)
    ]
    
    try:
        print(f"Converting {input_path.name} to standardized WAV format using ffmpeg...")
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"FFmpeg output: {result.stdout.decode()}")
        return output_path

    except subprocess.CalledProcessError as e:
        print(f"Error converting audio format with FFmpeg for file: {input_path.name}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
        return None
