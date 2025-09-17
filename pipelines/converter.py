from moviepy import VideoFileClip
from pathlib import Path

def convert_video_to_audio(input_path, output_path):
    """
    Convert video to audio using moviepy
    """
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
    try:
        with VideoFileClip(str(input_path)) as video:
            audio = video.audio
            audio.write_audiofile(str(output_path))
        print(f"converted '{input_path.name}' to '{output_path.name}'")
    except Exception as e:
        print(f"Error converting video: {e}")
