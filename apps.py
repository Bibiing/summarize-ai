import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

from pipelines.converter import convert_video_to_audio
from pipelines.transcriber import Transcriber
from pipelines.summarizer import Summarizer

load_dotenv()

dir_path = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='Speech Processing App with Authentication')
    parser.add_argument('--file', default="data/" , type=str, help='Path to the file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    input_file = Path(args.file)
    audio_path = None

    if not input_file.exists():
        print(f"file not found: {input_file}")
        exit(1)

    if input_file.suffix.lower() == ".mp4":
        print(f"video file: '{input_file.name}'")
        output_dir = Path("./data/audio")
        audio_path = output_dir / f"{input_file.stem}.mp3"
        convert_video_to_audio(input_file, audio_path)
    elif input_file.suffix.lower() in [".mp3"]:
        print(f"audio file: '{input_file.name}'")
        audio_path = input_file
    else:
        print(f"file type not supported: '{input_file.suffix}'. Harap gunakan mp4, mp3.")
        exit(1)
    
    try: 
        transcriber = Transcriber(model_name="small")
        summarizer = Summarizer(gemini_api_key=os.getenv("GOOGLE_API_KEY"))
    except Exception as e:
        print(f"Initialization error: {e}")
        exit(1)

    # Transcribe audio to text
    transcription = transcriber.transcribe(audio_path)
    if transcription:
        result, language = transcription
        # print(result[:300])
        print(result)

        # Split text into chunks
        chunks = summarizer.chunk_text(result, 2000)
        if len(chunks) > 1:
            # Cluster chunks by topic
            clusters = summarizer.cluster_chunks(chunks)
        else:
            clusters = {0: chunks}  # If only one chunk, assign to single cluster
    else:
        print("Transcription failed.")
        exit(1)

    # Summarize each cluster and then create a final summary
    final_summary = summarizer.get_final_summary(clusters, language=language)
    print("\nFinal Summary:\n", final_summary)