import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

from pipelines.converter import convert_video_to_audio
from pipelines.transcriber import Transcriber
from pipelines.summarizer import Summarizer
from pipelines.preprocessor import enhance_audio

LANGUAGES = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish", "ru": "russian",
    "ko": "korean", "fr": "french", "ja": "japanese", "pt": "portuguese", "tr": "turkish",
    "pl": "polish", "ca": "catalan", "nl": "dutch", "ar": "arabic", "sv": "swedish",
    "it": "italian", "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay", "cs": "czech",
    "ro": "romanian", "da": "danish", "hu": "hungarian", "ta": "tamil", "no": "norwegian",
    "th": "thai", "ur": "urdu", "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian",
    "la": "latin", "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali", "sr": "serbian",
    "az": "azerbaijani", "sl": "slovenian", "kn": "kannada", "et": "estonian", "mk": "macedonian",
    "br": "breton", "eu": "basque", "is": "icelandic", "hy": "armenian", "ne": "nepali",
    "mn": "mongolian", "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala", "km": "khmer",
    "sn": "shona", "yo": "yoruba", "so": "somali", "af": "afrikaans", "oc": "occitan",
    "ka": "georgian", "be": "belarusian", "tg": "tajik", "sd": "sindhi", "gu": "gujarati",
    "am": "amharic", "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk", "mt": "maltese",
    "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar", "bo": "tibetan", "tl": "tagalog",
    "mg": "malagasy", "as": "assamese", "tt": "tatar", "haw": "hawaiian", "ln": "lingala",
    "ha": "hausa", "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}

load_dotenv()

dir_path = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='Speech Processing App with Authentication')
    parser.add_argument('--file', default="data/" , type=str, required=True, help='Path to the file')
    language_help = "Force transcription language.\n" \
                    "Defaults to auto-detect.\n" \
                    "Available codes:\n" + \
                    "\n".join([f"  {code}: {name}" for code, name in LANGUAGES.items()])
    parser.add_argument('--language',  default=None, type=str, choices=LANGUAGES.keys(), help=language_help)
    parser.add_argument('--denoise', action='store_true', help='Enable audio noise reduction before transcription')
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
        audio_path = output_dir / f"{input_file.stem}.wav"
        convert_video_to_audio(input_file, audio_path)
    elif input_file.suffix.lower() in [".mp3", ".wav"]:
        print(f"audio file: '{input_file.name}'")
        audio_path = input_file
    else:
        print(f"file type not supported: '{input_file.suffix}'. Harap gunakan mp4, mp3.")
        exit(1)
    
    if args.denoise:
        enhanced_audio_path = enhance_audio(audio_path)
        if enhanced_audio_path:
            audio_path = enhanced_audio_path #
        else:
            print("Audio enhancement failed. Proceeding with original audio.")

    try: 
        transcriber = Transcriber(model_name="small")
        summarizer = Summarizer(gemini_api_key=os.getenv("GOOGLE_API_KEY"))
    except Exception as e:
        print(f"Initialization error: {e}")
        exit(1)

    # Transcribe audio to text
    transcription = transcriber.transcribe(audio_path, language=args.language)
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