import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from pipelines.converter import convert_video_to_audio, convert_audio_format
from pipelines.transcriber import Transcriber
from pipelines.summarizer import Summarizer
from pipelines.preprocessor import enhance_audio
from helper import JSONLogger

load_dotenv()

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

dir_path = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='AI Audio/Video Summarizer with Advanced Noise Reduction', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--file', default="data/" , type=str, required=True, help='Path to the audio/video file')
    parser.add_argument('--denoise', action='store_true', help='Enable adaptive audio noise reduction before transcription')
    parser.add_argument('--aggressive-denoise', action='store_true', help='Enable aggressive noise reduction for very poor quality audio')
    parser.add_argument('--force-wav', action='store_true', help='Force conversion to WAV format even if input is already WAV')
    parser.add_argument('--transcriber-model', default="small", type=str, help='Whisper model size: tiny, base, small, medium, large')
    parser.add_argument('--chunk_size', default=2000, type=int, help='Chunk size for text splitting during summarization (default: 2000 characters)')
    
    language_list = "\n".join([f"  {code:<5} : {name}" for code, name in sorted(LANGUAGES.items())])
    language_help = (
        "Force transcription language.\n"
        "Defaults to auto-detect.\n"
        "Available language codes:\n\n"
        f"{language_list}\n"
    )
    parser.add_argument('--language', default=None, type=str, help=language_help)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    input_file = Path(args.file)

    logger = JSONLogger()
    audio_path = None

    logger.log("INITIALIZATION", "INFO", "Starting audio/video processing pipeline", 
               input_file=str(input_file), 
               transcriber_model=args.transcriber_model,
               denoise_enabled=args.denoise,
               aggressive_denoise=args.aggressive_denoise)

    if not input_file.exists():
        logger.log("FILE_VALIDATION", "ERROR", f"Input file not found: {input_file}")
        exit(1)
    
    logger.log("FILE_VALIDATION", "SUCCESS", "Input file validated", file_type=input_file.suffix.lower())
    logger.log("AUDIO_CONVERSION", "INFO", "Starting audio format standardization")
    
    if input_file.suffix.lower() == ".mp4":
        logger.log("VIDEO_PROCESSING", "INFO", f"Processing video file: '{input_file.name}'")
        output_dir = Path("./data/audio")
        audio_path = output_dir / f"{input_file.stem}.wav"
        result = convert_video_to_audio(input_file, audio_path)

        if result is None:
            logger.log("VIDEO_PROCESSING", "ERROR", "Video conversion failed")
            exit(1)
        logger.log("VIDEO_PROCESSING", "SUCCESS", "Video converted to audio", output_file=str(audio_path))

    elif input_file.suffix.lower() in [".mp3", ".wav", ".m4a", ".flac", ".ogg"]:
        logger.log("AUDIO_PROCESSING", "INFO", f"Processing audio file: '{input_file.name}'")
        if input_file.suffix.lower() != ".wav" or args.force_wav:
            output_dir = Path("./data/audio")
            output_dir.mkdir(parents=True, exist_ok=True)
            audio_path = output_dir / f"{input_file.stem}_standardized.wav"
            result = convert_audio_format(input_file, audio_path)
            if result is None:
                logger.log("AUDIO_PROCESSING", "WARNING", "Audio format conversion failed. Using original file.")
                audio_path = input_file
            else:
                logger.log("AUDIO_PROCESSING", "SUCCESS", "Audio format standardized", output_file=str(audio_path))
        else:
            audio_path = input_file
            logger.log("AUDIO_PROCESSING", "INFO", f"Using WAV file directly: {audio_path.name}")
    else:
        logger.log("FILE_VALIDATION", "ERROR", f"Unsupported file type: '{input_file.suffix}'", supported_formats=[".mp4", ".mp3", ".wav", ".m4a", ".flac", ".ogg"])
        exit(1)
    
    # Audio Enhancement (if requested)
    if args.denoise or args.aggressive_denoise:
        logger.log("AUDIO_ENHANCEMENT", "INFO", "Starting audio enhancement", aggressive_mode=args.aggressive_denoise)
        
        enhanced_audio_path = enhance_audio(audio_path, aggressive_mode=args.aggressive_denoise)
        if enhanced_audio_path:
            audio_path = enhanced_audio_path
            logger.log("AUDIO_ENHANCEMENT", "SUCCESS", "Audio enhancement completed")
        else:
            logger.log("AUDIO_ENHANCEMENT", "ERROR", "Audio enhancement failed. Proceeding with original audio.")
    else:
        logger.log("AUDIO_ENHANCEMENT", "INFO", "Audio enhancement skipped by user choice")

    # Initialize AI Models
    logger.log("MODEL_INIT", "INFO", "Initializing AI models")
    
    try:
        transcriber = Transcriber(model_name=args.transcriber_model) 
        summarizer = Summarizer(gemini_api_key=os.getenv("GOOGLE_API_KEY"))
        logger.log("MODEL_INIT", "SUCCESS", "AI models initialized successfully")
    except Exception as e:
        logger.log("MODEL_INIT", "ERROR", f"Initialization error: {e}")
        exit(1)

    # Transcribe audio to text
    logger.log("TRANSCRIPTION", "INFO", f"Starting transcription of: {audio_path.name}")

    transcription = transcriber.transcribe(audio_path, language=args.language)
    if transcription:
        result, language = transcription
        logger.log("TRANSCRIPTION", "SUCCESS", "Transcription completed",
                   detected_language=language,
                   transcript_length_chars=len(result),
                   result=result)
    else:
        logger.log("TRANSCRIPTION", "ERROR", "Transcription failed")
        exit(1)

    # Text Processing and Summarization
    logger.log("TEXT_PROCESSING", "INFO", "Starting text processing and summarization")
    
    # Split text into chunks
    chunks = summarizer.chunk_text(result, args.chunk_size) 
    logger.log("TEXT_CHUNKING", "SUCCESS", "Text split into chunks",
               num_chunks=len(chunks),
               chunk_size=args.chunk_size)
    
    if len(chunks) > 1:
        logger.log("CLUSTERING", "INFO", "Clustering chunks by topic")
        clusters = summarizer.cluster_chunks(chunks)
        logger.log("CLUSTERING", "SUCCESS", "Topic clustering completed",
                   num_clusters=len(clusters))
    else:
        clusters = {0: chunks}
        logger.log("CLUSTERING", "INFO", "Single chunk - no clustering needed")

    # Generate Final Summary
    logger.log("SUMMARIZATION", "INFO", "Generating comprehensive summary")

    cluster_summaries, final_summary = summarizer.get_final_summary(clusters, language=language)
    logger.log("SUMMARIZATION", "SUCCESS", "Final summary generated",
               summary_length_chars=len(final_summary),
               cluster_summaries=cluster_summaries,
               summary=final_summary)

    logger.log("PIPELINE_COMPLETE", "SUCCESS", "Audio/video processing pipeline completed successfully")
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(final_summary)
    print("="*60)
    logger.save()