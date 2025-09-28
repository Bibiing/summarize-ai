import whisper
import time

class Transcriber:
    """
    transcribe audio using OpenAI Whisper.
    """
    def __init__(self, model_name="small"):
        print(f"load whisper model: '{model_name}'")
        start_time = time.time()
        self.model = whisper.load_model(model_name)
        print(f"model successfully loaded in {time.time() - start_time:.2f} seconds.")

    def transcribe(self, file_path, language=None):
        print(f"start transcribing: {file_path}")
        if language:
            print(f"Forcing transcription in language: {language}")
        else:
            print("Language not specified, using auto-detection.")

        start_time = time.time()
        try:
            options = {"language": language} if language else {}
            result = self.model.transcribe(str(file_path),  **options) 
            
            print(f"transcribed {time.time() - start_time:.2f} seconds.")
            print(f"Detected language: {result['language']}")
            return result['text'], result['language']
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None