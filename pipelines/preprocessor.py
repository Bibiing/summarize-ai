import soundfile as sf
import noisereduce as nr
from pathlib import Path

def enhance_audio(input_path: Path):
    """
    Enhances audio by reducing background noise.
    """
    try:
        data, rate = sf.read(str(input_path))
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None
        
    # prop_decrease mengontrol seberapa agresif pengurangan noise (0.0 - 1.0)
    reduced_noise_data = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)
    
    output_dir = input_path.parent / "enhanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_enhanced.wav"

    try:
        sf.write(str(output_path), reduced_noise_data, rate)
        print(f"Enhanced audio saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error writing enhanced audio file: {e}")
        return None