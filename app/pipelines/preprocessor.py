import soundfile as sf # https://pypi.org/project/soundfile/ for reading/writing WAV files
import noisereduce as nr # https://github.com/timsainb/noisereduce
import librosa #https://librosa.org/doc/latest/index.html, https://github.com/librosa/librosa,  python package for music and audio analysis
import numpy as np
from pathlib import Path
from scipy import signal
from app.pipelines.converter import convert_audio_format

def audio_quality(data: np.ndarray, sr: int) -> dict:
    """
    Assess audio quality to determine optimal enhancement parameters.
    """
    spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)[0] # Seberapa tinggi frekuensi dominan → indikator clarity.
    zcr = librosa.feature.zero_crossing_rate(data)[0] # Berapa sering sinyal melewati nol → indikator noise.
    rms_energy = librosa.feature.rms(y=data)[0] # Energi rata-rata → indikator volume/kekuatan sinyal.
    
    # Simple quality assessment based on energy distribution
    avg_rolloff = np.mean(spectral_rolloff)
    avg_zcr = np.mean(zcr)
    avg_energy = np.mean(rms_energy)
    
    # quality level (higher rolloff and moderate ZCR usually indicate better quality)
    if avg_rolloff > 4000 and avg_energy > 0.01:
        quality_level = "high"
    elif avg_rolloff > 2000 and avg_energy > 0.005:
        quality_level = "medium"
    else:
        quality_level = "low"
    
    return {
        "quality_level": quality_level,
        "spectral_rolloff": avg_rolloff,
        "zero_crossing_rate": avg_zcr,
        "rms_energy": avg_energy
    }

def apply_high_pass_filter(data: np.ndarray, sr: int, cutoff_freq: int = 80) -> np.ndarray:
    """
    Apply high-pass filter to remove low-frequency noise.
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='high')
    return signal.filtfilt(b, a, data)

def enhance_audio_adaptive(data: np.ndarray, sr: int, quality_info: dict) -> np.ndarray:
    """
    Apply adaptive noise reduction based on audio quality assessment.
    """
    quality_level = quality_info["quality_level"]
    
    if quality_level == "low":
        # Aggressive noise reduction for poor quality audio
        # Multi-stage approach for heavily degraded audio
        
        # Stage 1: High-pass filter to remove low-frequency noise
        data = apply_high_pass_filter(data, sr, cutoff_freq=100)
        
        # Stage 2: Spectral gating noise reduction
        data = nr.reduce_noise(
            y=data, 
            sr=sr, 
            prop_decrease=0.9,  # More aggressive
            stationary=False    # Handle non-stationary noise
        )
        
        # Stage 3: Additional spectral subtraction for residual noise
        data = nr.reduce_noise(
            y=data, 
            sr=sr, 
            prop_decrease=0.3,  # Lighter second pass
            stationary=True
        )
        
    elif quality_level == "medium":
        # Moderate noise reduction
        data = apply_high_pass_filter(data, sr, cutoff_freq=80)
        data = nr.reduce_noise(
            y=data, 
            sr=sr, 
            prop_decrease=0.7,
            stationary=False,
        )
        
    else:  # high quality
        # Light noise reduction to preserve quality
        data = nr.reduce_noise(
            y=data, 
            sr=sr, 
            prop_decrease=0.5,
            stationary=True
        )
    
    return data

def normalize_audio(data: np.ndarray, target_level: float = -20.0) -> np.ndarray:
    """
    Normalize audio to consistent level in dB.
    """
    # Calculate current RMS level
    rms = np.sqrt(np.mean(data**2))
    if rms > 0:
        # Convert target level from dB to linear
        target_rms = 10**(target_level/20)
        # Apply normalization
        data = data * (target_rms / rms)
    
    # Prevent clipping
    data = np.clip(data, -0.95, 0.95)
    return data

def enhance_audio(input_path: Path, aggressive_mode: bool = False):
    """
    Enhanced audio preprocessing with quality assessment and adaptive processing.
    """
    try:
        # First, ensure audio is in WAV format with consistent sample rate
        if input_path.suffix.lower() != '.wav':
            print(f"Converting {input_path.name} to WAV format...")
            wav_path = convert_audio_format(input_path)
        else:
            wav_path = input_path
        
        # Load the audio data
        data, rate = sf.read(str(wav_path))
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        print(f"Processing audio: {wav_path.name}")
        print(f"Sample rate: {rate} Hz, Duration: {len(data)/rate:.2f} seconds")
        
        # Assess audio quality
        quality_info = audio_quality(data, rate)
        print(f"Audio quality assessment: {quality_info['quality_level']} quality")
        print(f"  - Spectral rolloff: {quality_info['spectral_rolloff']:.0f} Hz")
        print(f"  - RMS energy: {quality_info['rms_energy']:.4f}")
        
        # Apply adaptive enhancement
        if aggressive_mode:
            # Force low quality processing for very noisy audio
            quality_info["quality_level"] = "low"
            print("Aggressive mode enabled - applying maximum noise reduction")
        
        enhanced_data = enhance_audio_adaptive(data, rate, quality_info)
        
        # Normalize audio level
        enhanced_data = normalize_audio(enhanced_data)
        
        # Save enhanced audio
        output_dir = input_path.parent / "enhanced"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_enhanced.wav"
        
        sf.write(str(output_path), enhanced_data, rate, subtype='PCM_16')
        print(f"Enhanced audio saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error enhancing audio: {e}")
        return None