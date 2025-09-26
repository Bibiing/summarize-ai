# AI Audio/Video Summarizer with Advanced Noise Reduction

An intelligent audio/video processing system that transcripts and summarizes content.

## Features

### 🎵 Audio Processing

- **Universal WAV Conversion**: Automatically converts all audio inputs to high-quality WAV format (16-bit, 16kHz, mono)
- **Advanced Noise Reduction**: Multi-stage noise reduction with adaptive processing based on audio quality assessment
- **Audio Quality Assessment**: Automatic detection of audio quality levels (high/medium/low)
- **Format Support**: MP4, MP3, WAV, M4A, FLAC, OGG

### 🧠 AI Processing

- **Speech-to-Text**: Uses OpenAI Whisper for accurate transcription
- **Intelligent Summarization**: Powered by Google Gemini AI
- **Multi-language Support**: Auto-detection and manual language selection
- **Text Clustering**: Groups content by topics for better organization

### 🔧 Audio Enhancement Pipeline

#### Step 1: Format Standardization

- Converts all inputs to consistent WAV format
- Resamples to optimal 16kHz for Whisper processing
- Ensures mono audio for consistent processing

#### Step 2: Quality Assessment

The system automatically assesses audio quality using:

- **Spectral Rolloff**: Frequency distribution analysis
- **Zero Crossing Rate**: Signal characteristics
- **RMS Energy**: Overall signal strength

#### Step 3: Adaptive Enhancement

Based on quality assessment:

**Low Quality Audio** (Aggressive Processing):

- High-pass filter (100Hz cutoff) to remove low-frequency noise
- Multi-stage spectral gating noise reduction
- Additional spectral subtraction for residual noise
- Up to 90% noise reduction

**Medium Quality Audio** (Moderate Processing):

- High-pass filter (80Hz cutoff)
- Balanced noise reduction with non-stationary noise handling
- 70% noise reduction

**High Quality Audio** (Preservation Mode):

- Light noise reduction to preserve original quality
- 50% noise reduction with stationary noise focus

#### Step 4: Audio Normalization

- Consistent level normalization to -20dB
- Clipping prevention
- Optimal levels for transcription

## 🚀 Usage

### Basic Usage

```bash
python apps.py --file path/to/your/audio_or_video.mp4
```

### With Noise Reduction

```bash
# Adaptive noise reduction (recommended)
python apps.py --file path/to/noisy_audio.mp3 --denoise

# Aggressive noise reduction for very poor quality audio
python apps.py --file path/to/very_noisy_audio.wav --aggressive-denoise
```

### Advanced Options

```bash
# Force language (if auto-detection fails)
python apps.py --file audio.mp3 --denoise --language en

# Force WAV conversion even for WAV inputs
python apps.py --file audio.wav --force-wav --denoise
```

## 📋 Command Line Arguments

| Argument               | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| `--file`               | Path to audio/video file (required)                        |
| `--denoise`            | Enable adaptive noise reduction                            |
| `--aggressive-denoise` | Enable maximum noise reduction for very poor quality audio |
| `--force-wav`          | Force WAV conversion even if input is already WAV          |
| `--language`           | Force specific language (e.g., 'en', 'id', 'zh')           |

## 🌍 Supported Languages

The system supports 99+ languages including:

- **English** (en), **Indonesian** (id), **Chinese** (zh)
- **Spanish** (es), **French** (fr), **German** (de)
- **Japanese** (ja), **Korean** (ko), **Russian** (ru)
- And many more...

Use `python apps.py --help` to see the complete list.

## 🛠️ Installation

### 1. Create Conda Environment

```bash
conda env create -f env.yml
conda activate summarize-ai
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Test Installation

```bash
python apps.py --file data/audio/sample.mp3 --denoise
```

## 📁 Project Structure

```
FP_PBKK/
├── apps.py                 # Main application
├── env.yml                 # Conda environment
├── README.md              # This file
├── data/
│   ├── audio/             # Audio files
│   │   ├── converted/     # Format-standardized files
│   │   └── enhanced/      # Noise-reduced files
│   └── video/             # Video files
└── pipelines/
    ├── converter.py       # Audio/video format conversion
    ├── preprocessor.py    # Advanced audio enhancement
    ├── transcriber.py     # Speech-to-text processing
    └── summarizer.py      # AI-powered summarization
```

## Use Cases

### Perfect for:

- **Noisy Meeting Recordings**: Clean up background noise and chatter
- **Phone Call Recordings**: Remove line noise and improve clarity
- **Lecture Recordings**: Handle room acoustics and microphone issues
- **Podcast Processing**: Standardize audio quality across different sources
- **Video Content**: Extract and clean audio from various video formats

### Audio Quality Scenarios:

- **Poor Quality**: Heavy background noise, low volume, distortion
- **Medium Quality**: Some background noise, reasonable clarity
- **High Quality**: Clean audio with minimal noise

## Tips for Best Results

1. **Use `--aggressive-denoise`** for recordings with heavy background noise
2. **Let the system auto-detect language** for better accuracy
3. **Ensure audio is at least 5 seconds long** for quality assessment
4. **Use consistent file naming** for better organization
5. **Check the enhanced audio files** in the `enhanced/` folder to verify improvements

## Technical Details

### Audio Processing Pipeline

1. **Format Standardization**: LibROSA-based conversion to WAV
2. **Quality Assessment**: Spectral analysis for adaptive processing
3. **Noise Reduction**: Multi-algorithm approach with noisereduce library
4. **Signal Processing**: SciPy filters for frequency domain processing
5. **Normalization**: Consistent audio levels for optimal transcription

### AI Models

- **Whisper**: OpenAI's robust speech recognition
- **Gemini**: Google's advanced language model for summarization
- **Sentence Transformers**: For semantic text clustering

## Troubleshooting

### Common Issues:

1. **"Audio enhancement failed"**: Check if audio file is corrupted
2. **"No audio track found"**: Video file may not contain audio
3. **Poor transcription quality**: Try `--aggressive-denoise` for noisy audio
4. **Language detection issues**: Use `--language` to force specific language

### Performance Tips:

- Aggressive noise reduction takes longer but provides better results for poor quality audio
- WAV conversion adds processing time but ensures consistent quality
- Larger files may require more processing time for quality assessment

---

**Note**: This system is optimized for handling challenging audio scenarios while maintaining the highest possible quality for AI processing.
