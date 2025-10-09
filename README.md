# AI Audio/Video Summarizer

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
