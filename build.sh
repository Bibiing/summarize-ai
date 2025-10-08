#!/usr/bin/env bash
set -o errexit

# Tingkatkan batas waktu unduhan pip untuk mengakomodasi file model yang besar
export PIP_DEFAULT_TIMEOUT=100

# 1. Instal dependensi sistem
# Proyek Anda membutuhkan 'ffmpeg' untuk pemrosesan audio dan video.
echo "Installing system dependencies (ffmpeg)..."
apt-get update && apt-get install -y ffmpeg

# 2. Instal dependensi Python
# Perintah ini akan menginstal semua library dari file requirements.txt Anda.
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Build script finished."