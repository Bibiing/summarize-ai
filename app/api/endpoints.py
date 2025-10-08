from fastapi import APIRouter, UploadFile, File, Form
from app.services.summarize import run_pipeline
from pathlib import Path
from typing import Optional
import shutil
import uuid
import os
import json

router = APIRouter()

@router.post("/process")
async def process_file(
    file: UploadFile = File(...),
    denoise: bool = Form(False),
    aggressive_denoise: bool = Form(False),
    force_wav: bool = Form(False),
    transcriber_model: str = Form("small"),
    chunk_size: int = Form(2000),
    language: str = Form(None)
):
    os.makedirs("data/temp", exist_ok=True)
    temp_filename = f"data/temp/{uuid.uuid4()}_{file.filename}"
    with open(temp_filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = run_pipeline(
            input_file=Path(temp_filename),
            denoise=denoise,
            aggressive_denoise=aggressive_denoise,
            force_wav=force_wav,
            transcriber_model=transcriber_model,
            chunk_size=chunk_size,
            language=language
        )
        return result
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@router.get("/logs")
def get_logs():
    try:
        with open("logs/apps.json", "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {"error": "Log file not found"}
    except json.JSONDecodeError:
        return {"error": "Log file is corrupted"}
