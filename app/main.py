import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router

app = FastAPI(
    title="Summarize AI API",
    description="AI-powered audio/video summarization service",
    version="1.0.0"
)

# Add CORS middleware for web deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Summarize AI API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# uvicorn app.main:app --reload