from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(title="Summarize AI API")
app.include_router(router)

# uvicorn app.main:app --reload