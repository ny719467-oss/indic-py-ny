from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# 🔐 API Key
API_KEY = os.getenv("API_KEY", "defaultkey")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model
MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print("✅ Model loaded successfully")

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.get("/")
def home():
    return {"message": "Indic Translation API running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/translate")
def translate(req: TranslationRequest, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text not allowed")

    try:
        input_text = f"{req.source_lang} {req.target_lang} {req.text}"

        inputs = tokenizer(input_text, return_tensors="pt", padding=True)

        outputs = model.generate(
            **inputs,
            max_length=128
        )

        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"translated_text": translated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
