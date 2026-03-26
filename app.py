from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicTransProcessor
from fastapi.middleware.cors import CORSMiddleware
import torch
import os

app = FastAPI()

# ✅ Fix 5: Validate API key is not the default insecure value at startup
API_KEY = os.getenv("API_KEY", "defaultkey")
if API_KEY == "defaultkey":
    raise ValueError("❌ Set a real API_KEY in environment variables. Do not use 'defaultkey' in production.")

# Hugging Face Token
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ Hugging Face token not found. Set HUGGINGFACEHUB_API_TOKEN")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Fix 3: Explicit device selection (GPU if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {DEVICE}")

# ✅ Fix 4: Whitelist of valid IndicTrans2 language codes
VALID_LANG_CODES = {
    "eng_Latn", "hin_Deva", "ben_Beng", "tam_Taml", "tel_Telu",
    "mar_Deva", "guj_Gujr", "kan_Knda", "mal_Mlym", "pan_Guru",
    "urd_Arab", "ory_Orya", "asm_Beng", "kas_Arab", "kas_Deva",
    "mai_Deva", "mni_Beng", "mni_Mtei", "sat_Olck", "snd_Arab",
    "snd_Deva", "doi_Deva", "brx_Deva", "kok_Deva",
}

# Model
MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    # ✅ Fix 3: Move model to correct device
    model = model.to(DEVICE)
    model.eval()

    # ✅ Fix 1 & 2: Initialize IndicTransToolkit processor
    processor = IndicTransProcessor(quantization=None)

    print("✅ Model and processor loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", str(e))
    raise e


class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str


@app.get("/")
def home():
    return {"message": "Indic Translation API running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/translate")
def translate(req: TranslationRequest, x_api_key: str = Header(None)):
    # ✅ Fix 5: Secure API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text not allowed")

    # ✅ Fix 6: Request length guard to prevent OOM / timeouts
    if len(req.text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long. Maximum allowed length is 1000 characters.")

    # ✅ Fix 4: Validate language codes against whitelist
    if req.source_lang not in VALID_LANG_CODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source_lang '{req.source_lang}'. Must be a valid IndicTrans2 language code."
        )
    if req.target_lang not in VALID_LANG_CODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target_lang '{req.target_lang}'. Must be a valid IndicTrans2 language code."
        )

    try:
        # ✅ Fix 1 & 2: Use IndicTransToolkit processor for correct preprocessing
        batch = processor.preprocess_batch(
            [req.text],
            src_lang=req.source_lang,
            tgt_lang=req.target_lang,
        )

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        # ✅ Fix 3: Move inputs to the same device as the model
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=5,
                early_stopping=True,
            )

        # ✅ Fix 1 & 2: Use IndicTransToolkit processor for correct postprocessing
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated_texts = processor.postprocess_batch(decoded, lang=req.target_lang)

        return {"translated_text": translated_texts[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
