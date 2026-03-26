from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor  # Fix 2: Correct class name (was IndicTransProcessor)
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Fix 1: No hardcoded fallback — raises immediately if env var is missing
API_KEY = os.getenv("INDIC_API")
if not API_KEY:
    raise ValueError("❌ Set INDIC_API in environment variables. No default is accepted.")

# Hugging Face Token
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ Hugging Face token not found. Set HUGGINGFACEHUB_API_TOKEN")

# Fix 7: CORS — credentials require explicit origins, not wildcard
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Explicit device selection (GPU if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {DEVICE}")

# Whitelist of valid IndicTrans2 language codes
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
    model = model.to(DEVICE)
    model.eval()

    # Fix 2 & 3: Correct class + correct constructor signature
    processor = IndicProcessor(inference_mode=True)

    print("✅ Model and processor loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", str(e))
    raise e

# Fix 9: Thread pool for blocking model inference (keeps event loop free)
executor = ThreadPoolExecutor(max_workers=2)


class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str


def run_translation(text: str, source_lang: str, target_lang: str) -> str:
    """Blocking translation logic — runs in thread pool."""
    # Fix 3: Correct call signature for preprocess_batch
    batch = processor.preprocess_batch(
        [text],
        src_lang=source_lang,
        tgt_lang=target_lang,
    )

    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            early_stopping=True,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Fix 3: Correct postprocess_batch call
    translated_texts = processor.postprocess_batch(decoded, lang=target_lang)
    return translated_texts[0]


@app.get("/")
def home():
    return {"message": "Indic Translation API running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/translate")
async def translate(req: TranslationRequest, x_api_key: str = Header(None)):
    # Secure API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text not allowed")

    # Request length guard to prevent OOM / timeouts
    if len(req.text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long. Maximum allowed length is 1000 characters.")

    # Validate language codes against whitelist
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
        # Fix 9: Run blocking inference in thread pool with a timeout
        loop = asyncio.get_event_loop()
        translated = await asyncio.wait_for(
            loop.run_in_executor(executor, run_translation, req.text, req.source_lang, req.target_lang),
            timeout=60.0  # 60-second max per request
        )
        return {"translated_text": translated}

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Translation timed out. Try a shorter input.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
