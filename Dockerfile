FROM python:3.10-slim

WORKDIR /app

# Fix 4: Use a proper persistent cache dir, not /tmp
ENV PIP_NO_CACHE_DIR=1
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fix 6: Pre-download model at build time so cold starts are instant
# Requires HUGGINGFACEHUB_API_TOKEN to be passed as a build arg
ARG HUGGINGFACEHUB_API_TOKEN
ENV HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
RUN python -c "\
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
import os; \
token = os.getenv('HUGGINGFACEHUB_API_TOKEN'); \
AutoTokenizer.from_pretrained('ai4bharat/indictrans2-en-indic-dist-200M', token=token, trust_remote_code=True); \
AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indictrans2-en-indic-dist-200M', token=token, trust_remote_code=True, low_cpu_mem_usage=True); \
print('Model cached successfully')"

EXPOSE 10000

# Fix 9: Run with 2 workers to handle concurrent requests
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "2"]
