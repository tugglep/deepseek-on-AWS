import os
import io
import json
import base64
import tempfile
import logging
import time
import threading
from typing import Optional, List, Union, Dict, Any

import torch
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from PIL import Image
import requests
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("deepseek-ocr-transformers")

# Configuration
MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-OCR")
DEFAULT_PROMPT = os.getenv("DEFAULT_PROMPT", "<image>\n<|grounding|>Convert the document to markdown.")

# Global model variables
_model = None
_tokenizer = None
_model_loading_error = None
_startup_time = None

def load_model():
    """Load DeepSeek OCR model using Transformers"""
    global _model, _tokenizer, _model_loading_error

    if _model is not None:
        log.info("Model already loaded, skipping...")
        return

    try:
        from transformers import AutoModel, AutoTokenizer

        log.info("=" * 80)
        log.info("LOADING DEEPSEEK OCR MODEL WITH TRANSFORMERS")
        log.info("Model ID: %s", MODEL_ID)
        log.info("This may take 3-5 minutes...")
        log.info("=" * 80)

        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        log.info("✓ Tokenizer loaded")

        # Load model
        _model = AutoModel.from_pretrained(
            MODEL_ID,
            _attn_implementation='flash_attention_2',
            trust_remote_code=True,
            use_safetensors=True
        )

        # Move to GPU and set to eval mode
        _model = _model.eval().cuda().to(torch.bfloat16)
        log.info("✓ Model loaded on GPU with bfloat16")

        log.info("=" * 80)
        log.info("✓ MODEL READY!")
        log.info("=" * 80)

    except Exception as e:
        _model_loading_error = str(e)
        log.error("=" * 80)
        log.error("✗ FATAL ERROR: Failed to load model")
        log.error("Error: %s", e)
        log.error("=" * 80)
        import traceback
        log.error(traceback.format_exc())
        raise

def _download_http(url: str) -> str:
    """Download file from HTTP/HTTPS URL"""
    log.info(f"Downloading from URL: {url[:100]}...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        suffix = os.path.splitext(url.split("?")[0])[-1] or ".bin"
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(r.content)
        log.info(f"✓ Downloaded {len(r.content)} bytes")
        return tmp
    except requests.RequestException as e:
        log.error(f"✗ HTTP download failed: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download from URL: {str(e)}")

def _download_s3(uri: str) -> str:
    """Download file from S3"""
    if not uri.startswith("s3://"):
        raise ValueError("Not an s3:// URI")
    _, _, rest = uri.partition("s3://")
    bucket, _, key = rest.partition("/")

    log.info(f"Downloading from S3: s3://{bucket}/{key}")

    try:
        s3 = boto3.client("s3")
        fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(key)[-1] or ".bin")
        with os.fdopen(fd, "wb") as f:
            s3.download_fileobj(bucket, key, f)
        log.info(f"✓ S3 download complete")
        return tmp
    except (BotoCoreError, ClientError) as e:
        log.error(f"✗ S3 download failed: {e}")
        raise HTTPException(status_code=400, detail=f"S3 download failed: {str(e)}")

def _bytes_to_tempfile(data: bytes, suffix: str = ".bin") -> str:
    """Write bytes to temporary file"""
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return tmp

def _image_to_path(source: Union[str, bytes]) -> str:
    """Convert image source to file path"""
    if isinstance(source, (bytes, bytearray)):
        return _bytes_to_tempfile(source, suffix=".jpg")

    # String path or URL
    s = str(source)
    if s.startswith(("http://", "https://")):
        return _download_http(s)
    elif s.startswith("s3://"):
        return _download_s3(s)
    else:
        return s

def _pdf_to_images(pdf_path: str, dpi: int = 200) -> List[str]:
    """Convert PDF to image paths"""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_path)
    image_paths = []

    for i in range(len(pdf)):
        page = pdf[i]
        pil = page.render(scale=dpi/72).to_pil()
        pil = pil.convert("RGB") if pil.mode != "RGB" else pil

        # Save to temp file
        fd, tmp = tempfile.mkstemp(suffix=f"_page{i}.jpg")
        with os.fdopen(fd, "wb") as f:
            pil.save(f, format="JPEG")
        image_paths.append(tmp)

    return image_paths

def infer_single_image(image_path: str, prompt: str) -> str:
    """Run inference on a single image using Transformers"""
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        log.info(f"Running inference on: {image_path}")
        start_time = time.time()

        # Use model.infer() as per DeepSeek OCR documentation
        # eval_mode=True is required to return the text result
        result = _model.infer(
            _tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path='/tmp',
            base_size=1024,
            image_size=640,
            crop_mode=True,
            test_compress=False,
            save_results=False,
            eval_mode=True
        )

        elapsed = time.time() - start_time
        log.info(f"✓ Inference complete in {elapsed:.2f}s")

        return result

    except Exception as e:
        log.error(f"✗ Inference failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# FastAPI app
app = FastAPI(title="DeepSeek OCR Transformers SageMaker BYOC", version="1.0.0")

class OCRRequest(BaseModel):
    prompt: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    pdf_url: Optional[str] = None
    pdf_base64: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Load model in background on startup"""
    global _startup_time
    _startup_time = time.time()

    log.info("=" * 80)
    log.info("CONTAINER STARTUP - DeepSeek OCR with Transformers")
    log.info("Model: %s", MODEL_ID)
    log.info("=" * 80)

    # Load model in background thread
    def load_model_background():
        try:
            load_model()
            elapsed = time.time() - _startup_time
            log.info(f"✓ Container ready (startup: {elapsed:.1f}s)")
        except Exception as e:
            log.error(f"✗ Model loading failed: {e}")

    thread = threading.Thread(target=load_model_background, daemon=True)
    thread.start()

@app.get("/ping")
def ping():
    """SageMaker health check"""
    if _model is None:
        if _model_loading_error:
            log.error(f"/ping - Model failed to load: {_model_loading_error}")
            raise HTTPException(status_code=503, detail=f"Model failed: {_model_loading_error}")
        else:
            log.warning("/ping - Model still loading...")
            raise HTTPException(status_code=503, detail="Model loading")

    return {"status": "healthy", "model": MODEL_ID, "backend": "transformers"}

@app.get("/health")
def health():
    """Detailed health check"""
    uptime = time.time() - _startup_time if _startup_time else 0
    return {
        "status": "healthy" if _model is not None else "loading",
        "model_loaded": _model is not None,
        "model_id": MODEL_ID,
        "backend": "transformers",
        "uptime_seconds": round(uptime, 1),
        "error": _model_loading_error
    }

@app.post("/invocations")
async def invocations(req: Request):
    """SageMaker inference endpoint"""
    if _model is None:
        log.error("/invocations called but model not loaded!")
        if _model_loading_error:
            raise HTTPException(status_code=503, detail=f"Model failed: {_model_loading_error}")
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Expected JSON request")

    o = OCRRequest(**body)
    prompt = o.prompt or DEFAULT_PROMPT

    # Handle single image
    if o.image_url or o.image_base64:
        if o.image_url:
            image_path = _image_to_path(o.image_url)
        else:
            image_path = _image_to_path(base64.b64decode(o.image_base64))

        text = infer_single_image(image_path, prompt)

        # Cleanup temp file
        try:
            os.unlink(image_path)
        except:
            pass

        return {"text": text}

    # Handle PDF
    if o.pdf_url or o.pdf_base64:
        if o.pdf_url:
            if o.pdf_url.startswith(("http://", "https://")):
                pdf_path = _download_http(o.pdf_url)
            else:
                pdf_path = _download_s3(o.pdf_url)
        else:
            pdf_path = _bytes_to_tempfile(base64.b64decode(o.pdf_base64), suffix=".pdf")

        # Convert PDF to images
        log.info("Converting PDF to images...")
        image_paths = _pdf_to_images(pdf_path)
        log.info(f"✓ PDF converted to {len(image_paths)} pages")

        # Process each page
        results = []
        for idx, img_path in enumerate(image_paths, start=1):
            log.info(f"Processing page {idx}/{len(image_paths)}...")
            text = infer_single_image(img_path, prompt)
            results.append(f"\n\n## Page {idx}\n\n{text}")

            # Cleanup page image
            try:
                os.unlink(img_path)
            except:
                pass

        # Cleanup PDF
        try:
            os.unlink(pdf_path)
        except:
            pass

        return {"text": "".join(results), "pages": len(image_paths)}

    raise HTTPException(status_code=400, detail="Provide image_url/base64 or pdf_url/base64")
