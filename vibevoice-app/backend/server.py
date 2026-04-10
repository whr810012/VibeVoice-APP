import os
import sys
import torch
import numpy as np
import fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
import tempfile
import time
import wave
import os
import threading
import uuid
import librosa
from typing import Dict, List

# Add the project root to sys.path to allow imports from vibevoice and demo
sys.path.append(str(Path(__file__).parent.parent.parent))

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from demo.web.app import StreamingTTSService
from vibevoice.modular.streamer import AudioStreamer

# Global model instances
asr_inference = None
tts_service = None
TMP_DIR = Path(__file__).parent / "tmp"
ASR_JOBS: Dict[str, Dict] = {}
ASR_HISTORY: List[Dict] = []
TTS_HISTORY: List[Dict] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    yield

app = FastAPI(title="VibeVoice Desktop Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "en-Carter_man"
    inference_steps: Optional[int] = 5

class ASRInference:
    def __init__(self, model_path="microsoft/VibeVoice-ASR", device="cuda"):
        self.device = device
        self.processor = VibeVoiceASRProcessor.from_pretrained(model_path)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device if device == "auto" else None,
            attn_implementation="flash_attention_2" if device == "cuda" else "sdpa",
            trust_remote_code=True
        )
        if device != "auto":
            self.model = self.model.to(device)
        self.model.eval()

    def transcribe(self, audio_path: str, context_info: str = None):
        inputs = self.processor(
            audio=audio_path,
            return_tensors="pt",
            add_generation_prompt=True,
            context_info=context_info
        )
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

@app.post("/asr")
async def transcribe_audio(file: UploadFile = File(...), context: Optional[str] = None):
    global asr_inference
    if asr_inference is None:
        env_pref = os.getenv("VV_DEVICE", "").lower().strip()
        if env_pref in ("cpu", "cuda"):
            device = "cuda" if env_pref == "cuda" and torch.cuda.is_available() else "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        asr_inference = ASRInference(device=device)
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        result = asr_inference.transcribe(temp_path, context_info=context)
        return {"text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.post("/asr/start")
async def asr_start(file: UploadFile = File(...), context: Optional[str] = None):
    global asr_inference
    if asr_inference is None:
        env_pref = os.getenv("VV_DEVICE", "").lower().strip()
        if env_pref in ("cpu", "cuda"):
            device = "cuda" if env_pref == "cuda" and torch.cuda.is_available() else "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        asr_inference = ASRInference(device=device)
    # persist input to a unique temp file
    job_id = uuid.uuid4().hex
    in_path = TMP_DIR / f"asr_{job_id}_{file.filename}"
    with open(in_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # estimate duration for progress baseline
    try:
        duration = float(librosa.get_duration(filename=str(in_path)))
    except Exception:
        duration = 0.0
    job = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "error": None,
        "result": None,
        "context": context,
        "input_path": str(in_path),
        "total_seconds": duration,
        "started_at": None,
        "canceled": False,
        "finished_at": None,
    }
    ASR_JOBS[job_id] = job
    # background worker
    def _runner():
        job["status"] = "running"
        job["started_at"] = time.time()
        est = max(job["total_seconds"], 30.0)  # fallback baseline
        try:
            # naive progress ticker
            def _tick():
                while job["status"] == "running" and not job["canceled"]:
                    elapsed = time.time() - job["started_at"]
                    pct = min(95, int((elapsed / est) * 90))
                    job["progress"] = max(job["progress"], pct)
                    time.sleep(1.0)
            ticker = threading.Thread(target=_tick, daemon=True)
            ticker.start()
            if not job["canceled"]:
                txt = asr_inference.transcribe(job["input_path"], context_info=job["context"])
                job["result"] = {"text": txt}
                # persist transcription to file for history/download
                txt_path = TMP_DIR / f"asr_{job_id}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(txt)
                job["artifact"] = str(txt_path)
            if job["canceled"]:
                job["status"] = "canceled"
            else:
                job["status"] = "done"
                job["progress"] = 100
                # append to history
                ASR_HISTORY.append({
                    "id": job_id,
                    "finished_at": job["finished_at"],
                    "total_seconds": job["total_seconds"],
                    "text_file": job.get("artifact"),
                    "context": job.get("context"),
                })
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
        finally:
            job["finished_at"] = time.time()
            # cleanup input file
            try:
                os.remove(job["input_path"])
            except Exception:
                pass
    threading.Thread(target=_runner, daemon=True).start()
    return {"job_id": job_id, "status": "queued"}

@app.get("/asr/status/{job_id}")
async def asr_status(job_id: str):
    job = ASR_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    payload = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "error": job["error"],
        "total_seconds": job["total_seconds"],
    }
    return payload

@app.get("/asr/result/{job_id}")
async def asr_result(job_id: str):
    job = ASR_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done" or not job["result"]:
        raise HTTPException(status_code=409, detail="Not ready")
    return job["result"]

@app.post("/asr/cancel/{job_id}")
async def asr_cancel(job_id: str):
    job = ASR_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] in ("done", "error", "canceled"):
        return {"job_id": job_id, "status": job["status"]}
    job["canceled"] = True
    return {"job_id": job_id, "status": "canceling"}

@app.get("/asr/download/{job_id}")
async def asr_download(job_id: str):
    # serve the saved transcription file
    txt_path = TMP_DIR / f"asr_{job_id}.txt"
    if not txt_path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(txt_path), media_type="text/plain", filename=f"ASR_{job_id}.txt")

@app.get("/history/asr")
async def history_asr():
    return {"items": ASR_HISTORY[-100:]}

@app.delete("/history/asr/{job_id}")
async def history_asr_delete(job_id: str):
    txt_path = TMP_DIR / f"asr_{job_id}.txt"
    if txt_path.exists():
        try:
            os.remove(txt_path)
        except Exception:
            pass
    global ASR_HISTORY
    ASR_HISTORY = [i for i in ASR_HISTORY if i.get("id") != job_id]
    return {"ok": True}

@app.post("/tts")
async def generate_speech(request: TTSRequest):
    global tts_service
    if tts_service is None:
        env_pref = os.getenv("VV_DEVICE", "").lower().strip()
        if env_pref in ("cpu", "cuda"):
            device = "cuda" if env_pref == "cuda" and torch.cuda.is_available() else "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_service = StreamingTTSService(model_path="microsoft/VibeVoice-Realtime-0.5B", device=device)
        tts_service.load()
    
    try:
        filename = f"tts_{int(time.time()*1000)}.wav"
        out_path = TMP_DIR / filename
        samplerate = 24000
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            for chunk in tts_service.stream(
                request.text,
                inference_steps=request.inference_steps,
                voice_key=request.voice,
            ):
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.detach().cpu().to(torch.float32).numpy()
                pcm = tts_service.chunk_to_pcm16(chunk)
                wf.writeframes(pcm)
        # record history
        TTS_HISTORY.append({
            "id": filename,
            "voice": request.voice,
            "steps": request.inference_steps,
            "created_at": time.time(),
            "file": str(out_path),
            "text_len": len(request.text or ""),
        })
        return {"message": "ok", "filename": filename, "url": f"/audio/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
async def get_voices():
    # Directly list the directory to avoid dependency on model loading
    root_dir = Path(__file__).parent.parent.parent
    voices_dir = root_dir / "demo" / "voices" / "streaming_model"
    
    if not voices_dir.exists():
        print(f"Warning: Voices directory not found at {voices_dir}")
        return {"voices": []}
    
    voices = [f.stem for f in voices_dir.glob("*.pt")]
    return {"voices": sorted(voices)}

@app.get("/history/tts")
async def history_tts():
    return {"items": TTS_HISTORY[-100:]}

@app.delete("/history/tts/{filename}")
async def history_tts_delete(filename: str):
    target = TMP_DIR / filename
    if target.exists():
        try:
            os.remove(target)
        except Exception:
            pass
    global TTS_HISTORY
    TTS_HISTORY = [i for i in TTS_HISTORY if i.get("id") != filename]
    return {"ok": True}

@app.get("/status")
async def get_status():
    cuda = torch.cuda.is_available()
    dev_name = None
    total_mem = None
    try:
        if cuda:
            dev_name = torch.cuda.get_device_name(0)
            # mem_get_info returns (free, total) in bytes
            free_b, total_b = torch.cuda.mem_get_info()
            total_mem = int(total_b)
    except Exception:
        pass
    return {
        "asr_loaded": asr_inference is not None,
        "tts_loaded": tts_service is not None,
        "device": "cuda" if cuda else "cpu",
        "cuda_available": cuda,
        "cuda_device_name": dev_name,
        "cuda_total_mem_bytes": total_mem,
    }

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    target = TMP_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(target), media_type="audio/wav")

if __name__ == "__main__":
    port_env = os.getenv("VV_PORT")
    try:
        port = int(port_env) if port_env else 8000
    except Exception:
        port = 8000
    uvicorn.run(app, host="127.0.0.1", port=port)
