import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union, Any
from .modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from .processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

class StreamingTTSService:
    def __init__(self, model_path: str = "microsoft/VibeVoice-Realtime-0.5B", device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.voice_cache = {}

    def load(self):
        if self.model is not None:
            return
        
        print(f"Loading TTS model from {self.model_path} to {self.device}...")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)
        self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        print("TTS model loaded.")

    def get_voice_prompt(self, voice_key: str):
        if voice_key in self.voice_cache:
            return self.voice_cache[voice_key]
        
        # Look for voice file in the new resources directory
        # Path relative to the app structure: vibevoice-app/backend/resources/voices/streaming_model/
        # Or relative to where server.py is.
        # But here in service.py, we should ideally be told where the voices are.
        # For now, let's assume a default search path or let the caller provide it.
        
        # Try to find the voice file
        voice_file = None
        possible_paths = [
            Path(__file__).parent.parent / "vibevoice-app" / "backend" / "resources" / "voices" / "streaming_model" / f"{voice_key}.pt",
            Path("vibevoice-app/backend/resources/voices/streaming_model") / f"{voice_key}.pt"
        ]
        
        for p in possible_paths:
            if p.exists():
                voice_file = p
                break
        
        if not voice_file:
            raise FileNotFoundError(f"Voice file for {voice_key} not found.")

        print(f"Loading voice prompt: {voice_key}")
        prompt = torch.load(voice_file, map_local=self.device)
        # Move all tensors in prompt to device
        prompt = {k: {sk: sv.to(self.device) if isinstance(sv, torch.Tensor) else sv 
                  for sk, sv in v.items()} if isinstance(v, dict) else v 
                  for k, v in prompt.items()}
        
        self.voice_cache[voice_key] = prompt
        return prompt

    def stream(self, text: str, voice_key: str = "en-Carter_man", inference_steps: int = 5):
        if self.model is None:
            self.load()
            
        cached_prompt = self.get_voice_prompt(voice_key)
        
        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=cached_prompt,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Set DDPM steps
        self.model.set_ddpm_inference_steps(inference_steps)
        
        with torch.no_grad():
            # VibeVoiceStreaming generate returns a generator of audio chunks
            for chunk in self.model.generate(**inputs):
                yield chunk

    @staticmethod
    def chunk_to_pcm16(chunk: Union[torch.Tensor, np.ndarray]) -> bytes:
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.detach().cpu().to(torch.float32).numpy()
        
        # Ensure range is [-1, 1]
        chunk = np.clip(chunk, -1, 1)
        # Convert to PCM16
        pcm16 = (chunk * 32767).astype(np.int16)
        return pcm16.tobytes()
