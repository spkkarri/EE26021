from fastapi import FastAPI
from pydantic import BaseModel
import torch
import tiktoken
from model import GPT
from fastapi.middleware.cors import CORSMiddleware

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt   = torch.load("ckpt_0026000.pt", map_location=device, weights_only=False)
model  = GPT().to(device)
model.load_state_dict(ckpt["model"])
model.eval()
enc = tiktoken.get_encoding("gpt2")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt:      str
    max_tokens:  int   = 200
    temperature: float = 0.8
    top_k:       int   = 40

class GenerateResponse(BaseModel):
    generated: str

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    tokens = enc.encode(req.prompt)
    idx    = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model.generate(idx, req.max_tokens,
                             temperature=req.temperature,
                             top_k=req.top_k)
    text = enc.decode(out[0].tolist())
    return GenerateResponse(generated=text)

@app.get("/test")
def health():
    return {"status": "ok"}