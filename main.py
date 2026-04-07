from fastapi import FastAPI
from pydantic import BaseModel
import torch
import tiktoken
import torch.nn as nn
from model import GPT
from fastapi.middleware.cors import CORSMiddleware
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_fineweb   = torch.load("model_weights/ckpt_0023000_fineweb_edu.pt", map_location=device, weights_only=False)
ckpt_tiny_stories   = torch.load("model_weights/ckpt_0015500_tiny_stories.pt", map_location=device, weights_only=False)

fineweb_model  = GPT().to(device)
fineweb_model.load_state_dict(ckpt_fineweb["model"])
fineweb_model.eval()

tiny_stories_model = GPT().to(device)
tiny_stories_model.load_state_dict(ckpt_tiny_stories["model"])
tiny_stories_model.eval()

class LoRALinear(nn.Module):
    def __init__(self, linear_module, r=16, alpha=16, dropout=0.1):
        super().__init__()
        self.linear = linear_module
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        in_features = linear_module.in_features
        out_features = linear_module.out_features

        self.A = nn.Parameter(torch.zeros(in_features, r))
        self.B = nn.Parameter(torch.zeros(r, out_features))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.dropout(x @ self.A @ self.B) * self.scaling


def apply_lora(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in ["c_attn", "c_proj"]:
            setattr(module, name, LoRALinear(child))
        else:
            apply_lora(child)

lora_ckpt_path = "model_weights/lora_ckpt_0001000.pt"

lora_model = GPT().to(device)
lora_model.load_state_dict(ckpt_fineweb["model"])  # base = fineweb

apply_lora(lora_model)

lora_state = torch.load(lora_ckpt_path, map_location=device, weights_only=False)
lora_model.load_state_dict(lora_state, strict=False)

lora_model.eval()

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
    max_tokens:  int   = 30
    temperature: float = 0.8
    top_k:       int   = 14

class GenerateResponse(BaseModel):
    generated: str

@app.post("/fineweb-edu/generate", response_model=GenerateResponse)
def generate_fine(req: GenerateRequest):
    tokens = enc.encode(req.prompt)
    idx    = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        out = fineweb_model.generate(idx, req.max_tokens,
                             temperature=req.temperature,
                             top_k=req.top_k)
    text = enc.decode(out[0].tolist())
    return GenerateResponse(generated=text)

@app.post("/tiny-stories/generate", response_model=GenerateResponse)
def generate_tiny(req: GenerateRequest):
    tokens = enc.encode(req.prompt)
    idx    = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        out = tiny_stories_model.generate(idx, req.max_tokens,
                             temperature=req.temperature,
                             top_k=req.top_k)
    text = enc.decode(out[0].tolist())
    return GenerateResponse(generated=text)

@app.post("/lora-us-affairs/generate", response_model=GenerateResponse)
def generate_lora(req: GenerateRequest):
    tokens = enc.encode(req.prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        out = lora_model.generate(idx, req.max_tokens,
                                  temperature=req.temperature,
                                  top_k=req.top_k)

    text = enc.decode(out[0].tolist())
    return GenerateResponse(generated=text)

@app.get("/test")
def health():
    return {"status": "ok"}