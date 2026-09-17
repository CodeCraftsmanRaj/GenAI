import sys
sys.path.append("src")
from common import *
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok

def load_model(cfg):
    return AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])

def prompt(cfg, question):
    return f'Instruction: {cfg["data"]["instruction"]}\nInput: {question}\nResponse:'

def generate(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs,max_new_tokens=60,do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    decoded = tokenizer.decode(out[0],skip_special_tokens=True)
    return decoded[len(text):].strip() if decoded.startswith(text) else decoded
