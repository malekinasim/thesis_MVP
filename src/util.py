import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def read_file_lines(file_path):
    input_file=None
    if not os.path.exists(file_path):
        raise ValueError(f"the file {file_path}  does not exist")  
    try:
        input_file=open(file_path, 'r' )
        lines=input_file.readlines()
        for i in  enumerate(lines):
            lines[i]=lines[i].strip()
        return lines
    except IOError as e:
        raise ValueError(f"an error occurred when reading file {file_path} : {e}")
    finally:
        if(input_file):
            input_file.close()
# -----------------------------
# Config & device
# -----------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")
# -----------------------------
# Load model/tokenizer
# -----------------------------
def load_model_and_tokenizer(model_name: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 family has no pad token -> map pad to EOS
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
    ).to(device)
    model.eval()
    return model, tokenizer

# -----------------------------
# Utilities for pooling hidden states
# -----------------------------
WORD_RE = re.compile(r"[A-Za-z]")

def last_content_token_index(input_ids, tokenizer):
    toks = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    for i in range(len(toks) - 1, -1, -1):
        if WORD_RE.search(toks[i]):
            return i
    return len(toks) - 1

def mean_over_content_tokens(fused, input_ids, tokenizer):
    # fused: (1, seq, hidden)
    toks = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    idxs = [i for i, t in enumerate(toks) if WORD_RE.search(t)]
    if not idxs:
        return fused.mean(dim=1)  # fallback: mean over all tokens
    sub = fused[:, idxs, :]      # (1, n_content, hidden)
    return sub.mean(dim=1)       # (1, hidden)


def get_vec(text, model, tokenizer, layer=-4, fuse_last_k=4, pooling="content_mean", device=None):
    
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)

    hs = out.hidden_states  # tuple of length = n_layers+1 (incl embedding)
    L = len(hs)
    if layer < 0:
        layer = L + layer
    layer = max(0, min(layer, L - 1))

    k = max(1, min(fuse_last_k, layer + 1))
    # Fuse چند لایه‌ی انتهایی تا لایه‌ی هدف
    fused = sum(hs[layer - k + 1: layer + 1]) / k  # (1, seq, hidden)

    if pooling == "content_mean":
        vec = mean_over_content_tokens(fused, inputs["input_ids"], tokenizer)
    elif pooling == "mean":
        vec = fused.mean(dim=1)
    elif pooling == "content_last":
        idx = last_content_token_index(inputs["input_ids"], tokenizer)
        vec = fused[:, idx, :]
    else:  # "last_token"
        seq_len = inputs["input_ids"].shape[1]
        vec = fused[:, seq_len - 1, :]

    return vec.squeeze(0).detach().cpu().numpy()


# -----------------------------
# KL divergence on next-token distributions
# -----------------------------
def kl_p_ab(prompt_a, prompt_b, model, tokenizer, device=None):
   
    if device is None:
        device = next(model.parameters()).device
    ia = tokenizer(prompt_a, return_tensors="pt").to(device)
    ib = tokenizer(prompt_b, return_tensors="pt").to(device)
    with torch.no_grad():
        oa = model(**ia)  # logits for A
        ob = model(**ib)  # logits for B
    P = F.softmax(oa.logits[:, -1, :], dim=-1)          # (1, V)
    logQ = F.log_softmax(ob.logits[:, -1, :], dim=-1)   # (1, V)
    return float(F.kl_div(logQ, P, reduction="batchmean"))  # KL(P||Q)


def topk_tokens(prompt, model, tokenizer, k=10, device=None):
    if device is None:
        device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
    with torch.no_grad():
        o = model(input_ids=ids)
    probs = F.softmax(o.logits[:, -1, :], dim=-1).squeeze(0)
    vals, idxs = torch.topk(probs, k)
    toks = [tokenizer.decode([i]) for i in idxs.tolist()]
    return list(zip(toks, [float(v) for v in vals]))


# -----------------------------
# Generation (sampled) just to show continuation
# -----------------------------
def generate_answer(text, model, tokenizer, max_new=40, device=None):
    if device is None:
        device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = gen_ids[0, input_len:]  # فقط ادامه
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

# -----------------------------
# show prompt pca in 2 dimention 
# -----------------------------
def pca_prompts(prompts, model, tokenizer, layer=-4, fuse_last_k=4,
                pooling="content_mean", labels=None, annotate=True,path='data'):
    

    vecs = [get_vec(p, model, tokenizer, layer=layer, fuse_last_k=fuse_last_k,
                    pooling=pooling) for p in prompts]
    X = np.vstack(vecs)  
   
    pca = PCA(n_components=2, random_state=0)
    XY = pca.fit_transform(X)  
    evr = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}")
   
    if labels is None:
        labels = [f"p{i}" for i in range(len(prompts))]

  
    plt.figure(figsize=(6,6))
    for i, (x, y) in enumerate(XY):
        plt.scatter(x, y, s=70)
        if annotate:
     
            short = labels[i]
            plt.annotate(f"{i}:{short}", (x, y), textcoords="offset points", xytext=(6,3))
    plt.title(f"PCA of hidden vectors (layer={layer}, k={fuse_last_k}, pooling={pooling})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(path, exist_ok=True) 
    filename = f"PCA_hidden_vectors_layer{layer}_k{fuse_last_k}_pooling_{pooling}.png"
    plt.savefig(os.path.join(path, filename), dpi=300, bbox_inches="tight")

def generate_stepwise_baseline(prompt: str,model, tokenizer, max_new=50, device=None):
    ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
    out_tokens = []
    with torch.no_grad():
        for t in range(max_new):
            out = model(input_ids=ids, use_cache=False, return_dict=True)
            logits_next = out.logits[:, -1, :]
            next_id = torch.argmax(logits_next, dim=-1)  
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
            out_tokens.append(int(next_id))
    return tokenizer.decode(out_tokens, skip_special_tokens=True)





def generate_stepwise_perturbed(prompt, model, tokenizer, max_new=60, inject_step=30, sigma=0.8, device=None):
    if device is None:
        device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
    out_tokens = []
    step_box = {"t": -1}

    def hook_fn(module, inputs, output):
        if step_box["t"] == inject_step:
            output[:, -1, :].add_(sigma * torch.randn_like(output[:, -1, :]))
        return output

    handle = model.transformer.ln_f.register_forward_hook(hook_fn)

    with torch.no_grad():
        for t in range(max_new):
            step_box["t"] = t
            out = model(input_ids=ids, use_cache=False, return_dict=True)
            logits = out.logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1)
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
            out_tokens.append(int(next_id))

    handle.remove()
    return tokenizer.decode(out_tokens, skip_special_tokens=True)



def generate_with_logits_perturb(prompt,tokenizer,model, max_new=40,sigma=0.2):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    out_tokens = []

    with torch.no_grad():
        for _ in range(max_new):
            out = model(input_ids=ids, use_cache=False, return_dict=True)
            logits = out.logits[:, -1, :]

            logits = logits + sigma * torch.randn_like(logits)  

            next_id = torch.argmax(logits, dim=-1)  
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
            out_tokens.append(int(next_id))

    return tokenizer.decode(out_tokens, skip_special_tokens=True)





