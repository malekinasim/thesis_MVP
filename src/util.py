import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoConfig,AutoModelForCausalLM, AutoTokenizer,logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import random
import json

def load_prompts_from_json(path):
    if not os.path.exists(path):
        raise ValueError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts, labels = [], []

    def norm_label(t):
        if not t:
            return "unknown"
        t = str(t).strip().lower()
        if t.startswith("dec"):
            return "decision"
        if t.startswith("con"):
            return "control"
        return t

    for i, item in enumerate(data):
        if isinstance(item, dict):
            p = item.get("prompt", "")
            t = item.get("type", "")
        else:
            p, t = str(item), ""
        p = p.strip()
        if not p:
            continue
        prompts.append(p)
        labels.append(norm_label(t))
    if not prompts:
        raise ValueError("No prompts found in JSON.")
    return prompts, labels

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
    logging.set_verbosity_info()
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        attn_implementation="eager",   
        dtype=torch.float32
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

    return vec.squeeze(0).detach().cpu().numpy(),layer - k + 1, layer + 1


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
    new_ids = gen_ids[0, input_len:]  
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

# -----------------------------
# show prompt pca in 2 dimention 
# -----------------------------
def pca_prompts(prompts, model, tokenizer, layer=-4, fuse_last_k=4,
                pooling="content_mean", labels=None, annotate=True, path='fig'):
    # 1) بردار نهان هر پرامپت
    vecs= []
    for idx,p in enumerate(prompts):
         vec,l_start,l_end = get_vec(p, model, tokenizer, layer=layer, fuse_last_k=fuse_last_k, pooling=pooling)
         vecs.append(vec)

    X = np.vstack(vecs)

    # 2) PCA دو بعدی
    pca = PCA(n_components=2, random_state=0)
    XY = pca.fit_transform(X)

    # 3) برچسب‌ها (اگر نباشد فقط شماره می‌نویسیم)
    if labels is None:
        labels = [f"p{i}" for i in range(len(prompts))]

    # نگاشت رنگ ساده برای دو کلاس
    color_map = {
        "decision": "tab:blue",
        "control":  "tab:orange"
    }
    # اگر برچسب دیگری بود، خاکستری می‌شود
    def get_color(lab): return color_map.get(str(lab).lower(), "tab:gray")

    plt.figure(figsize=(6.5,6.5))
    for i, (x, y) in enumerate(XY):
        lab = labels[i]
        plt.scatter(x, y, s=70, c=get_color(lab))
        if annotate:
            plt.annotate(f"{i}:{lab}", (x, y), textcoords="offset points", xytext=(6,3), fontsize=9)

    # لگند ساده
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker='o', linestyle='', color=get_color("decision"), label='decision'),
        Line2D([], [], marker='o', linestyle='', color=get_color("control"),  label='control'),
    ]
    plt.legend(handles=handles, frameon=True, loc='best')

    plt.title(f"PCA of hidden vectors (layer={layer}, k={fuse_last_k}, pooling={pooling})")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(path, exist_ok=True)
    filename = f"PCA_hidden_vectors_layer{layer}_k{fuse_last_k}_pooling_{pooling}_by2labels.png"
    plt.savefig(os.path.join(path, filename), dpi=300, bbox_inches="tight")

def generate_stepwise_baseline(prompt: str,model, tokenizer, max_new=50, device=None):
    if device is None:
        device = next(model.parameters()).device 
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device).contiguous()
    out_tokens = []
    saved_logits = [] 
    with torch.no_grad():
        for t in range(max_new):
            out = model(input_ids=ids, use_cache=False, return_dict=True)
            logits_next = out.logits[:, -1, :]
            saved_logits.append(logits_next.detach())
            next_id = torch.argmax(logits_next, dim=-1)  
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1).contiguous()
            out_tokens.append(int(next_id))
    return tokenizer.decode(out_tokens, skip_special_tokens=True),saved_logits





def generate_stepwise_perturbed(prompt, model, tokenizer, max_new=None, inject_step_start=None,inject_step_end=None, 
                                sigma=0.8, device=None,last_k=1,layer=None,use_direction=True, alpha=4.0):
    if device is None:
        device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device).contiguous()
    out_tokens = []
    step_box = {"t": -1,"phase":"measure" ,"swap": None}
    handle = None

    if inject_step_start is not None and inject_step_end is not None:
        if layer is None:
           
            def hook_ln_f_fn(m, inp, out):
                t=step_box["t"]
                if inject_step_start <= t <= inject_step_end   and step_box["phase"]=='inject':
                    k = max(1, last_k)
                    k = min(k, out.shape[1])
                    if use_direction and step_box["swap"] is not None:
                        i1, i2 = step_box["swap"]  
                        W = model.lm_head.weight.to(out.device)
                        d = (W[i2] - W[i1])           
                        d = d / (d.norm() + 1e-8)
                        d_expanded = d.view(1, 1, -1).expand(out.size(0), k, d.numel())
                        out[:, -k:, :] = out[:, -k:, :] + alpha * d_expanded
                    else: 
                        out[:, -k:, :].add_(sigma * torch.randn_like(out[:, -k:, :]))
                return out
            handle = model.transformer.ln_f.register_forward_hook(hook_ln_f_fn)
        else:
            def hook_block_fn(m, inp, out):
                hidden = out[0] 
                t=step_box["t"]
                if inject_step_start <= t <= inject_step_end and step_box["phase"]=='inject':
                    k = max(1, last_k)
                    k = min(k, hidden.shape[1])
                    hidden = hidden.clone()
                    if use_direction and step_box["swap"] is not None:
                        i1, i2 = step_box["swap"]
                        W = model.lm_head.weight.to(hidden.device)
                        d = (W[i2] - W[i1])
                        d = d / (d.norm() + 1e-8)
                        d_expanded = d.view(1, 1, -1).expand(hidden.size(0), k, d.numel())
                        hidden[:, -k:, :] = hidden[:, -k:, :] + alpha * d_expanded
                    else:
                        hidden[:, -k:, :].add_(sigma * torch.randn_like(hidden[:, -k:, :]))
                return (hidden,) + out[1:]
            handle = model.transformer.h[layer].register_forward_hook(hook_block_fn)
    saved_logits = [] 
    with torch.no_grad():
        for t in range(max_new):
            step_box["t"] = t
            step_box["phase"] = "measure"
            out_m= model(input_ids=ids, use_cache=False, return_dict=True)
            logits_m = out_m.logits[:, -1, :]
            if inject_step_start is not None and inject_step_end is not None and inject_step_start <= t <= inject_step_end:
                probs = torch.softmax(logits_m, dim=-1)
                topv, topi = torch.topk(probs, 2, dim=-1)
                step_box["swap"] = (int(topi[0,0]), int(topi[0,1]))
            else:
                step_box["swap"] =None
            
            step_box["phase"] = "inject"
            out= model(input_ids=ids, use_cache=False, return_dict=True)
            logits_next= out.logits[:, -1, :]
            saved_logits.append(logits_next.detach())
            next_id = torch.argmax(logits_next, dim=-1)

            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1).contiguous()
            out_tokens.append(int(next_id))
    if handle is not None:
        handle.remove()
    return tokenizer.decode(out_tokens, skip_special_tokens=True),saved_logits

def generate_with_logits_perturb(prompt,model,tokenizer, inject_step_start=None,inject_step_end=None,max_new=40,sigma=0.2, device=None): 
    if device is None:
        device = next(model.parameters()).device 
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device).contiguous()
    out_tokens = []
    saved_logits=[]
    with torch.no_grad():
        for _ in range(max_new):
            out = model(input_ids=ids, use_cache=False, return_dict=True)
            logits_next = out.logits[:, -1, :]
            logits = logits_next + sigma * torch.randn_like(logits_next)  
            saved_logits.append(logits)
            next_id = torch.argmax(logits, dim=-1)  
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1).contiguous()
            out_tokens.append(int(next_id))

    return tokenizer.decode(out_tokens, skip_special_tokens=True),saved_logits


def kl_from_logits(logits_base, logits_pert):
    P  = torch.softmax(logits_base, dim=-1)     
    logQ = torch.log_softmax(logits_pert, dim=-1)
    kl = F.kl_div(logQ, P, reduction="batchmean")
    return float(kl)



# -----------------------------
# show prompt lhiddenstate logit  pca in 2 dimention 
# -----------------------------
def pca_logits(logit_trajs, labels, colors=None, annotate=True, path='fig'):
    X_all = np.vstack(logit_trajs)             # (sum_T, V)
    pca = PCA(n_components=2, random_state=0)
    X_all=StandardScaler().fit_transform(X_all)
    Z_all = pca.fit_transform(X_all)
    evr = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}")

    # split back per-trajectory
    outs, start = [], 0
    for arr in logit_trajs:
        T = arr.shape[0]
        outs.append(Z_all[start:start+T])
        start += T

    plt.figure(figsize=(6,6))
    for i, Z in enumerate(outs):
        plt.plot(Z[:,0], Z[:,1], label=labels[i])
    plt.title("PCA of next-token logits trajectories")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(True, alpha=0.3); plt.legend()
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "pca_logits_trajectories.png"), dpi=300, bbox_inches="tight")
    plt.close()

def to_np_matrix(logit_list):
    # logit_list: list of tensors shape (1, V)
    return np.vstack([l.squeeze(0).cpu().numpy() for l in logit_list])

def instability_score(prompts, model, tokenizer, vec_pooling="content_mean",
                      layer=-4, k=4, variants_per_prompt=3):
    def cheap_paraphrases(s):
        tpl = [
            s,
            s + " Please think step by step.",
            "In brief: " + s,
            s.replace("Do you", "Would you"),
            s + " Be concise."
        ]
        random.shuffle(tpl)
        return tpl[:variants_per_prompt]

    scores = []
    for p in prompts:
        v0,l_start,l_end = get_vec(p, model, tokenizer, layer=layer, fuse_last_k=k, pooling=vec_pooling)
        dists = []
        for q in cheap_paraphrases(p):
            vq,l_start,l_end  = get_vec(q, model, tokenizer, layer=layer, fuse_last_k=k, pooling=vec_pooling)
            # cosine_distance = 1 - cosine_similarity
            c = cosine_similarity([v0], [vq])[0,0]
            dists.append(1.0 - c)
        scores.append(sum(dists) / len(dists))
    return scores ,l_start,l_end  


