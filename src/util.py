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
import json
import pandas as pd
import matplotlib.ticker as mtick
from svcca import svcca_between_prompts,hidden_matrix_for_prompt,svcca_holdout
# --- add to util.py ---
import random, re
import numpy as np
from typing import List, Dict




def load_prompts_with_options(path, tokenizer, require_single_token=True):
    """
    Loads a mixed dataset:
      - MCQ items: have 'options' (and optional 'gold'), go to mcq_items
      - Single items: {'task':'single', 'prompt', 'gold'} go to free_items
      - Others (Control/Decision without options) also go to free_items
    Each MCQ is validated for single-token options when 'require_single_token' is True.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mcq_items, free_items = [], []
    for it in data:
        task = it.get("task") or ("mcq" if "options" in it else "free")
        it["task"] = task
        it.setdefault("pos", -1)

        if task == "mcq":
            opts = it.get("options", [])
            if not opts:
                raise ValueError(f"{it.get('id','?')}: 'options' required for mcq.")
            opt_ids = []
            for o in opts:
                ids = tokenizer.encode(o, add_special_tokens=False)
                if require_single_token and len(ids) != 1:
                    raise ValueError(
                        f"{it.get('id','?')}: option {o!r} not single-token (len={len(ids)}). "
                        "Try leading space/casing to make it a single token."
                    )
                opt_ids.append(ids)
            it["_option_ids"] = opt_ids
            mcq_items.append(it)
        else:
            free_items.append(it)
    return mcq_items, free_items


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

def calc_svcca_between_prompts(base_prompt,pert_propmpt,model, tokenizer,fuse_last_k=4,start_layer=-12,end_layer=-1,
              energy=0.99,out_doc_path='out_doc',fig_path='fig'):

    layers = list(range(start_layer, end_layer + 1)) if start_layer <= end_layer else list(range(start_layer, end_layer - 1, -1))

    rows = svcca_between_prompts(
        base_prompt, pert_propmpt, model, tokenizer,
        layers=layers,
        fuse_last_k=fuse_last_k,
        dim=None,
        energy=energy
    )

   
    os.makedirs(out_doc_path, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_doc_path, f"svcca_between_prompts_layer_{start_layer}_{end_layer}.csv"), index=False)

   
    os.makedirs(fig_path, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
   
    df_plot = df.copy()

    df_plot["layer"] = pd.Categorical(df_plot["layer"], categories=layers, ordered=True)
    df_plot = df_plot.sort_values("layer")

    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))  # مثل 0.983
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)

    plt.plot(df_plot["layer"].astype(int), df_plot["svcca"], marker="o")
    plt.xlabel("Layer index")
    plt.ylabel("SVCCA similarity")
    plt.title("Layer-wise SVCCA between base and perturbed prompt")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f"svcca_between_prompts_layer_{start_layer}_{end_layer}.png"))
    

    print(f"[OK] SVCCA computed for {len(rows)} layers and saved ")
    print(f"[OK] Plot saved")


def calc_svcca_between_prompts(
    base_prompt,
    pert_prompt,
    model,
    tokenizer,
    fuse_last_k=4,
    start_layer=-12,
    end_layer=-1,
    energy=0.95,               
    out_doc_path='out_doc',
    fig_path='fig',
    max_components=20, 
    test_ratio=0.2,
    random_state=42
):
    layers = (list(range(start_layer, end_layer + 1))
              if start_layer <= end_layer
              else list(range(start_layer, end_layer - 1, -1)))

    rows = []
    for layer in layers:
        Xa, l_start, l_end = hidden_matrix_for_prompt(
            base_prompt, model, tokenizer, layer=layer, fuse_last_k=fuse_last_k
        )
        Xb, _, _ = hidden_matrix_for_prompt(
            pert_prompt, model, tokenizer, layer=layer, fuse_last_k=fuse_last_k
        )
        n = min(Xa.shape[0], Xb.shape[0])  # توکن‌های مشترک
        # n_components را محدود می‌کنیم تا از بیش‌برازش جلوگیری شود
        n_components = max(1, min(max_components, n // 2))

        score = svcca_holdout(
            Xa[:n], Xb[:n],dim=None, energy=energy,n_components=n_components,
            test_ratio=test_ratio,random_state=random_state
        )

        rows.append({
            "layer": layer,
            "layer_start": l_start,
            "layer_end": l_end,
            "seq_used": int(n),
            "hidden": int(Xa.shape[1]),
            "n_components": int(n_components),
            "svcca": float(score)
        })

    # ذخیره CSV
    os.makedirs(out_doc_path, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_doc_path, f"svcca_between_prompts_layer_{start_layer}_{end_layer}.csv")
    df.to_csv(csv_path, index=False)

    # رسم نمودار
    os.makedirs(fig_path, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    df_plot = df.copy()
    df_plot["layer"] = pd.Categorical(df_plot["layer"], categories=layers, ordered=True)
    df_plot = df_plot.sort_values("layer")

    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))  

    ymin = max(0.0, df_plot['svcca'].min() - 0.02)
    ymax = min(1.0, df_plot['svcca'].max() + 0.02)
    if ymin >= ymax:  # در صورت اتفاقی برابر شدن
        ymin, ymax = 0.0, 1.0
    ax.set_ylim(ymin, ymax)

    plt.plot(df_plot["layer"].astype(int), df_plot["svcca"], marker="o")
    plt.xlabel("Layer index")
    plt.ylabel("SVCCA similarity (hold-out)")
    plt.title("Layer-wise SVCCA between base and perturbed prompt")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    fig_out = os.path.join(fig_path, f"svcca_between_prompts_layer_{start_layer}_{end_layer}.png")
    plt.savefig(fig_out, dpi=200)

    print(f"[OK] SVCCA computed for {len(rows)} layers and saved -> {csv_path}")
    print(f"[OK] Plot saved -> {fig_out}")

# --- add to util.py ---

def split_train_test(items, test_ratio=0.3, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_test = max(1, int(len(items) * test_ratio))
    test_idx = set(idx[:n_test])
    train, test = [], []
    for i, it in enumerate(items):
        (test if i in test_idx else train).append(it)
    return train, test

def mass_mean_fit_per_layer(train_items, model, tokenizer, get_layer_hiddens_fn, pos=-1):
    """
    Returns dict: layer_idx -> (w vector as torch.Tensor on CPU)
    train_items: list of MCQ dicts: {"prompt","options","gold"}
    get_layer_hiddens_fn: function like mcq_alllayer_hiddens
    """
    # Accumulate per-layer positives/negatives
    pos_acc = {}  # layer -> list[tensor]
    neg_acc = {}
    for it in train_items:
        prompt = it["prompt"]; options = it["options"]; gold = it.get("gold", None)
        if (gold is None) or (gold not in options):  # skip if no label
            continue
        layer_dicts = get_layer_hiddens_fn(model, tokenizer, prompt, options, pos=pos, skip_embedding=True)
        for li, od in enumerate(layer_dicts):
            for opt, h in od.items():
                (pos_acc.setdefault(li, []) if opt == gold else neg_acc.setdefault(li, [])).append(h)
    # compute w = mu_T - mu_F per layer
    W = {}
    for li in pos_acc.keys():
        if li not in neg_acc or len(pos_acc[li]) == 0 or len(neg_acc[li]) == 0:
            continue
        mu_T = torch.stack(pos_acc[li]).mean(dim=0)
        mu_F = torch.stack(neg_acc[li]).mean(dim=0)
        w = (mu_T - mu_F).detach().cpu()
        # normalize optional (often helps)
        w = w / (w.norm() + 1e-8)
        W[li] = w
    return W

def mass_mean_eval_per_layer(test_items, W, model, tokenizer, get_layer_hiddens_fn, pos=-1):
    """
    Given learned W per layer, returns:
      per_layer_acc: dict[layer_idx] -> accuracy over test MCQs
      best_layer, best_acc
    """
    per_layer_right = {li: 0 for li in W.keys()}
    per_layer_total = {li: 0 for li in W.keys()}
    for it in test_items:
        prompt = it["prompt"]; options = it["options"]; gold = it.get("gold", None)
        if (gold is None) or (gold not in options):
            continue
        layer_dicts = get_layer_hiddens_fn(model, tokenizer, prompt, options, pos=pos, skip_embedding=True)
        for li, od in layer_dicts.items():
            if li not in W: 
                continue
            w = W[li]
            # score each option by dot(w, h_opt)
            best_opt, best_s = None, -1e30
            for opt, h in od.items():
                s = float(torch.dot(w, h))
                if s > best_s:
                    best_s, best_opt = s, opt
            per_layer_total[li] += 1
            per_layer_right[li] += int(best_opt == gold)
    per_layer_acc = {li: (per_layer_right[li] / per_layer_total[li] if per_layer_total[li] else 0.0)
                     for li in per_layer_total}
    # pick best
    best_layer, best_acc = None, -1.0
    for li, acc in per_layer_acc.items():
        if acc > best_acc:
            best_layer, best_acc = li, acc
    return per_layer_acc, best_layer, best_acc


_num_re = re.compile(r"^[+-]?\d+$")

def _corrupt_numeric(ans: str):
    m = _num_re.match(ans.strip())
    if not m:
        return None
    v = int(ans.strip())
    delta = random.choice([1,-1,2,-2,3,-3])
    return str(v + delta)

def build_single_pairs(items, all_gold_pool=None, negatives_per=2):
    """
    items: list of dicts with keys {"task","id","prompt","gold", optional "distractors": [..]}
    Returns list of tuples: (prompt, completion, label, group_id)
    """
    pairs = []
    for it in items:
        task = it.get("task","").lower()
        if task not in ("single","singel"):
            continue
        p = it["prompt"]; g = it["gold"]; gid = it.get("id", None)
        # positive
        pairs.append((p, g, 1, gid))
        # negatives
        negs = []
        if "distractors" in it and it["distractors"]:
            negs = list(it["distractors"])
        else:
            # numeric corruption
            c = _corrupt_numeric(g)
            if c is not None:
                negs = [c]
            # fallback to impostor golds
            if (not negs) and all_gold_pool:
                # sample a few different golds
                cand = [z for z in all_gold_pool if z != g]
                random.shuffle(cand)
                negs.extend(cand[:negatives_per])
        # cap count
        negs = negs[:max(negatives_per, 1)]
        for n in negs:
            pairs.append((p, n, 0, gid))
    return pairs  # list of (prompt, completion, label, group)

def mass_mean_fit_from_pairs(
    pairs, model, tokenizer, get_single_hiddens_fn, skip_embedding=True
):
    """
    pairs: list of (prompt, completion, label, group_id)
    Returns: dict[layer_idx] -> w (torch.Tensor on CPU, normalized)
    """
    # batch forward in chunks to keep memory sane
    BATCH = 32
    pos_per_layer, neg_per_layer = {}, {}
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i+BATCH]
        prompts = [x[0] for x in chunk]
        comps   = [x[1] for x in chunk]
        labels  = [x[2] for x in chunk]
        layer_vecs = get_single_hiddens_fn(
            model, tokenizer, prompts, comps, skip_embedding=skip_embedding
        )  # List[L] of [b,D]
        L = len(layer_vecs)
        for li in range(L):
            V = layer_vecs[li]  # [b,D] on CPU
            for j, lab in enumerate(labels):
                if lab == 1:
                    pos_per_layer.setdefault(li, []).append(V[j])
                else:
                    neg_per_layer.setdefault(li, []).append(V[j])
    W = {}
    for li in pos_per_layer.keys():
        if li not in neg_per_layer or len(pos_per_layer[li]) == 0 or len(neg_per_layer[li]) == 0:
            continue
        mu_T = torch.stack(pos_per_layer[li]).mean(dim=0)
        mu_F = torch.stack(neg_per_layer[li]).mean(dim=0)
        w = mu_T - mu_F
        w = w / (w.norm() + 1e-8)
        W[li] = w.detach().cpu()
    return W

def eval_binary_from_pairs(pairs, W, model, tokenizer, get_single_hiddens_fn, skip_embedding=True):
    """
    Returns per-layer AUROC-like ranking metric is not computed here to keep deps minimal;
    we return per-layer accuracy at threshold 0 (dot>=0 -> True).
    """
    from sklearn.metrics import accuracy_score
    import numpy as np

    BATCH = 64
    all_scores = {li: [] for li in W.keys()}
    all_labels = []

    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i+BATCH]
        prompts = [x[0] for x in chunk]
        comps   = [x[1] for x in chunk]
        labels  = [x[2] for x in chunk]
        layer_vecs = get_single_hiddens_fn(
            model, tokenizer, prompts, comps, skip_embedding=skip_embedding
        )
        for li, w in W.items():
            V = layer_vecs[li]  # [b,D]
            sc = torch.matmul(V, w)  # [b]
            all_scores[li].extend(sc.tolist())
        all_labels.extend(labels)

    per_layer_acc = {}
    y = np.array(all_labels)
    for li in W.keys():
        s = np.array(all_scores[li])
        yhat = (s >= 0).astype(int)
        per_layer_acc[li] = float((yhat == y).mean())
    # pick best
    best_layer, best_acc = None, -1.0
    for li, acc in per_layer_acc.items():
        if acc > best_acc:
            best_layer, best_acc = li, acc
    return per_layer_acc, best_layer, best_acc
