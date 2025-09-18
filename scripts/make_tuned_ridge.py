# scripts/make_tuned_ridge.py
import os, sys, json, argparse, random
from typing import List

# --- Make sure we can import from src/ ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch
from util import load_model_and_tokenizer, get_device

try:
    from tqdm import tqdm
except Exception:
    # fallback if tqdm not installed
    def tqdm(it, **kw): return it


def parse_args():
    p = argparse.ArgumentParser(
        description="Build diagonal tuned-lens (gamma,beta per layer) via ridge to match x_L."
    )
    p.add_argument("--model", type=str, default="gpt2-medium",
                   help="HF model name (e.g., gpt2, gpt2-medium)")
    p.add_argument("--prompts", type=str, default=os.path.join(ROOT, "data", "prompts_for_tuned.json"),
                   help="JSON file: list of objects with key 'prompt'")
    p.add_argument("--out", type=str, default=os.path.join(ROOT, "data", "tuned_diag.json"),
                   help="Output JSON path for tuned diag weights")
    p.add_argument("--max_txt", type=int, default=800,
                   help="Max number of prompts to use")
    p.add_argument("--pos", type=int, default=-1,
                   help="Position used when reading hidden states (usually -1)")
    p.add_argument("--skip_embedding", action="store_true",
                   help="If set, starts from block1 (drops embedding row). Recommended.")
    p.add_argument("--lnf_mode", type=str, default="last_only",
                   choices=["raw", "last_only", "all"],
                   help="Apply ln_f to last layer only (recommended), or raw/all.")
    p.add_argument("--l2", type=float, default=1e-4,
                   help="Ridge strength (lambda)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

@torch.no_grad()
def compute_layer_gains(model, tokenizer, texts, layers_json, pos=-1, skip_embedding=True, lnf_mode="last_only"):
    device = next(model.parameters()).device
    W_U = model.lm_head.weight.T
    ln_f = getattr(model.transformer, "ln_f", None)
    gains = {}
    use_texts = texts[:200]  # 200 متن کافی است
    for t in use_texts:
        out = model(**tokenizer(t, return_tensors="pt").to(device))
        hs = out.hidden_states
        L = len(hs)-1
        xL = hs[L][0, pos]
        if ln_f is not None and lnf_mode in ("last_only","all"):
            xL = ln_f(xL)
        zL = xL @ W_U
        start = 1 if skip_embedding else 0
        for i in range(start, len(hs)):
            key = str(i)
            if key not in layers_json: 
                continue
            x = hs[i][0, pos]
            if ln_f is not None and lnf_mode=="all":
                x = ln_f(x)
            g = torch.tensor(layers_json[key]["gamma"], device=x.device)
            b = torch.tensor(layers_json[key]["beta"],  device=x.device)
            zt = (g * x + b) @ W_U
            num = (zt * zL).sum()
            den = (zt * zt).sum().clamp(min=1e-12)
            a = float(num / den)
            gains.setdefault(key, []).append(a)
    # میانگین بگیر
    return {k: float(sum(v)/len(v)) for k,v in gains.items() if v}

@torch.no_grad()
def collect_pairs(model, tokenizer, texts: List[str], pos=-1,
                  skip_embedding=True, lnf_mode="last_only"):
    """
    For each prompt, collect (x_l, x_L) pairs at the same position.
    Returns: list of dicts per layer: {"X":[N,d], "Y":[N,d]}, start_index
    """
    device = next(model.parameters()).device
    model.eval()

    pairs_per_layer = None
    start = None

    for t in tqdm(texts, desc="collect"):
        inp = tokenizer(t, return_tensors="pt").to(device)
        out = model(**inp)
        hs = out.hidden_states                 # [emb, h1, ..., hL]
        ln_f = getattr(model.transformer, "ln_f", None)
        L = len(hs) - 1
        s = 1 if skip_embedding else 0
        if start is None:
            start = s

        # Target = last hidden (optionally with ln_f) — matches standard logits path
        xL = hs[L][0, pos]
        if ln_f is not None and lnf_mode in ("last_only", "all"):
            xL = ln_f(xL)

        for i in range(s, len(hs)):
            xi = hs[i][0, pos]
            # Note: we usually DO NOT apply ln_f to middle layers unless lnf_mode=='all'
            if ln_f is not None and lnf_mode == "all":
                xi = ln_f(xi)

            if pairs_per_layer is None:
                d = xi.shape[-1]
                K = len(hs) - s
                pairs_per_layer = [{"X": [], "Y": []} for _ in range(K)]
            li = i - s
            pairs_per_layer[li]["X"].append(xi.detach().cpu())
            pairs_per_layer[li]["Y"].append(xL.detach().cpu())

    # stack
    for li in range(len(pairs_per_layer)):
        pairs_per_layer[li]["X"] = torch.stack(pairs_per_layer[li]["X"], dim=0)  # [N,d]
        pairs_per_layer[li]["Y"] = torch.stack(pairs_per_layer[li]["Y"], dim=0)  # [N,d]
    return pairs_per_layer, start


def fit_diag_ridge(X: torch.Tensor, Y: torch.Tensor, l2=1e-4):
    """
    Solve, per-dimension j:
        min_{gamma_j, beta_j} || gamma_j X_j + beta_j - Y_j ||^2 + l2 * gamma_j^2
    Closed form (ridge on slope only):
        gamma = Cov(X,Y) / (Var(X) + l2)
        beta  = mean(Y) - gamma * mean(X)
    X, Y: [N, d] (float tensors on CPU)
    """
    Xm = X.mean(0)
    Ym = Y.mean(0)
    Xc = X - Xm
    Yc = Y - Ym
    num = (Xc * Yc).sum(0)                 # covariance numerator [d]
    den = (Xc * Xc).sum(0) + float(l2)     # variance + lambda     [d]
    gamma = num / den
    beta  = Ym - gamma * Xm
    return gamma, beta


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load model
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    model.config.output_hidden_states = True

    # load prompts
    texts = []
    if os.path.isfile(args.prompts):
        try:
            data = json.load(open(args.prompts, "r", encoding="utf-8"))
            # accept list of dicts with "prompt" or list of strings
            if isinstance(data, list):
                if len(data) and isinstance(data[0], dict) and "prompt" in data[0]:
                    texts = [d["prompt"] for d in data]
                elif len(data) and isinstance(data[0], str):
                    texts = data
        except Exception as e:
            print("[warn] could not read prompts file:", e)

    if not texts:
        # fallback tiny set (better to provide a real prompts file)
        texts = [
            "The capital of France is ",
            "In 2010, a key idea in computer science was ",
            "Compute 5 + 7 = ",
            "Opposite of cold is ",
            "In Python, write a one-liner to reverse a list: "
        ] * 200

    texts = texts[: args.max_txt]
    if args.debug:
        print(f"[info] using {len(texts)} prompts from: {args.prompts}")

    # collect pairs (x_l, x_L)
    pairs, start = collect_pairs(
        model, tokenizer, texts, pos=args.pos,
        skip_embedding=args.skip_embedding, lnf_mode=args.lnf_mode
    )

    # fit ridge per layer
    layers_json = {}
    for li, buf in enumerate(pairs):
        X, Y = buf["X"], buf["Y"]     # [N,d] on CPU
        gamma, beta = fit_diag_ridge(X, Y, l2=args.l2)
        layers_json[str(li + start)] = {
            "gamma": gamma.tolist(),
            "beta":  beta.tolist(),
        }
    alphas = compute_layer_gains(model, tokenizer, texts, layers_json,
                             pos=args.pos, skip_embedding=args.skip_embedding, lnf_mode=args.lnf_mode)
    for k, a in alphas.items():
        layers_json[k]["alpha"] = a
    print("[info] layer gains (sample):", list(alphas.items())[:5])

    # save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"layers": layers_json}, f)
    print(f"[OK] saved tuned diag -> {args.out}")
    print(f"    layers: {list(layers_json.keys())[:5]} ...")
    print(f"    mode: lnf={args.lnf_mode}, skip_embedding={args.skip_embedding}, l2={args.l2}, N={len(texts)}")


if __name__ == "__main__":
    main()
