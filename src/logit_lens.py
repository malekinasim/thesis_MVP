# logit_lens.py
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

# =============== Tuned Lens (Diagonal) ===============
class TunedDiag:
    """
    Simple diagonal tuned lens:
        x' = gamma[l] ⊙ x + beta[l]
    Optional per-layer gain on logits:
        z'_tuned = alpha[l] * (x' @ W_U)
    """
    def __init__(self, gamma=None, beta=None, alpha=None):
        self.gamma  = gamma or {}   # dict[int] -> torch.Tensor[d]
        self.beta   = beta  or {}   # dict[int] -> torch.Tensor[d]
        self._alpha = alpha or {}   # dict[int] -> float

    @staticmethod
    def from_json(path, device):
        import json, torch
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)  # {"layers": {"12": {"gamma":[...], "beta":[...], "alpha": 0.97}, ...}}
        gamma, beta, alpha = {}, {}, {}
        for k, v in data.get("layers", {}).items():
            l = int(k)
            if "gamma" in v:
                gamma[l] = torch.tensor(v["gamma"], dtype=torch.float32, device=device)
            if "beta" in v:
                beta[l]  = torch.tensor(v["beta"],  dtype=torch.float32, device=device)
            if "alpha" in v:
                alpha[l] = float(v["alpha"])
        return TunedDiag(gamma=gamma, beta=beta, alpha=alpha)

    def apply(self, l, x):  # x: [d] -> returns transformed hidden
        g = self.gamma.get(l)
        b = self.beta.get(l)
        if g is not None:
            x = x * g
        if b is not None:
            x = x + b
        return x

    def alpha(self, l):     # returns float or None
        return self._alpha.get(l, None)


# =============== Layerwise logits (full vocab or options) ===============
@torch.no_grad()
def layerwise_logits_for_pos(
    model, tokenizer, text=None, outputs=None, pos=-1,
    ln_f_mode="last_only",      # "raw" | "last_only" | "all"
    skip_embedding=False,       # True => start from block1 (drop embedding row)
    tuned: "TunedDiag|None" = None,
    option_ids: "list[int]|None" = None   # if provided -> return logits only for these ids
):
    """
    Returns:
      - if tuned is None: list[Tensor]   (per-layer logits)
      - else: {"raw": list[Tensor], "tuned": list[Tensor]}
    If option_ids is not None, each Tensor has shape [|options|] instead of [V].
    """
    device = next(model.parameters()).device
    if outputs is None:
        assert text is not None, "Provide `text` or `outputs`"
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(**inputs)

    hs  = outputs.hidden_states                 # [emb, h1, ..., hL]
    W_U = model.lm_head.weight.T               # [d, V]
    ln_f = getattr(model.transformer, "ln_f", None)
    start = 1 if skip_embedding else 0
    L = len(hs) - 1


    WU_opts = None
    if option_ids is not None:
        opt_idx = torch.as_tensor(option_ids, device=W_U.device, dtype=torch.long)
        WU_opts = torch.index_select(W_U, dim=1, index=opt_idx)  # [d, |opts|]

    def project_vec(x, i):
        if ln_f is not None:
            if ln_f_mode == "last_only" and i == L:
                x = ln_f(x)
            elif ln_f_mode == "all":
                x = ln_f(x)
        if option_ids is None:
            return x @ W_U                    # [V]
        else:
            WU_opts = W_U[:, option_ids]      # [d, |opts|]
            return x @ WU_opts                # [|opts|]

    layer_raw, layer_tuned = [], []
    for i in range(start, len(hs)):
        x = hs[i][0, pos]                     # [d]
        z_raw = project_vec(x, i)
        layer_raw.append(z_raw)
        if tuned is not None:
            xt = tuned.apply(i, x)
            z_tuned = project_vec(xt, i)
            a_i  = tuned.alpha(i)
            if a_i is not None:    
                z_tuned = a_i * z_tuned
            layer_tuned.append(z_tuned)

    if tuned is None:
        return layer_raw
    else:
        return {"raw": layer_raw, "tuned": layer_tuned}

# =============== Margins (full vocab & options) ===============
def _top1_top2_margin_from_logits(z: torch.Tensor) -> float:
    v = torch.topk(z, k=2).values
    return float(v[0] - v[1])

def _gold_margin_from_logits(z: torch.Tensor, gold_id: int) -> float:
    mask = torch.ones_like(z, dtype=torch.bool); mask[gold_id] = False
    return float(z[gold_id] - torch.max(mask * z + (~mask) * (-1e30)))

def _gold_margin_opts_1d(z: torch.Tensor, gidx: int) -> float:
    # z shape: [|opts|]
    if z.numel() < 2:
        return float("nan")
    if gidx == 0:
        rival = torch.max(z[1:])
    elif gidx == z.shape[0] - 1:
        rival = torch.max(z[:-1])
    else:
        rival = torch.max(torch.stack([torch.max(z[:gidx]), torch.max(z[gidx+1:])]))
    return float(z[gidx] - rival)

@torch.no_grad()
def compute_margins_per_layer(
    model, tokenizer, text=None, outputs=None, pos=-1,
    ln_f_mode="last_only", skip_embedding=False,
    gold_text=None,                   # full-vocab gold (single-token)
    options: "list[str]|None" = None, # MCQ options (single-token)
    gold_option: "str|None" = None,   # gold among options
    tuned: "TunedDiag|None" = None
):
    # IDs
    gold_id = None
    if gold_text and gold_text.strip():
        g_ids = tokenizer(gold_text, add_special_tokens=False)["input_ids"]
        if g_ids:
            gold_id = g_ids[0]

    option_ids, gold_opt_idx = None, None
    if options:
        option_ids = []
        for o in options:
            ids = tokenizer.encode(o, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Option {o!r} must be single-token.")
            option_ids.append(ids[0])
        if gold_option is not None:
            if gold_option not in options:
                raise ValueError("gold_option must be one of options")
            gold_opt_idx = options.index(gold_option)

    # logits
    Z_full = layerwise_logits_for_pos(
        model, tokenizer, text=text, outputs=outputs, pos=pos,
        ln_f_mode=ln_f_mode, skip_embedding=skip_embedding,
        tuned=tuned, option_ids=None
    )
    Z_opts = None
    if option_ids is not None:
        Z_opts = layerwise_logits_for_pos(
            model, tokenizer, text=text, outputs=outputs, pos=pos,
            ln_f_mode=ln_f_mode, skip_embedding=skip_embedding,
            tuned=tuned, option_ids=option_ids
        )

    def pack(Zs, kind):
        out = {}
        def compute_list(t_list):
            res = {}
            res[f"top1_top2_{kind}"] = [_top1_top2_margin_from_logits(z) for z in t_list]
            if kind == "full" and gold_id is not None:
                res["gold_full"] = [_gold_margin_from_logits(z, gold_id) for z in t_list]
            if kind == "opts" and gold_opt_idx is not None:
                res["gold_opts"] = [_gold_margin_opts_1d(z, gold_opt_idx) for z in t_list]
            return res
        if isinstance(Zs, dict):
            out["raw"]   = compute_list(Zs["raw"])
            out["tuned"] = compute_list(Zs["tuned"])
        else:
            out["raw"]   = compute_list(Zs)
        return out

    out = {"full": pack(Z_full, "full")}
    if Z_opts is not None:
        out["opts"] = pack(Z_opts, "opts")
    return out

# =============== MCQ layerwise (logits per option) ===============
@torch.no_grad()
def mcq_alllayer_scores(
    model, tokenizer, prompt_text, options, gold_opt=None,
    pos=-1, ln_f_mode="last_only", skip_embedding=True,
    tuned: "TunedDiag|None" = None
):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    out = model(**inputs)
    hs  = out.hidden_states                   # [emb, h1, ..., hL]
    W_U = model.lm_head.weight.T
    ln_f = getattr(model.transformer, "ln_f", None)
    start = 1 if skip_embedding else 0
    L = len(hs) - 1
    opt_ids = []
    for o in options:
        ids = tokenizer.encode(o, add_special_tokens=False)
        assert len(ids) == 1, f"Option {o!r} must be single-token"
        opt_ids.append(ids[0])
    
    opt_idx = torch.as_tensor(opt_ids, device=W_U.device, dtype=torch.long)
    WU_opts = torch.index_select(W_U, dim=1, index=opt_idx)
    def project_vec(x, i):
            if ln_f is not None and ((ln_f_mode == "last_only" and i == L) or ln_f_mode == "all"):
                x = ln_f(x)
            return x @ WU_opts

    # RAW
    raw_scores, raw_winners, raw_m12, raw_gold = [], [], [], []
    # TUNED
    tuned_scores, tuned_winners, tuned_m12, tuned_gold = [], [], [], []

    for i in range(start, len(hs)):
        x = hs[i][0, pos]
        zr = project_vec(x, i)
        scores_r = {opt: float(zr[j]) for j, opt in enumerate(options)}
        raw_scores.append(scores_r)
        ranked_r = sorted(scores_r.items(), key=lambda kv: kv[1], reverse=True)
        raw_winners.append(ranked_r[0][0])
        raw_m12.append(ranked_r[0][1] - (ranked_r[1][1] if len(ranked_r)>1 else float("nan")))
        if gold_opt is not None and gold_opt in scores_r:
            best_r = max(v for k, v in scores_r.items() if k != gold_opt)
            raw_gold.append(scores_r[gold_opt] - best_r)

        # tuned
        if tuned is not None:
            xt = tuned.apply(i, x)
            zt = project_vec(xt, i)
            a_i = tuned.alpha(i)
            if a_i is not None:
                zt = a_i * zt
            scores_t = {opt: float(zt[j]) for j, opt in enumerate(options)}
            tuned_scores.append(scores_t)
            ranked_t = sorted(scores_t.items(), key=lambda kv: kv[1], reverse=True)
            tuned_winners.append(ranked_t[0][0])
            tuned_m12.append(ranked_t[0][1] - (ranked_t[1][1] if len(ranked_t)>1 else float("nan")))
            if gold_opt is not None and gold_opt in scores_t:
                best_t = max(v for k, v in scores_t.items() if k != gold_opt)
                tuned_gold.append(scores_t[gold_opt] - best_t)

    return {
        "raw":   (raw_scores, raw_winners, raw_m12, raw_gold),
        "tuned": (tuned_scores, tuned_winners, tuned_m12, tuned_gold) if tuned is not None else None,
        "outputs": out
    }

# =============== Save / Plot (CSV & Figures) ===============
def save_perlayer_csv_both(res, options, out_dir="out_doc", fname="mcq_perlayer_margins.csv"):
    os.makedirs(out_dir, exist_ok=True)
    raw_scores, raw_winners, raw_m12, raw_gold = res["raw"]
    tuned_part = res["tuned"]

    rows = []
    L = len(raw_scores)
    for li in range(L):
        row = {
            "layer": li,
            "winner_raw":  raw_winners[li],
            "margin_top1_top2_raw": raw_m12[li],
            "gold_margin_opts_raw": (raw_gold[li] if raw_gold else None),
        }
        if tuned_part is not None and li < len(tuned_part[0]):
            tuned_scores, tuned_winners, tuned_m12, tuned_gold = tuned_part
            row.update({
                "winner_tuned": tuned_winners[li],
                "margin_top1_top2_tuned": tuned_m12[li],
                "gold_margin_opts_tuned": (tuned_gold[li] if tuned_gold else None),
            })
        for opt in options:
            row[f"logit_raw[{opt}]"] = raw_scores[li][opt]
            if tuned_part is not None and li < len(tuned_part[0]):
                row[f"logit_tuned[{opt}]"] = tuned_scores[li][opt]
        rows.append(row)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, fname), index=False)

def save_csv_margins(margins, out_dir='out_doc', fname='margins_per_layer.csv'):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    def add_rows(branch_name, data_dict):
        if not data_dict: return
        L = max((len(v) for v in data_dict.values()), default=0)
        for i in range(L):
            row = {"layer": i, "branch": branch_name}
            for k, seq in data_dict.items():
                if i < len(seq): row[k] = seq[i]
            rows.append(row)
    for space in ("full", "opts"):
        if space in margins:
            for ver in ("raw","tuned"):
                if ver in margins[space]:
                    add_rows(f"{space}_{ver}", margins[space][ver])
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, fname), index=False)

def plot_perlayer_margins_both(res, out_png="fig/mcq_margins_per_layer_both.png",
                               title="Per-layer margins (raw vs tuned)"):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    raw_scores, _, raw_m12, raw_gold = res["raw"]
    tuned_part = res["tuned"]

    plt.figure(figsize=(8,4))
    plt.plot(raw_m12, label="top1-top2 (raw)")
    if raw_gold:
        plt.plot(raw_gold, label="gold-margin (raw)")
    if tuned_part is not None:
        _, _, tuned_m12, tuned_gold = tuned_part
        plt.plot(tuned_m12, label="top1-top2 (tuned)")
        if tuned_gold:
            plt.plot(tuned_gold, label="gold-margin (tuned)")
    plt.xlabel("Layer index (0 = first after embedding if skipped)")
    plt.ylabel("Margin")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_plot_margins(margins, path='fig', fname='margins_per_layer.png', title='Margins per Layer'):
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(8,4))
    if "full" in margins:
        m = margins["full"]
        plt.plot(m["raw"]["top1_top2_full"], label="top1-top2 full (raw)")
        if "gold_full" in m["raw"]:
            plt.plot(m["raw"]["gold_full"], label="gold full (raw)")
        if "tuned" in m:
            plt.plot(m["tuned"]["top1_top2_full"], label="top1-top2 full (tuned)")
            if "gold_full" in m["tuned"]:
                plt.plot(m["tuned"]["gold_full"], label="gold full (tuned)")
    if "opts" in margins:
        m = margins["opts"]
        plt.plot(m["raw"]["top1_top2_opts"], label="top1-top2 opts (raw)")
        if "gold_opts" in m["raw"]:
            plt.plot(m["raw"]["gold_opts"], label="gold opts (raw)")
        if "tuned" in m:
            plt.plot(m["tuned"]["top1_top2_opts"], label="top1-top2 opts (tuned)")
            if "gold_opts" in m["tuned"]:
                plt.plot(m["tuned"]["gold_opts"], label="gold opts (tuned)")
    plt.xlabel("Layer index (0=embedding unless skipped)")
    plt.ylabel("Margin"); plt.title(title)
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(path, fname), dpi=200); plt.close()

# =============== Early Decision Layer ===============
def early_decision_layer(
    res,
    margin_thresh: float = 0.0,
    use_tuned: bool = False,   # False -> RAW, True -> TUNED (if available)
    use_gold: bool = False,    # False -> top1–top2, True -> gold-margin (options)
    persist_k: int = 1
):
    series = "tuned" if use_tuned and (res.get("tuned") is not None) else "raw"
    winners = res[series][1]
    margins_top1 = res[series][2]
    gold_margins = res[series][3] if len(res[series]) > 3 else None

    if use_gold:
        if not gold_margins: return None
        margins = gold_margins; metric = "gold"
    else:
        margins = margins_top1; metric = "top1_top2"

    final_w = winners[-1]
    L = len(winners)
    for i in range(L):
        ok = True
        for j in range(i, min(i + persist_k, L)):
            if winners[j] != final_w or margins[j] is None or margins[j] < margin_thresh:
                ok = False; break
        if ok:
            return {
                "idx": i, "winner_final": final_w, "winner_at_i": winners[i],
                "margin_at_i": margins[i], "series": series, "metric": metric,
                "persist_k": persist_k, "threshold": margin_thresh,
            }
    return None


# --- add to logit_lens.py ---
import torch
from typing import List, Dict

@torch.no_grad()
def mcq_alllayer_hiddens(
    model,
    tokenizer,
    prompt_text: str,
    options: List[str],
    *,
    pos: int = -1,
    skip_embedding: bool = True,
    add_space_between: bool = True,
):
    """
    Return per-layer hidden vectors for each MCQ option.

    Args:
        model, tokenizer: HF causal LM & tokenizer
        prompt_text: the question/prompt (string)
        options: list of candidate answers (strings)
        pos: kept for API symmetry with your code; we always read the *last token*
             of (prompt+option). 'pos' is ignored here but retained for compatibility.
        skip_embedding: if True, layer 0 in the output corresponds to the first transformer block
                        (i.e., skip the embedding hidden_states[0]).
        add_space_between: prepend a space before option if needed (useful for GPT-2 BPE).

    Returns:
        layer_dicts: List[Dict[str, torch.Tensor]]
            layer_dicts[layer_idx][option] -> 1D tensor [hidden_dim] (on CPU)
    """
    device = next(model.parameters()).device

    # Ensure we get hidden states
    # Either set globally: model.config.output_hidden_states = True
    # or request per-call:
    model_wants_hidden = getattr(model.config, "output_hidden_states", False)
    if not model_wants_hidden:
        # Temporarily call with flag; safest is to pass the kwarg in forward.
        pass

    # Build batched inputs: concat prompt + option for each candidate
    merged_texts = []
    for opt in options:
        sep = " " if add_space_between and (len(opt) > 0 and not opt.startswith((" ", "\n"))) else ""
        merged_texts.append(prompt_text + sep + opt)

    enc = tokenizer(
        merged_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    # Forward (batched) with hidden states
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple: [emb, h1, ..., hL]

    start = 1 if skip_embedding else 0
    L = len(hs) - start
    B, T = enc["input_ids"].shape

    # For each sequence, find index of the last non-pad token
    # (right padding assumed by default; tokenizer.padding_side may vary)
    if tokenizer.pad_token_id is None:
        # fallback: assume last position is the last token
        last_idx = torch.full((B,), T - 1, dtype=torch.long, device=device)
    else:
        pad_id = tokenizer.pad_token_id
        # mask of non-pad
        nonpad = (enc["input_ids"] != pad_id).int()
        # argmax over reversed to find last non-pad
        # last_index = T-1 - argmax(reverse(nonpad)==1)
        rev_idx = torch.flip(torch.arange(T, device=device), dims=[0])
        # compute last_nonpad via cumulative technique
        # simpler: for each row get last where nonpad==1
        last_idx = nonpad.argmax(dim=1)  # WRONG if left padding. Handle both sides:
        # Robust last index regardless of padding side:
        last_idx = (nonpad * torch.arange(T, device=device)).max(dim=1).values

    # Build output structure: one dict per layer
    layer_dicts: List[Dict[str, torch.Tensor]] = [dict() for _ in range(L)]

    # Fill per option
    for b, opt in enumerate(options):
        li = int(last_idx[b].item())
        for layer_i in range(start, len(hs)):
            # hs[layer_i]: [B, T, D]
            vec = hs[layer_i][b, li].detach().cpu()
            layer_dicts[layer_i - start][opt] = vec

    return layer_dicts

# --- add to logit_lens.py ---
import torch
from typing import List, Dict

@torch.no_grad()
def single_alllayer_hiddens(
    model,
    tokenizer,
    prompt_texts: List[str],
    completions: List[str],
    *,
    skip_embedding: bool = True,
    add_space_between: bool = True,
):
    """
    Batched hidden states for a list of (prompt, completion) pairs.
    Returns:
        layer_h: List[torch.Tensor] with length = n_layers (emb skipped if skip_embedding)
                 Each element is a tensor of shape [B, D], where B=len(prompt_texts),
                 holding the hidden vector at the last token of (prompt+completion).
    """
    device = next(model.parameters()).device

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    merged = []
    for p, c in zip(prompt_texts, completions):
        sep = " " if add_space_between and (len(c) > 0 and not c.startswith((" ", "\n"))) else ""
        merged.append(p + sep + c)

    enc = tokenizer(
        merged, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # [emb, h1, ..., hL]

    start = 1 if skip_embedding else 0
    T = enc["input_ids"].shape[1]

    # last non-pad idx per row, robust to left/right padding
    ids = enc["input_ids"]
    ar = torch.arange(T, device=ids.device).unsqueeze(0).expand_as(ids)
    mask = (ids != tokenizer.pad_token_id)
    last_idx = (ar * mask).max(dim=1).values  # [B]

    layer_h = []
    for i in range(start, len(hs)):
        # gather per row last token vector
        H = hs[i]  # [B, T, D]
        B, _, D = H.shape
        vecs = H[torch.arange(B, device=H.device), last_idx]  # [B, D]
        layer_h.append(vecs.detach().cpu())
    return layer_h  # length L, each [B, D]
