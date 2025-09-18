# mass_mean_probe.py
import json, random, math
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------
# Config
# --------------------
DATA_PATH = "../data/prompt_pool.json"   # فایل ارسالی تو
MODEL_NAME = "gpt2"              # اگر مدل بزرگ‌تری داری اینو عوض کن (مثلاً مسیر لوکال)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

# --------------------
# IO
# --------------------
data = json.loads(Path(DATA_PATH).read_text(encoding="utf-8"))

# --------------------
# Model
# --------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# --------------------
# Helpers
# --------------------
def get_last_token_hidden(prompt: str, cont: str):
    """
    برمی‌گرداند hidden state توکن آخرِ continuation
    وقتی prompt+cont را به مدل می‌دهیم.
    """
    text = prompt + cont
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    input_ids = enc["input_ids"].to(DEVICE)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    hs = out.hidden_states[-1][0]  # آخرین لایه، batch=1
    # محدوده توکن‌های continuation:
    # طول prompt:
    len_prompt = len(tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN)["input_ids"][0])
    last_idx = min(hs.size(0)-1, len_prompt + len(tok(cont)["input_ids"][0]) - 1)
    return hs[last_idx].detach().cpu().numpy()

def corrupt_numeric_answer(ans: str):
    # تولید یک جوابِ عددیِ غلطِ ساده (±1 یا ±2؛ اگر عدد بزرگ بود ±(1..3))
    try:
        v = int(ans.strip())
        delta = random.choice([1, -1, 2, -2, 3, -3])
        if delta == 0: delta = 1
        return f" {v + delta}"
    except:
        # اگر عدد نبود، یک جایگزین ساده
        return " 0"

def build_pairs_mcq(items):
    H, y, groups = [], [], []  # groups برای اینکه هر MCQ یک سؤال است
    for it in items:
        if it.get("task") != "mcq": continue
        prompt = it["prompt"]
        gold = it["gold"]
        for opt in it["options"]:
            h = get_last_token_hidden(prompt, opt)
            H.append(h)
            y.append(1 if opt == gold else 0)
            groups.append(it["id"])
    return np.stack(H), np.array(y), np.array(groups)

def build_pairs_single(items, negatives_per=2):
    H, y, groups = [], [], []
    for it in items:
        if it.get("task") != "singel": continue  # (تایپ فایل singel است)
        prompt = it["prompt"]
        gold = it["gold"]
        # مثبت
        h_pos = get_last_token_hidden(prompt, gold)
        H.append(h_pos); y.append(1); groups.append(it["id"])
        # منفی‌های مصنوعی
        for _ in range(negatives_per):
            neg = corrupt_numeric_answer(gold)
            if neg == gold: continue
            h_neg = get_last_token_hidden(prompt, neg)
            H.append(h_neg); y.append(0); groups.append(it["id"])
    return np.stack(H), np.array(y), np.array(groups)

# --------------------
# Build dataset
# --------------------
H_mcq, y_mcq, g_mcq = build_pairs_mcq(data)
H_single, y_single, g_single = build_pairs_single(data, negatives_per=2)

H = np.vstack([H_mcq, H_single])
y = np.concatenate([y_mcq, y_single])
groups = np.concatenate([g_mcq, g_single])

print("Shapes:", H.shape, y.shape)

# --------------------
# Mass-Mean probe
# --------------------
mu_T = H[y == 1].mean(axis=0)
mu_F = H[y == 0].mean(axis=0)
w = mu_T - mu_F

def score(H):
    return H @ w  # ضرب داخلی

# --------------------
# Evaluation
# 1) باینری: AUROC روی همهٔ زوج‌ها
# 2) MCQ: انتخاب گزینه با بیشترین score داخل هر سؤال
# --------------------
# 1) Binary AUROC/ACC
s_all = score(H)
auc = roc_auc_score(y, s_all)
acc_bin = accuracy_score(y, (s_all >= 0).astype(int))
print(f"Binary: AUROC={auc:.3f}  Acc(threshold@0)={acc_bin:.3f}")

# 2) MCQ Top-1
# برای هر گروهِ MCQ، گزینه‌ای که بالاترین score را دارد باید همان gold باشد.
from collections import defaultdict
by_q = defaultdict(list)
for h, lab, gid in zip(H_mcq, y_mcq, g_mcq):
    by_q[gid].append((score(h.reshape(1,-1))[0], lab))
top1_correct = 0
for gid, lst in by_q.items():
    lst.sort(key=lambda x: x[0], reverse=True)  # descending by score
    top_is_correct = (lst[0][1] == 1)
    top1_correct += int(top_is_correct)
mcq_acc = top1_correct / max(1, len(by_q))
print(f"MCQ Top-1 Accuracy={mcq_acc:.3f}")

# --------------------
# Optional: تفکیک گزارش برای Single و MCQ
s_mcq = score(H_mcq)
auc_mcq = roc_auc_score(y_mcq, s_mcq)
s_single = score(H_single)
auc_single = roc_auc_score(y_single, s_single)
print(f"AUROC (MCQ)={auc_mcq:.3f} | AUROC (Single)={auc_single:.3f}")
