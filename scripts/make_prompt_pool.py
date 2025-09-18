# make_mcq_pool.py
import json, random

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from util import load_model_and_tokenizer, get_device

COUNTRIES = [
    ("France"," Paris"," London"," Berlin"),
    ("Germany"," Berlin"," Paris"," Rome"),
    ("Italy"," Rome"," Madrid"," Berlin"),
    ("Spain"," Madrid"," Rome"," Lisbon"),
    ("Japan"," Tokyo"," Osaka"," Kyoto"),
    ("Canada"," Ottawa"," Toronto"," Montreal"),
    ("Brazil"," Brasilia"," Rio"," Sao Paulo"),
    ("India"," New Delhi"," Mumbai"," Kolkata"),
    ("China"," Beijing"," Shanghai"," Shenzhen"),
    ("Australia"," Canberra"," Sydney"," Melbourne"),
    ("Iran"," Tehran"," Isfahan"," Shiraz"),
    ("Turkey"," Ankara"," Istanbul"," Izmir"),
    ("Russia"," Moscow"," Saint Petersburg"," Kazan"),
    ("Egypt"," Cairo"," Alexandria"," Giza"),
    ("Mexico"," Mexico City"," Guadalajara"," Monterrey"),
]

def is_single_token(tokenizer, s):
    ids = tokenizer.encode(s, add_special_tokens=False)
    return len(ids) == 1

def build_mcq_items(tokenizer, n_math=400, seed=0):
    random.seed(seed)
    items = []

    # 1) Country capitals
    for c, a, b, d in COUNTRIES * 20: 
        opts = [a,b,d]
        if all(is_single_token(tokenizer, o) for o in opts):
            items.append({
                "id": f"mcq-cap-{c.lower()}-{random.randint(0,9999)}",
                "task": "mcq",
                "prompt": f"The capital of {c} is ",
                "options": opts,
                "gold": a
            })
    for c, a ,_,_ in COUNTRIES * 20: 
        if all(is_single_token(tokenizer, o) for o in opts):
            items.append({
                "id": f"singel-cap-{c.lower()}-{random.randint(0,9999)}",
                "task": "singel",
                "prompt": f"The capital of {c} is ",
                "gold": a
            })

    # 2) Simple math additions (ensure single-token options)
    for _ in range(n_math):
        x = random.randint(2, 60); y = random.randint(2, 60)
        gold = f" {x+y}"
        decoys = [f" {x+y+1}", f" {x+y-1}", f" {x+y+2}"]
        opts = [gold] + decoys
        random.shuffle(opts)
        if all(is_single_token(tokenizer, o) for o in opts):
            items.append({
                "id": f"mcq-math-{x}-{y}-{random.randint(0,9999)}",
                "task": "mcq",
                "prompt": f"{x} + {y} = ",
                "options": opts,
                "gold": gold
            })

    for _ in range(n_math):
        x = random.randint(2, 60); y = random.randint(2, 60)
        gold = f" {x+y}"
        opts = [gold] + decoys
        random.shuffle(opts)
        if all(is_single_token(tokenizer, o) for o in opts):
            items.append({
                "id": f"singel-math-{x}-{y}-{random.randint(0,9999)}",
                "task": "singel",
                "prompt": f"{x} + {y} = ",
                "gold": gold
            })

    random.shuffle(items)
    return items

if __name__ == "__main__":
    device = get_device()
    model, tokenizer = load_model_and_tokenizer("gpt2-medium", device)
    mcqs = build_mcq_items(tokenizer, n_math=800, seed=0) 
    with open("data/prompt_pool.json","w",encoding="utf-8") as f:
        json.dump(mcqs, f, ensure_ascii=False, indent=2)
    print("Saved mcq_pool.json with", len(mcqs), "items")
