# make_prompts_for_tuned.py
import json, random

def gen_templates(n_each=200, seed=0):
    random.seed(seed)
    countries = ["France","Germany","Italy","Spain","Japan","Canada","Brazil","India","China","Australia","Iran","Turkey","Russia","Egypt","Mexico","Netherlands","Sweden","Norway","Poland","Greece","Portugal","Denmark","Finland","Ireland","Belgium","Austria","Switzerland","South Korea","Argentina"]
    topics = ["mathematics","physics","history","biology","computer science","economics","philosophy","music","art","sports","geography","chemistry","literature"]
    verbs  = ["explain","describe","summarize","compare","contrast","list","outline"]
    names  = ["Alice","Bob","Carol","Dave","Eve","Mallory","Trent","Peggy","Sara","Omid","Neda","Reza"]
    funcs  = ["sort","search","reverse","sum","average","max","min","count","filter","map","reduce"]

    prompts = []


    for _ in range(n_each):
        c = random.choice(countries)
        prompts.append(f"The capital of {c} is ")
        prompts.append(f"Question: What is the capital of {c}? Answer: ")

    # 2) سال/رویداد
    years = list(range(1990, 2025))
    for _ in range(n_each):
        y = random.choice(years); t = random.choice(topics)
        prompts.append(f"In {y}, an important event in {t} was ")
        prompts.append(f"By {y}, the main idea in {t} had become ")

   
    for _ in range(n_each):
        t = random.choice(topics); v = random.choice(verbs)
        prompts.append(f"{v.capitalize()} the core concept of {t} in one sentence: ")

    
    for _ in range(n_each):
        a, b = random.sample(names, 2)
        prompts.append(f"{a}: Hi {b}, can you summarize the key point?\n{b}: ")

  
    for _ in range(n_each):
        a = random.randint(2, 50); b = random.randint(2, 50)
        prompts.append(f"Compute {a} + {b} = ")
        prompts.append(f"Compute {a} - {b} = ")

    
    for _ in range(n_each):
        f = random.choice(funcs)
        prompts.append(f"In Python, write a one-liner to {f} a list: ")

    

    # یکتا سازی
    seen = set(); uniq = []
    for p in prompts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p); uniq.append(p)
    return uniq

if __name__ == "__main__":
    prompts = gen_templates(n_each=250, seed=0)  # ~3000 پرامپت
    with open("data\prompts_for_tuned.json","w",encoding="utf-8") as f:
        json.dump([{"prompt":p} for p in prompts], f, ensure_ascii=False, indent=2)
    print("Saved prompts_for_tuned.json with", len(prompts), "items")
