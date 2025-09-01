import argparse
from itertools import combinations
import torch
from sklearn.metrics.pairwise import cosine_similarity
from util import *


# -----------------------------
# Evaluation helpers
# -----------------------------
def eval_prompts(prompts, model, tokenizer, vec_pooling="content_mean", layer=-4, k=4, show_gen=False):
    vecs = [get_vec(p, model, tokenizer, layer=layer, fuse_last_k=k, pooling=vec_pooling) for p in prompts]
    print(f"\n== Similarity by {vec_pooling} (layer={layer}, fuse_last_k={k}) ==")
    for i, j in combinations(range(len(prompts)), 2):
        cos = cosine_similarity([vecs[i]], [vecs[j]])[0, 0]
        kl_val = kl_p_ab(prompts[i], prompts[j], model, tokenizer)
        print(f"pair ({i},{j})  cosine={cos:.3f}   KL(P{i}||P{j})={kl_val:.3f}")

    if show_gen:
        print("\n-- Deterministic next-token (argmax) + Sampled continuation --")
        for idx, p in enumerate(prompts):
            ia = tokenizer(p, return_tensors="pt").to(next(model.parameters()).device)
            with torch.no_grad():
                logits = model(**ia).logits[:, -1, :]
                top_id = int(torch.argmax(logits, dim=-1))
            print(f"[{idx}] prompt: {p}")
            print(f"  next-token (argmax): {tokenizer.decode([top_id])}")
            print(f"  sampled: {generate_answer(p, model, tokenizer, max_new=30)}\n")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-medium", help="e.g., distilgpt2, gpt2, gpt2-medium")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print("Device:", device)

    model, tokenizer = load_model_and_tokenizer(args.model, device)

   
    #prompts = read_file_lines("prompr.txt")
    #if(prompts is None or len(prompts)==0):
        # Prompts: decision-heavy + control
    prompts = [
            "You see your friend cheating on an exam. Do you report it or stay quiet?",
            "A self-driving car must choose between hitting one person or swerving and hitting five. What should it do?",
            "If a colleague steals credit for your work, do you confront them or stay silent?",
            "You find a lost credit card. Do you return it to the bank or try to use it?",
            "The cat is sleeping on the sofa.",  # control (non-decision)
        ]
    labels = ["decision","decision","decision","decision","control"]

    # --- Similarity & KL (no noise) ---
    eval_prompts(prompts, model, tokenizer, vec_pooling="content_mean", layer=-4, k=4, show_gen=True)
    eval_prompts(prompts, model, tokenizer, vec_pooling="mean", layer=-1, k=1, show_gen=False)
    eval_prompts(prompts, model, tokenizer, vec_pooling="content_last", layer=-2, k=2, show_gen=False)
    eval_prompts(prompts, model, tokenizer, vec_pooling="last_token", layer=-2, k=2, show_gen=False)
   

    pca_prompts(prompts, model, tokenizer,
            layer=-2, fuse_last_k=4, pooling="content_mean",
            labels=labels, annotate=True)
    
    
    pca_prompts(prompts, model, tokenizer,
            layer=-6, fuse_last_k=4, pooling="content_mean",
            labels=labels, annotate=True)
    
    
    pca_prompts(prompts, model, tokenizer,
            layer=-10, fuse_last_k=4, pooling="content_mean",
            labels=labels, annotate=True)
    


    pca_prompts(prompts, model, tokenizer,
            layer=-2, fuse_last_k=4, pooling="content_last",
            labels=labels, annotate=True)
    
    
    pca_prompts(prompts, model, tokenizer,
            layer=-6, fuse_last_k=4, pooling="content_last",
            labels=labels, annotate=True)

    
    pca_prompts(prompts, model, tokenizer,
            layer=-10, fuse_last_k=4, pooling="content_last",
            labels=labels, annotate=True)
    

    
    

#if __name__ == "__main__":
   # main()
   # تست

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2-medium", help="e.g., distilgpt2, gpt2, gpt2-medium")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = get_device()
print("Device:", device)
model, tokenizer = load_model_and_tokenizer(args.model, device)
test_prompt = "You see your friend cheating on an exam. Do you report it or stay quiet?"


base_txt = generate_stepwise_baseline(test_prompt,model,tokenizer ,max_new=60)
pert_txt = generate_stepwise_perturbed(test_prompt,model,tokenizer , max_new=60, inject_steps=30, sigma=0.8)

print(generate_with_logits_perturb(test_prompt,model,tokenizer , max_new=50))
print("\nBASE:\n", base_txt[:300], "\n")
print("PERTURBED:\n", pert_txt[:300], "\n")