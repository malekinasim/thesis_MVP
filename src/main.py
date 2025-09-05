import argparse
from itertools import combinations
import torch
from sklearn.metrics.pairwise import cosine_similarity
from util import *
import numpy as np
import os
import pandas as pd
from itertools import combinations
from xlsxwriter import Workbook



def eval_prompts(prompts, labels, model, tokenizer,
                          vec_pooling="content_mean", layer=-4, k=4,
                          show_gen=False, out_dir="out_doc", excel_name=None):    
    vecs= []
    for idx,p in enumerate(prompts):
         vec,l_start,l_end = get_vec(p, model, tokenizer, layer=layer, fuse_last_k=k, pooling=vec_pooling)
         vecs.append(vec)
  
    os.makedirs(out_dir, exist_ok=True)
    if excel_name is None:
        excel_name = f"eval_pairs_layer[{l_start}-{l_end}]_pooling_{vec_pooling}.xlsx"
    excel_path = os.path.join(out_dir, excel_name)

    X = np.vstack(vecs)

    pair_rows = []
    for i, j in combinations(range(len(prompts)), 2):
        cos = float(cosine_similarity([X[i]], [X[j]])[0, 0])
        kl_val = float(kl_p_ab(prompts[i], prompts[j], model, tokenizer))
        pair_rows.append({
            "i": i, "j": j,
            "label_i": labels[i], "label_j": labels[j],
            "prompt_i": prompts[i], "prompt_j": prompts[j],
            "cosine": cos,
            "KL(Pi||Pj)": kl_val
        })
    df_pairs = pd.DataFrame(pair_rows)

    df_gen = None
    if show_gen:
        rows = []
        device = next(model.parameters()).device
        for idx, p in enumerate(prompts):
            ia = tokenizer(p, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**ia).logits[:, -1, :]
                top_id = int(torch.argmax(logits, dim=-1))
            rows.append({
                "id": idx,
                "label": labels[idx],
                "prompt": p,
                "next_token_argmax": tokenizer.decode([top_id]),
                "sampled_continuation": generate_answer(p, model, tokenizer, max_new=30)
            })
        df_gen = pd.DataFrame(rows)

 
    df_meta = pd.DataFrame([{
        "layer_start": l_start,
        "layer_end": l_end,
        "pooling": vec_pooling,
        "n_prompts": len(prompts),
        "n_pairs": len(df_pairs)
    }])
    
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_pairs.to_excel(writer, sheet_name="pairs", index=False)
        if df_gen is not None:
            df_gen.to_excel(writer, sheet_name="generations", index=False)
        df_meta.to_excel(writer, sheet_name="meta", index=False)

    print(f"[OK] Excel saved -> {excel_path}")

def prompts_instability(prompts, labels, model, tokenizer,
                                 vec_pooling="content_mean", layer=-4, k=4,
                                 out_dir="out_doc", excel_name=None, also_print=False):
   
    os.makedirs(out_dir, exist_ok=True)
    if excel_name is None:
        excel_name = f"instability_layer{layer}_k{k}_pooling_{vec_pooling}.xlsx"
    excel_path = os.path.join(out_dir, excel_name)

    instab,l_start,l_end  = instability_score(prompts, model, tokenizer,
                               vec_pooling=vec_pooling, layer=layer, k=k)
    df_inst = pd.DataFrame({
        "id": range(len(prompts)),
        "label": [str(l).lower() for l in labels],
        "prompt": prompts,
        "instability": instab
    })

    df_summary = (df_inst
                  .groupby("label")["instability"]
                  .agg(["count", "mean", "median", "std"])
                  .reset_index())
    
    df_meta = pd.DataFrame([{
        "layer_start": l_start,
        "layer_end": l_end,
        "pooling": vec_pooling,
        "n_prompts": len(prompts)
    }])

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_inst.to_excel(writer, sheet_name="instability", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_meta.to_excel(writer, sheet_name="meta", index=False)

    if also_print:
        for _, r in df_inst.iterrows():
            print(f"[{r['label']}]  instability={r['instability']:.4f} | {r['prompt']}")
       
        dec = df_inst[df_inst["label"]=="decision"]["instability"].values
        ctl = df_inst[df_inst["label"]=="control"]["instability"].values
        if len(dec) and len(ctl):
            print(f"\nMean instability | decision={np.mean(dec):.4f}  control={np.mean(ctl):.4f}")

    print(f"[OK] Excel saved -> {excel_path}")

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
    prompts ,labels= load_prompts_from_json("prompt.json")


    # --- Similarity & KL (no noise) ---
    #eval_prompts(prompts,labels, model, tokenizer, vec_pooling="content_mean", layer=-4, k=4, show_gen=True)
    #eval_prompts(prompts, labels,model, tokenizer, vec_pooling="mean", layer=-1, k=1, show_gen=False)
    #eval_prompts(prompts, labels,model, tokenizer, vec_pooling="content_last", layer=-2, k=2, show_gen=False)
    #eval_prompts(prompts, labels,model, tokenizer, vec_pooling="last_token", layer=-2, k=2, show_gen=False)
   

    #pca_prompts(prompts, model, tokenizer,layer=-2, fuse_last_k=4, pooling="content_mean",labels=labels, annotate=True)
    #pca_prompts(prompts, model, tokenizer,layer=-2, fuse_last_k=4, pooling="content_last",labels=labels, annotate=True)
    
    #prompts_instability(prompts, labels, model, tokenizer,vec_pooling="content_mean", layer=-4, k=4, also_print=False)
 
    test_prompt = prompts[random.randint(0,len(prompts)-1)]
    inject_start=30
    inject_end=38
    n_layers = model.config.n_layer
    target_layer = min(19, n_layers - 1)


    print("Test Prompts:\n", test_prompt, "\n")
    base_txt,base_logits = generate_stepwise_baseline(test_prompt,model,tokenizer ,max_new=70)

    pert_hiddenState_txt,pert_hiddenState_logits= generate_stepwise_perturbed(test_prompt,model,tokenizer , max_new=70, inject_step_start=inject_start,
                                                                inject_step_end=inject_end,sigma=100,last_k=4,layer=target_layer,use_direction=False)
    pert_logit_text,pert_logits=generate_with_logits_perturb(test_prompt,model,tokenizer , max_new=60)
    print("BASE:\n", base_txt[:300], "\n")
    print("PERTURBED hidden state :\n", pert_hiddenState_txt[:300], "\n")
    for t in range(inject_start-3, inject_end+4):
        kl = kl_from_logits(base_logits[t], pert_hiddenState_logits[t])
        mark = " <== injected window" if inject_start <= t <= inject_end else ""
        print(f"t={t:02d}  KL(base||pert_hiddenState)={kl:.4f}{mark}")


    print("PERTURBED logit:\n", pert_logit_text[:300], "\n")

    for t in range(inject_start-3, inject_end+4):
        kl = kl_from_logits(base_logits[t], pert_logits[t])
        mark = " <== injected window" if kl<=0.3 else ""
        print(f"t={t:02d}  KL(base||pert_logit)={kl:.4f}{mark}")


    base_mat = to_np_matrix(base_logits)
    pertH_mat = to_np_matrix(pert_hiddenState_logits)
    pertL_mat = to_np_matrix(pert_logits)

    pca_logits([base_mat, pertH_mat, pertL_mat],
            labels=["BASE", "PERT(hidden)", "PERT(logits)"],
            path="fig")
            
    
if __name__ == "__main__":
    main()

