import argparse
from itertools import combinations
import torch
from sklearn.metrics.pairwise import cosine_similarity
from util import *
import numpy as np
import os
import pandas as pd
from itertools import combinations
import random
# -----------------------------
# Main
# main.py
import argparse, os, random, torch
import numpy as np
import pandas as pd

from util import *  # توابع قبلی خودت + load_prompts_with_options
from svcca import *
from logit_lens import (
    TunedDiag,
    compute_margins_per_layer,
    save_csv_margins,
    save_plot_margins,
    mcq_alllayer_scores,
    save_perlayer_csv_both,
    plot_perlayer_margins_both,
    early_decision_layer,
    mcq_alllayer_hiddens, single_alllayer_hiddens
)




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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-medium", help="e.g., distilgpt2, gpt2, gpt2-medium")
    parser.add_argument("--seed", type=int, default=0)

    # --- logit-lens args (additive) ---
    parser.add_argument("--task", type=str, default="decision_control_separation",
    choices=["decision_control_separation","Pertuted_hidden_state", "logit_lens", "massmean_mcq", "massmean_single", "massmean_combined"])

    parser.add_argument("--dataset", type=str, default="propmp_lenz_logit.json",
                        help="Dataset path for MCQ/single items")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Free prompt for full-vocab lens (optional)")
    parser.add_argument("--gold", type=str, default=None,
                        help="Gold token text (single-token). For full-vocab lens.")
    parser.add_argument("--pos", type=int, default=-1,
                        help="Position for next-token prediction (usually -1).")
    parser.add_argument("--lens", type=str, default="last_only",
                        choices=["raw", "last_only", "all"],
                        help="Apply ln_f: raw=none, last_only=only last layer, all=every layer (exp).")
    parser.add_argument("--skip_embedding", action="store_true", help="Drop embedding row from layer list.")
    parser.add_argument("--mcq_idx", type=int, default=0, help="Which MCQ item to run from dataset.")
    parser.add_argument("--margin_thresh", type=float, default=0.0, help="Threshold for early-decision layer.")
    parser.add_argument("--tuned_json", type=str, default=None, help="Path to tuned-lens diagonal weights JSON.")
    parser.add_argument("--debug", action="store_true", help="Verbose debug logs & extra checks.")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = get_device()
    print("Device:", device)

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    model.config.output_hidden_states = True
    if args.task == "massmean_mcq":
        # Load MCQ + single using your util
        mcq_items, free_items = load_prompts_with_options(args.dataset, tokenizer, require_single_token=False)
        # Keep only MCQs that have a gold label & at least 2 options
        mcq_items = [it for it in mcq_items if it.get("gold") in it.get("options", []) and len(it.get("options", [])) >= 2]
        if not mcq_items:
            print("[MassMean] No valid MCQ items with gold found in dataset.")
            return
        train, test = split_train_test(mcq_items, test_ratio=0.3, seed=args.seed)

        # Fit per-layer mass-mean
        from logit_lens import mcq_alllayer_hiddens
        W = mass_mean_fit_per_layer(train, model, tokenizer, mcq_alllayer_hiddens, pos=args.pos)

        # Evaluate per-layer; pick best
        per_layer_acc, best_layer, best_acc = mass_mean_eval_per_layer(
            test, W, model, tokenizer, mcq_alllayer_hiddens, pos=args.pos
        )

        # Save CSV
        os.makedirs("out_doc", exist_ok=True)
        pd.DataFrame([
            {"layer": li, "acc": acc, "n_test": len(test)}
            for li, acc in per_layer_acc.items()
        ]).to_csv(os.path.join("out_doc", "massmean_mcq_perlayer_acc.csv"), index=False)

        print("[MassMean] per-layer accuracy:", per_layer_acc)
        print(f"[MassMean] best_layer={best_layer}  best_acc={best_acc:.3f}")
        return

    if args.task == "massmean_single":
        # load dataset with your existing loader
        mcq_items, single_items = load_prompts_with_options(args.dataset, tokenizer, require_single_token=False)

        # build pool of all golds (for impostor negatives if needed)
        all_golds = []
        for it in single_items:
            if it.get("gold"): all_golds.append(it["gold"])
        for it in mcq_items:
            if it.get("gold"): all_golds.append(it["gold"])

        pairs = build_single_pairs(single_items, all_gold_pool=all_golds, negatives_per=2)
        if not pairs:
            print("[MassMean-Single] No single items found.")
            return
        train_pairs, test_pairs = split_train_test(pairs, test_ratio=0.3, seed=args.seed)

        W = mass_mean_fit_from_pairs(train_pairs, model, tokenizer, single_alllayer_hiddens)
        per_layer_acc, best_layer, best_acc = eval_binary_from_pairs(
            test_pairs, W, model, tokenizer, single_alllayer_hiddens
        )

        os.makedirs("out_doc", exist_ok=True)
        pd.DataFrame([{"layer": li, "acc": acc} for li, acc in per_layer_acc.items()]) \
            .to_csv(os.path.join("out_doc", "massmean_single_perlayer_acc.csv"), index=False)
        print("[MassMean-Single] per-layer acc:", per_layer_acc)
        print(f"[MassMean-Single] best_layer={best_layer}  best_acc={best_acc:.3f}")
        return


    if args.task == "massmean_combined":
        # 1) MCQ positives/negatives → pairs
        mcq_items_all, single_items = load_prompts_with_options(args.dataset, tokenizer, require_single_token=False)
        mcq_items = [it for it in mcq_items_all if it.get("gold") in it.get("options", []) and len(it.get("options", []))>=2]
        mcq_pairs = []
        for it in mcq_items:
            p = it["prompt"]; gold = it["gold"]; gid = it.get("id")
            # gold as positive
            mcq_pairs.append((p, gold, 1, gid))
            # others as negatives
            for opt in it["options"]:
                if opt == gold: continue
                mcq_pairs.append((p, opt, 0, gid))

        # 2) Single pairs
        all_golds = [it.get("gold") for it in mcq_items if it.get("gold")] + \
                    [it.get("gold") for it in single_items if it.get("gold")]
        single_pairs = build_single_pairs(single_items, all_gold_pool=all_golds, negatives_per=2)

        pairs = mcq_pairs + single_pairs
        random.shuffle(pairs)
        if not pairs:
            print("[MassMean-Combined] No data!")
            return

        train_pairs, test_pairs = split_train_test(pairs, test_ratio=0.3, seed=args.seed)

        # Fit W on combined pairs using single_alllayer_hiddens (works for both; it's prompt+completion)
        W = mass_mean_fit_from_pairs(train_pairs, model, tokenizer, single_alllayer_hiddens)

        # Evaluate (a) binary over test_pairs
        per_layer_acc, best_layer, best_acc = eval_binary_from_pairs(
            test_pairs, W, model, tokenizer, single_alllayer_hiddens
        )

        # (b) MCQ Top-1 using same W
        per_layer_right, per_layer_total = {li:0 for li in W}, {li:0 for li in W}
        for it in mcq_items:
            p = it["prompt"]; opts = it["options"]; gold = it["gold"]
            layer_dicts = mcq_alllayer_hiddens(model, tokenizer, p, opts, pos=args.pos)
            for li, od in enumerate(layer_dicts):
                if li not in W: continue
                w = W[li]
                best_opt, best_s = None, -1e30
                for opt, h in od.items():
                    s = float(torch.dot(w, h))
                    if s > best_s:
                        best_s, best_opt = s, opt
                per_layer_total[li] += 1
                per_layer_right[li] += int(best_opt == gold)
        per_layer_mcq_acc = {li: (per_layer_right[li]/per_layer_total[li] if per_layer_total[li] else 0.0)
                            for li in per_layer_total}

        os.makedirs("out_doc", exist_ok=True)
        pd.DataFrame([{"layer": li, "bin_acc": per_layer_acc.get(li,0.0),
                    "mcq_acc": per_layer_mcq_acc.get(li,0.0)} for li in W.keys()]) \
            .to_csv(os.path.join("out_doc", "massmean_combined_perlayer.csv"), index=False)

        print("[MassMean-Combined] binary acc per layer:", per_layer_acc)
        print("[MassMean-Combined] mcq acc per layer:", per_layer_mcq_acc)
        print(f"[MassMean-Combined] best_layer (binary)={best_layer}  best_acc={best_acc:.3f}")
        return
    # ---------- Task: Logit Lens ----------
    if args.task == "logit_lens":
        # Tuned lens (optional)
        tuned = None
        if args.tuned_json:
            tuned = TunedDiag.from_json(args.tuned_json, device=device)

        # Load dataset (MCQ + single)
        mcq_items, free_items = load_prompts_with_options(args.dataset, tokenizer, require_single_token=True)

        # A) MCQ branch (options + optional gold among options)
        if len(mcq_items) > 0:
            if(args.mcq_idx):
              i = max(0, min(args.mcq_idx, random.randint( len(mcq_items)-1)))
            else:
              i = random.randint( 0,len(mcq_items)-1)

            item = mcq_items[i]
            prompt_text = item["prompt"]
            options     = item["options"]
            gold_opt    = item.get("gold", None)
            pos         = item.get("pos", args.pos)

            print(f"[MCQ] id={item.get('id','?')}  prompt={prompt_text!r}  options={options}  gold={gold_opt}")

            res = mcq_alllayer_scores(
                model, tokenizer, prompt_text, options, gold_opt=gold_opt,
                pos=pos, ln_f_mode=args.lens, skip_embedding=True, tuned=tuned
            )
            os.makedirs("out_doc", exist_ok=True); os.makedirs("fig", exist_ok=True)
            base = item.get("id","mcq")
            save_perlayer_csv_both(res, options, out_dir="out_doc", fname=f"{base}_perlayer_margins.csv")
            plot_perlayer_margins_both(res, out_png=f"fig/{base}__margins_per_layer_both.png" , 
                                        title=f"Per-layer margins (raw vs tuned): {base}")
            print("[OK] MCQ CSV  -> out_doc/mcq_perlayer_margins.csv")
            print(f"[OK] MCQ plot -> fig/Per-layer margins (raw vs tuned): {base}.png")
            
            # Early decision layer (using top1–top2 on RAW; tweak flags if needed)
            info = early_decision_layer(res, margin_thresh=args.margin_thresh, use_tuned=False, use_gold=False, persist_k=2)
            print("[Early Decision Layer]", info)

        # B) Free-prompt full-vocab lens (single gold token)
        # If --prompt given on CLI OR dataset has 'task':'single'
        did_any_single = False
        if args.prompt and args.prompt.strip():
            margins = compute_margins_per_layer(
                model, tokenizer, text=args.prompt, pos=args.pos,
                ln_f_mode=args.lens, skip_embedding=True,
                gold_text=args.gold, options=None, gold_option=None, tuned=tuned
            )
            save_csv_margins(margins, out_dir="out_doc", fname="margins_per_layer_free.csv")
            save_plot_margins(margins, path="fig", fname="margins_per_layer_free.png",
                              title=f"Layer-wise Margins (full vocab)")
            print("[OK] Free CSV -> out_doc/margins_per_layer_free.csv")
            print("[OK] Free Fig -> fig/margins_per_layer_free.png")
            did_any_single = True

        single_items = [it for it in free_items if  "gold" in it]
        if len(single_items)>0:

            if(args.mcq_idx):
              i = max(0, min(args.mcq_idx, random.randint( len(single_items)-1)))
            else:
              i = random.randint(0, len(single_items)-1)
            it=single_items[i]
            margins = compute_margins_per_layer(
                model, tokenizer,
                text=it["prompt"], pos=it.get("pos", args.pos),
                ln_f_mode=args.lens, skip_embedding=True,
                gold_text=it["gold"], options=None, gold_option=None, tuned=tuned
            )
            base = it.get("id","single")
            save_csv_margins(margins, out_dir="out_doc", fname=f"{base}_margins_full.csv")
            save_plot_margins(margins, path="fig", fname=f"{base}_margins_full.png",
                              title=f"Layer-wise Margins (full vocab): {base}")
            did_any_single = True

        if not (len(mcq_items) or did_any_single):
            print("[LogitLens] No MCQ or single items found. Provide --prompt/--gold or use the MCQ dataset.")
        return

    if(args.task =='decision_control_separation'):
        prompts ,labels= load_prompts_from_json("prompt.json")

        eval_prompts(prompts,labels, model, tokenizer, vec_pooling="content_mean", layer=-4, k=4, show_gen=True)
        eval_prompts(prompts, labels,model, tokenizer, vec_pooling="mean", layer=-1, k=1, show_gen=False)
        eval_prompts(prompts, labels,model, tokenizer, vec_pooling="content_last", layer=-2, k=2, show_gen=False)
        eval_prompts(prompts, labels,model, tokenizer, vec_pooling="last_token", layer=-2, k=2, show_gen=False)

        pca_prompts(prompts, model, tokenizer,layer=-2, fuse_last_k=4, pooling="content_mean",labels=labels, annotate=True)
        pca_prompts(prompts, model, tokenizer,layer=-2, fuse_last_k=4, pooling="content_last",labels=labels, annotate=True)

        prompts_instability(prompts, labels, model, tokenizer,vec_pooling="content_mean", layer=-4, k=4, also_print=False)

    if(args.task =='Pertuted_hidden_state'):
        prompts ,labels= load_prompts_from_json("prompt.json")
        test_prompt = prompts[random.randint(0,len(prompts)-1)]
        inject_start=30
        inject_end=38
        n_layers = model.config.n_layer
        target_layer = min(19, n_layers - 1)

        print("Test Prompts:\n", test_prompt, "\n")
        base_txt,base_logits = generate_stepwise_baseline(test_prompt,model,tokenizer ,max_new=70)

        pert_hiddenState_txt,pert_hiddenState_logits= generate_stepwise_perturbed(
            test_prompt,model,tokenizer , max_new=70,
            inject_step_start=inject_start, inject_step_end=inject_end,
            sigma=100,last_k=4,layer=target_layer,use_direction=False
        )
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

        base =prompts[random.randint(0,len(prompts)-1)]
        pert = base + " Please think step by step."
        calc_svcca_between_prompts(base,pert,model, tokenizer,fuse_last_k=4,start_layer=-12,end_layer=-1,
                                energy=0.99,out_doc_path='out_doc',fig_path='fig')

if __name__ == "__main__":
    main()