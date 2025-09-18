import numpy as np
from scipy.linalg import svd
from sklearn.cross_decomposition import CCA
import torch


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def svd_reduce(X, dim=None, energy=None):
    """
    Reduce columns of X using SVD.
    X: (n_samples, d_features)
    Either specify `dim` (int) or `energy` (float in (0,1]) to keep enough singular values.
    Returns: X_reduced (n_samples, r), Vh (r, d_features)
    """
    X = _to_numpy(X)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vh = svd(Xc, full_matrices=False)
    if dim is None:
        if energy is None:
            energy = 0.99
        s2 = (S**2)
        cum = np.cumsum(s2) / np.sum(s2)
        r = int(np.searchsorted(cum, energy) + 1)
    else:
        r = int(dim)
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    X_red = U_r * S_r  # (n_samples, r)
    return X_red, Vh_r


def svcca(X, Y, dim=None, energy=0.99, n_components=None, return_corrs=False):
    """
    SVCCA between two views (n_samples, d1) and (n_samples, d2).
    1) SVD-reduce each view to r1, r2 via `dim` (if given) or `energy`.
    2) CCA on reduced views; return mean canonical correlation (or list if return_corrs).
    """
    Xr, _ = svd_reduce(X, dim=dim, energy=energy)
    Yr, _ = svd_reduce(Y, dim=dim, energy=energy)
    k = min(Xr.shape[1], Yr.shape[1])
    if n_components is None:
        n_components = k
    n_components = min(n_components, k)
    cca = CCA(n_components=n_components, max_iter=5000)
    Xc, Yc = cca.fit_transform(Xr, Yr)
    # compute per-component correlation
    corrs = []
    for i in range(n_components):
        x = Xc[:, i]
        y = Yc[:, i]
        num = (x * y).sum()
        den = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
        corrs.append(float(num / (den + 1e-12)))
    if return_corrs:
        return corrs
    return float(np.mean(corrs))


def hidden_matrix_for_prompt(text, model, tokenizer, layer=-4, fuse_last_k=4, device=None):
    """
    Return fused hidden states matrix for a prompt: (seq_len, hidden_dim)
    Following user's get_vec logic but without token pooling (keep per-token).
    """
    if device is None:
        device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    hs = out.hidden_states  # tuple length = n_layers+1 (incl embedding)
    L = len(hs)
    if layer < 0:
        layer = L + layer
    layer = max(0, min(layer, L - 1))
    k = max(1, min(fuse_last_k, layer + 1))
    fused = 0
    for h in hs[layer - k + 1: layer + 1]:
        fused = fused + h
    fused = fused / k  # (1, seq, hidden)
    mat = fused.squeeze(0).detach().cpu().numpy()  # (seq, hidden)
    return mat, (layer - k + 1), (layer + 1)


def svcca_between_prompts(prompt_a, prompt_b, model, tokenizer, layers, fuse_last_k=4, dim=None, energy=0.99):
    """
    Compute SVCCA for each layer in `layers` between two prompts' fused hidden states.
    Returns list of dicts with layer index and svcca score.
    """
    results = []
    for layer in layers:
        Xa, l_start, l_end = hidden_matrix_for_prompt(prompt_a, model, tokenizer, layer=layer, fuse_last_k=fuse_last_k)
        Xb, _, _ = hidden_matrix_for_prompt(prompt_b, model, tokenizer, layer=layer, fuse_last_k=fuse_last_k)
        n = min(Xa.shape[0], Xb.shape[0])
        # align by truncating to common sequence length
        score = svcca(Xa[:n], Xb[:n], dim=dim, energy=energy)
        results.append({
            "layer": layer,
            "layer_start": l_start,
            "layer_end": l_end,
            "seq_used": n,
            "hidden": Xa.shape[1],
            "svcca": score
        })
    return results

def svcca_holdout(
    X, Y,dim=None,energy=0.95,n_components=None,
    test_ratio=0.2,random_state=42,return_corrs=False
):
    """
    SVCCA با ارزیابی روی hold-out برای جلوگیری از بیش‌برازش CCA.
    مراحل:
      1) کاهش‌بُعد هر نما با SVD (مرکزبندی ستونی).
      2) تقسیم نمونه‌ها (توکن‌ها) به train/test.
      3) فیت CCA روی train و محاسبه‌ی همبستگی‌ها روی test.
    ورودی‌ها:
      X, Y: آرایه‌های (n_samples, d) — مثلاً (n_tokens, hidden_dim)
    خروجی:
      میانگین همبستگی‌های کانونیک روی test (float) یا لیست آن‌ها اگر return_corrs=True
    """
    # --- SVD reduction ---
    X = np.asarray(X); 
    Y = np.asarray(Y)
    n = min(len(X), len(Y))
    X = X[:n]; Y = Y[:n]

    # مرکزسازی ستونی
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    Ux, Sx, Vhx = svd(Xc, full_matrices=False)
    Uy, Sy, Vhy = svd(Yc, full_matrices=False)

    if dim is None:
        # تعیین رتبه بر اساس انرژی تجمیعی
        s2x = Sx**2; s2y = Sy**2
        rx = int(np.searchsorted(np.cumsum(s2x)/s2x.sum(), energy) + 1)
        ry = int(np.searchsorted(np.cumsum(s2y)/s2y.sum(), energy) + 1)
    else:
        rx = ry = int(dim)

    Xr = Ux[:, :rx] * Sx[:rx]   # (n_samples, rx)
    Yr = Uy[:, :ry] * Sy[:ry]   # (n_samples, ry)

    # --- split tokens into train/test ---
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    if len(train_idx) < 2:  
        k_fallback = max(1, min(rx, ry, n - 1))
        cca = CCA(n_components=k_fallback, max_iter=5000)
        Xc_fit, Yc_fit = cca.fit_transform(Xr, Yr)
        corrs = []
        for i in range(k_fallback):
            x, y = Xc_fit[:, i], Yc_fit[:, i]
            num = float((x * y).sum())
            den = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()) + 1e-12)
            val = num / den
            val = max(-1.0, min(1.0, val))   
            corrs.append(val)
        mean_corr = float(np.mean(corrs))
        mean_corr = max(-1.0, min(1.0, mean_corr))
        return corrs if return_corrs else mean_corr

    Xtr, Ytr = Xr[train_idx], Yr[train_idx]
    Xte, Yte = Xr[test_idx],  Yr[test_idx]

    # --- cap components safely (جلوگیری از overfit) ---
    k = min(Xtr.shape[1], Ytr.shape[1], len(train_idx) - 1)
    if n_components is None:
        n_components = k
    n_components = max(1, min(n_components, k))

    cca = CCA(n_components=n_components, max_iter=5000)
    cca.fit(Xtr, Ytr)                 # fit روی train
    Xte_c, Yte_c = cca.transform(Xte, Yte)  # ارزیابی روی test

    # --- correlations on test projections ---
    corrs = []
    for i in range(n_components):
        x = Xte_c[:, i]; y = Yte_c[:, i]
        num = float((x * y).sum())
        den = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()) + 1e-12)
        corrs.append(num / den)

    return corrs if return_corrs else float(np.mean(corrs))

