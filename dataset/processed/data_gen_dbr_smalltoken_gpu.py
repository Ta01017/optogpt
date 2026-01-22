p_train = idx[:split]
    idx_dev = idx[split:]

    train_struct = [struct_list[i] for i in idx_train]
    train_spec = [spec_list[i] for i in idx_train]
    train_meta = [meta_list[i] for i in idx_train]

    dev_struct = [struct_list[i] for i in idx_dev]
    dev_spec = [spec_list[i] for i in idx_dev]
    dev_meta = [meta_list[i] for i in idx_dev]

    with open(os.path.join(out_dir, "Structure_train.pkl"), "wb") as f:
        pkl.dump(train_struct, f)
    with open(os.path.join(out_dir, "Spectrum_train.pkl"), "wb") as f:
        pkl.dump(train_spec, f)
    with open(os.path.join(out_dir, "Structure_dev.pkl"), "wb") as f:
        pkl.dump(dev_struct, f)
    with open(os.path.join(out_dir, "Spectrum_dev.pkl"), "wb") as f:
        pkl.dump(dev_spec, f)

    with open(os.path.join(out_dir, "meta_train.pkl"), "wb") as f:
        pkl.dump(train_meta, f)
    with open(os.path.join(out_dir, "meta_dev.pkl"), "wb") as f:
        pkl.dump(dev_meta, f)

    print("\n== Saved ==")
    print("Train:", len(train_struct), "Dev:", len(dev_struct))
    print(" -", os.path.join(out_dir, "Structure_train.pkl"))
    print(" -", os.path.join(out_dir, "Spectrum_train.pkl"))
    print(" -", os.path.join(out_dir, "Structure_dev.pkl"))
    print(" -", os.path.join(out_dir, "Spectrum_dev.pkl"))
    print(" -", os.path.join(out_dir, "meta_train.pkl"))
    print(" -", os.path.join(out_dir, "meta_dev.pkl"))


# =========================
# 8) 验证：参考 tmm（逐波长） vs tmm_fast（向量化）
# =========================
def calc_RT_tmm_reference(materials, thicknesses_nm, nk_dict_np, wavelengths_um, pol="s", theta_deg=0.0):
    """
    参考实现：逐波长循环 + tmm.coh_tmm
    注意：
      - tmm 的 lam_vac 可用 nm 或任意一致单位，只要 n_list 是对应波长处折射率
      - 这里 thickness 用 nm（与你结构一致），lam_vac 用 nm
    """
    th0 = np.deg2rad(theta_deg)
    d_list = [np.inf] + list(map(float, thicknesses_nm)) + [np.inf]

    R = np.zeros_like(wavelengths_um, dtype=np.float64)
    T = np.zeros_like(wavelengths_um, dtype=np.float64)

    wl_nm_list = (wavelengths_um * 1000.0).astype(np.float64)
    for i, lam_nm in enumerate(wl_nm_list):
        n_list = [1.0] + [nk_dict_np[m][i] for m in materials] + [1.0]
        res = coh_tmm_ref(pol=pol, n_list=n_list, d_list=d_list, th_0=th0, lam_vac=lam_nm)
        R[i] = float(res["R"])
        T[i] = float(res["T"])
    return R.astype(np.float32), T.astype(np.float32)


def stopband_metrics(R, wavelengths_um, lambda0_um):
    w = wavelengths_um
    i0 = int(np.argmin(np.abs(w - lambda0_um)))

    thr = 0.90
    mask = (R >= thr)
    if mask.sum() == 0:
        thr = 0.80
        mask = (R >= thr)

    band_width = 0.0
    band_mean_R = float(np.mean(R[mask])) if mask.sum() > 0 else float(np.mean(R))

    if mask[i0]:
        l = i0
        while l - 1 >= 0 and mask[l - 1]:
            l -= 1
        r = i0
        while r + 1 < len(mask) and mask[r + 1]:
            r += 1
        band_width = float(w[r] - w[l])
        band_mean_R = float(np.mean(R[l:r + 1]))
    else:
        win = 5
        l = max(0, i0 - win)
        r = min(len(R) - 1, i0 + win)
        band_mean_R = float(np.mean(R[l:r + 1]))
        band_width = 0.0

    R0 = float(R[i0])
    return {
        "R_at_lambda0": R0,
        "stopband_width_um_est": band_width,
        "stopband_mean_R_est": band_mean_R,
        "threshold_used": thr,
    }


def validate_dbr(
    nk_dict_torch: Dict[str, torch.Tensor],
    wavelengths_um: np.ndarray,
    thk_table: Dict[str, Dict[int, int]],
    industry_pairs: List[Tuple[str, str]],
    num_checks: int = 10,
    seed: int = 123,
    plot: bool = False,
    eps_energy: float = 2e-4,
):
    """
    1) tmm_fast vs tmm（参考逐波长 coh_tmm）数值一致性
    2) DBR 行为 sanity：R(lambda0)、stopband 宽度估计
    3) 能量守恒 sanity：R+T <= 1 +eps
    """
    import matplotlib.pyplot as plt

    nk_dict_np = {k: v.detach().cpu().numpy() for k, v in nk_dict_torch.items()}

    wl_m = torch.tensor(wavelengths_um * 1e-6, dtype=REAL_DTYPE, device=DEVICE)
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)

    print("\n================ VALIDATION ================")
    print(f"num_checks={num_checks} | plot={plot} | energy_eps={eps_energy:g}")

    err_R = []
    err_T = []
    energy_viol = []
    R0s = []
    bws = []

    for ci in range(num_checks):
        mats, thks, meta = generate_dbr_discrete(industry_pairs, thk_table)
        lambda0_um = meta["lambda0_um"]

        # tmm_fast（batch=1）
        R_fast, T_fast = calc_RT_fast_batch(
            batch_mats=[mats],
            batch_thks_nm=[thks],
            nk_dict_torch=nk_dict_torch,
            wl_m=wl_m,
            theta_rad=theta_rad,
            pol="s",
        )
        R_fast = R_fast[0]
        T_fast = T_fast[0]

        # reference（逐点 tmm）
        R_ref, T_ref = calc_RT_tmm_reference(
            mats, thks, nk_dict_np, wavelengths_um, pol="s", theta_deg=0.0
        )

        dR = R_fast - R_ref
        dT = T_fast - T_ref
        max_abs_R = float(np.max(np.abs(dR)))
        rmse_R = float(np.sqrt(np.mean(dR ** 2)))
        max_abs_T = float(np.max(np.abs(dT)))
        rmse_T = float(np.sqrt(np.mean(dT ** 2)))

        err_R.append((max_abs_R, rmse_R))
        err_T.append((max_abs_T, rmse_T))

        # 能量守恒 sanity
        rt_sum = R_fast + T_fast
        viol = float(np.max(rt_sum - 1.0))
        energy_viol.append(viol)

        # DBR 指标
        m = stopband_metrics(R_fast, wavelengths_um, lambda0_um)
        R0s.append(m["R_at_lambda0"])
        bws.append(m["stopband_width_um_est"])

        print(
            f"[{ci:02d}] {meta['pair_name']} pairs={meta['pairs']} lambda0={lambda0_um:.3f}um | "
            f"max|dR|={max_abs_R:.3e} rmseR={rmse_R:.3e} | max|dT|={max_abs_T:.3e} rmseT={rmse_T:.3e} | "
            f"max(R+T-1)={viol:.3e} | R0={m['R_at_lambda0']:.3f} bandW~{m['stopband_width_um_est']:.3f}"
        )

        if plot:
            plt.figure()
            plt.plot(wavelengths_um, R_ref, label="R_ref(tmm)")
            plt.plot(wavelengths_um, R_fast, "--", label="R_fast(tmm_fast)")
            plt.axvline(lambda0_um, linestyle=":", label="lambda0")
            plt.ylim(-0.05, 1.05)
            plt.xlabel("Wavelength (um)")
            plt.ylabel("Reflectance R")
            plt.title(f"{meta['pair_name']} pairs={meta['pairs']} lambda0={lambda0_um:.3f}um")
            plt.legend()
            plt.show()

    max_abs_R_all = max(x[0] for x in err_R)
    rmse_R_mean = float(np.mean([x[1] for x in err_R]))
    max_abs_T_all = max(x[0] for x in err_T)
    rmse_T_mean = float(np.mean([x[1] for x in err_T]))

    worst_energy = float(np.max(energy_viol))

    print("\n---- Summary (tmm_fast vs tmm) ----")
    print(f"R: worst max|dR|={max_abs_R_all:.3e}, mean rmseR={rmse_R_mean:.3e}")
    print(f"T: worst max|dT|={max_abs_T_all:.3e}, mean rmseT={rmse_T_mean:.3e}")

    print("\n---- Energy sanity ----")
    print(f"worst max(R+T-1) = {worst_energy:.3e}")
    if worst_energy > eps_energy:
        print("WARNING: energy violation seems large. Check units/shapes/dtypes.")
    else:
        print("OK: energy violation within tolerance (absorption can make R+T<1, but should not exceed 1 much).")

    print("\n---- DBR sanity (using tmm_fast R) ----")
    print(f"R(lambda0): mean={float(np.mean(R0s)):.3f}, min={float(np.min(R0s)):.3f}, max={float(np.max(R0s)):.3f}")
    print(f"stopband_width_est(um): mean={float(np.mean(bws)):.3f}, min={float(np.min(bws)):.3f}, max={float(np.max(bws)):.3f}")
    print("==========================================\n")


# =========================
# 【新增】X) 绘图：单曲线 + 覆盖曲线
# =========================
def _ensure_plot_dirs(out_dir: str):
    os.makedirs(os.path.join(out_dir, "plots_single"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots_overlay"), exist_ok=True)

def plot_single_structure(wavelengths_um, R, T, meta, out_dir, idx):
    """
    单条曲线：R/T + lambda0
    """
    import matplotlib.pyplot as plt

    _ensure_plot_dirs(out_dir)

    plt.figure(figsize=(7, 4.2))
    plt.plot(wavelengths_um, R, label="R")
    plt.plot(wavelengths_um, T, label="T")
    plt.axvline(meta["lambda0_um"], linestyle="--", color="gray", label="lambda0")

    plt.ylim([-0.05, 1.05])
    plt.xlabel("Wavelength (um)")
    plt.ylabel("R / T")
    plt.title(f"[{idx}] {meta['pair_name']} pairs={meta['pairs']}  λ0={meta['lambda0_um']:.3f}um")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, "plots_single", f"single_{idx:06d}.png")
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_overlay(wavelengths_um, R_list, meta_list, out_dir, batch_idx, K=10):
    """
    覆盖图：叠加多条 R(λ)，用于观察覆盖/多样性
    """
    import matplotlib.pyplot as plt

    _ensure_plot_dirs(out_dir)

    n = len(R_list)
    K = min(K, n)
    if K <= 0:
        return
    idxs = np.random.choice(n, K, replace=False)

    plt.figure(figsize=(7, 4.2))
    for i in idxs:
        plt.plot(wavelengths_um, R_list[i], alpha=0.28)

    plt.ylim([-0.05, 1.05])
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Reflectance R")
    plt.title(f"Overlay batch={batch_idx}  (K={K})")
    plt.tight_layout()

    save_path = os.path.join(out_dir, "plots_overlay", f"overlay_{batch_idx:06d}.png")
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_global_overlay_from_saved(out_dir: str, wavelengths_um: np.ndarray, spec_list: List[list], K: int = 80):
    """
    生成结束后，从所有 spec 中随机抽 K 条画一张全局覆盖图
    spec_list: 每条是 [R...,T...] 的 list
    """
    import matplotlib.pyplot as plt

    _ensure_plot_dirs(out_dir)

    n = len(spec_list)
    if n == 0:
        return
    K = min(K, n)
    idxs = np.random.choice(n, K, replace=False)

    num_wl = len(wavelengths_um)

    plt.figure(figsize=(7, 4.2))
    for i in idxs:
        spec = np.array(spec_list[i], dtype=np.float32)
        R = spec[:num_wl]
        plt.plot(wavelengths_um, R, alpha=0.18)

    plt.ylim([-0.05, 1.05])
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Reflectance R")
    plt.title(f"Global overlay (random {K} / {n})")
    plt.tight_layout()

    save_path = os.path.join(out_dir, "plots_overlay", f"overlay_global_{K}.png")
    plt.savefig(save_path, dpi=180)
    plt.close()


# =========================
# 9) 主流程
# =========================
def main():
    print("DEVICE =", DEVICE, "| BATCH_SIZE =", BATCH_SIZE)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)

    # 检查 nk 文件
    needed = set()
    for H, L in INDUSTRY_CORE:
        needed.add(H)
        needed.add(L)

    missing = [m for m in sorted(needed) if not os.path.exists(os.path.join(NK_DIR, f"{m}.csv"))]
    if missing:
        raise FileNotFoundError("Missing nk csv:\n  " + "\n  ".join(missing) + f"\nExpected under: {NK_DIR}")

    # wl/theta torch（SI）
    wl_m = torch.tensor(WAVELENGTHS_UM * 1e-6, dtype=REAL_DTYPE, device=DEVICE)  # [num_wl]
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)             # [1]

    # nk -> torch on DEVICE
    nk_dict_torch = load_nk_torch(sorted(list(needed)), WAVELENGTHS_UM)

    # 厚度表
    thk_table = precompute_qw_thickness_table(nk_dict_torch, WAVELENGTHS_UM, LAMBDA0_SET_UM)

    print("\n== Discrete thickness table (nm) ==")
    for m in sorted(thk_table.keys()):
        vals = sorted(set(thk_table[m].values()))
        print(f"{m:8s} unique_thk={len(vals):2d}  values={vals}")

    # ==========（可选但强烈建议）先做验证 ==========
    validate_dbr(
        nk_dict_torch=nk_dict_torch,
        wavelengths_um=WAVELENGTHS_UM,
        thk_table=thk_table,
        industry_pairs=INDUSTRY_CORE,
        num_checks=10,
        seed=123,
        plot=False,
        eps_energy=2e-4,
    )

    # ========== 生成数据 ==========
    struct_list, spec_list, meta_list = [], [], []

    batch_counter = 0
    pbar = tqdm(total=NUM_SAMPLES, desc="Generating DBR small-token 30k (tmm_fast coh_tmm batch)")

    while len(struct_list) < NUM_SAMPLES:
        cur = min(BATCH_SIZE, NUM_SAMPLES - len(struct_list))
        batch_mats, batch_thks, batch_meta = [], [], []

        for _ in range(cur):
            mats, thks, meta = generate_dbr_discrete(INDUSTRY_CORE, thk_table)
            batch_mats.append(mats)
            batch_thks.append(thks)
            batch_meta.append(meta)

        R_batch, T_batch = calc_RT_fast_batch(
            batch_mats=batch_mats,
            batch_thks_nm=batch_thks,
            nk_dict_torch=nk_dict_torch,
            wl_m=wl_m,
            theta_rad=theta_rad,
            pol="s",
        )  # [cur, num_wl]

        # ===== 【新增】绘图：单张 + 覆盖 =====
        # 单张：概率触发（别太频繁，不然 IO 很慢）
        if random.random() < PLOT_SINGLE_PROB:
            plot_single_structure(
                wavelengths_um=WAVELENGTHS_UM,
                R=R_batch[0],
                T=T_batch[0],
                meta=batch_meta[0],
                out_dir=OUT_DIR,
                idx=len(struct_list),
            )

        # 覆盖图：每隔若干 batch 保存一次（建议 10/20）
        if (batch_counter % PLOT_OVERLAY_EVERY_BATCH) == 0:
            plot_overlay(
                wavelengths_um=WAVELENGTHS_UM,
                R_list=R_batch,
                meta_list=batch_meta,
                out_dir=OUT_DIR,
                batch_idx=len(struct_list),
                K=PLOT_OVERLAY_K,
            )

        batch_counter += 1
        # ===== 结束绘图 =====

        for bi in range(cur):
            mats = batch_mats[bi]
            thks = batch_thks[bi]
            meta = batch_meta[bi]

            struct_tokens = [f"{m}_{int(t)}" for m, t in zip(mats, thks)]
            spec_vec = np.concatenate([R_batch[bi], T_batch[bi]], axis=0).astype(np.float32).tolist()

            struct_list.append(struct_tokens)
            spec_list.append(spec_vec)
            meta_list.append(meta)

        pbar.update(cur)

    pbar.close()

    # 保存 train/dev
    save_split(struct_list, spec_list, meta_list, OUT_DIR)

    # ===== 【新增】生成结束后：全局覆盖图 =====
    try:
        plot_global_overlay_from_saved(
            out_dir=OUT_DIR,
            wavelengths_um=WAVELENGTHS_UM,
            spec_list=spec_list,
            K=PLOT_GLOBAL_OVERLAY_K,
        )
        print(f"[Plot] Saved global overlay: plots_overlay/overlay_global_{min(PLOT_GLOBAL_OVERLAY_K, len(spec_list))}.png")
    except Exception as e:
        print("[Plot] Global overlay failed:", repr(e))

    # ========== 统计 ==========
    pair_cnt = Counter()
    cover_cnt = Counter()
    tok_cnt = Counter()

    for seq, meta in zip(struct_list, meta_list):
        pair_cnt[meta["pair_name"]] += 1
        cover_cnt[(meta["pair_name"], meta["lambda0_nm"], meta["pairs"])] += 1
        for tok in seq:
            tok_cnt[tok] += 1

    print("\n== Pair distribution ==")
    total = sum(pair_cnt.values())
    for k, v in pair_cnt.most_common():
        print(f"  {k:15s} {v:6d} ({v/total:.3%})")

    print("\n== Coverage (pair_name, lambda0_nm, pairs) top 30 ==")
    for k, v in cover_cnt.most_common(30):
        print(" ", k, v)
    print("unique (pair,lambda0,pairs) combos:", len(cover_cnt))

    print("\n== Top tokens (raw) ==")
    tot_tok = sum(tok_cnt.values())
    for t, c in tok_cnt.most_common(20):
        print(f"  {t:15s} {c:6d} ({c/tot_tok:.3%})")

    mats = sorted(list(needed))
    approx_unique_thk = {m: len(set(thk_table[m].values())) for m in mats}
    approx_vocab = sum(approx_unique_thk.values())
    print("\n== Approx token space ==")
    print("materials:", mats)
    print("unique_thk per material:", approx_unique_thk)
    print("approx total structure tokens (material_thk):", approx_vocab)

    print("\nDone. OUT_DIR =", OUT_DIR)
    print(f"[Plot] single curves -> {os.path.join(OUT_DIR, 'plots_single')}")
    print(f"[Plot] overlay curves -> {os.path.join(OUT_DIR, 'plots_overlay')}")


if __name__ == "__main__":
    main()
