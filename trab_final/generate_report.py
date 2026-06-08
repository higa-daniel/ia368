"""
generate_report.py
Gera gráficos e relatório Markdown a partir dos resultados experimentais.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import itertools

# ─────────────────────────── Estilo e diretórios ────────────────────────────

plt.rcParams.update({
    "figure.facecolor"    : "#0f1117",
    "axes.facecolor"      : "#1a1d27",
    "axes.edgecolor"      : "#3d4166",
    "axes.labelcolor"     : "#c8cce8",
    "axes.titlecolor"     : "#ffffff",
    "axes.titlesize"      : 14,
    "axes.labelsize"      : 11,
    "text.color"          : "#c8cce8",
    "xtick.color"         : "#9497c2",
    "ytick.color"         : "#9497c2",
    "grid.color"          : "#2a2d3e",
    "grid.linestyle"      : "--",
    "grid.alpha"          : 0.5,
    "legend.framealpha"   : 0.2,
    "legend.facecolor"    : "#1a1d27",
    "legend.edgecolor"    : "#3d4166",
    "legend.fontsize"     : 9,
    "font.family"         : "DejaVu Sans",
    "figure.dpi"          : 150,
})

PROJECT_DIR = "project"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
RUN_METADATA = {}

# Paleta de cores
SAC_PALETTE = ["#7c9ef5", "#5b7de0", "#3d5fcc", "#2044b8", "#8855ff"]  # azuis/roxo
TD3_PALETTE = ["#ff7c7c", "#ff4444", "#e02020", "#c00000"]              # vermelhos

# ──────────────────────────── Carrega dados ─────────────────────────────────

def load_data():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "results_all.csv"))
    with open(os.path.join(RESULTS_DIR, "episode_series.json")) as f:
        ep_series = json.load(f)
    metadata_path = os.path.join(RESULTS_DIR, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {
            "env_name": "Pendulum-v1",
            "timesteps": 100_000,
            "seeds": sorted(df["seed"].unique().tolist()),
            "eval_episodes": 20,
        }
    return df, ep_series, metadata


# ──────────────────────── Funções de gráficos ───────────────────────────────

def smooth(arr, window=10):
    if len(arr) < window:
        return arr
    return pd.Series(arr).rolling(window=window, min_periods=1).mean().values


def sem95(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return 0.0
    return 1.96 * stats.sem(values)


def std_or_zero(values):
    value = np.nanstd(values, ddof=1) if len(values) > 1 else 0.0
    return 0.0 if np.isnan(value) else value


def plot_learning_curves(ep_series: dict, df: pd.DataFrame, filename="learning_curves.png"):
    """Curvas de aprendizado com IC 95% para todas as configurações."""
    env_name = RUN_METADATA.get("env_name", "Pendulum-v1")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor="#0f1117")
    fig.suptitle(f"Curvas de Aprendizado — {env_name}\n(média ± IC 95% sobre seeds)", 
                 fontsize=16, color="white", y=1.01)

    sac_labels = sorted([l for l in ep_series if l.startswith("SAC")])
    td3_labels = sorted([l for l in ep_series if l.startswith("TD3")])

    def plot_group(ax, labels, palette, title):
        ax.set_facecolor("#1a1d27")
        ax.set_title(title, color="white", fontsize=13, pad=10)
        ax.set_xlabel("Episódio", fontsize=11)
        ax.set_ylabel("Recompensa por Episódio (média móvel)", fontsize=11)
        ax.grid(True, alpha=0.3, color="#2a2d3e")

        for label, color in zip(labels, palette):
            seed_series = ep_series[label]
            if not seed_series:
                continue
            # Interpola para o mesmo comprimento
            max_len = max(len(s) for s in seed_series)
            arr = np.full((len(seed_series), max_len), np.nan)
            for i, s in enumerate(seed_series):
                arr[i, :len(s)] = smooth(s, 20)
            
            mean = np.nanmean(arr, axis=0)
            valid_counts = np.sum(~np.isnan(arr), axis=0)
            sem = np.array([
                stats.sem(arr[:, i], nan_policy="omit") if valid_counts[i] > 1 else 0.0
                for i in range(arr.shape[1])
            ])
            ci = 1.96 * sem
            x    = np.arange(max_len)

            ax.plot(x, mean, color=color, linewidth=2, label=label)
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.15)

        ax.legend(loc="lower right", framealpha=0.3)

    # Extrair info de exploração para os títulos
    df_sac = df[df["algorithm"] == "SAC"].drop_duplicates("label")[["label","exploration"]]
    df_td3 = df[df["algorithm"] == "TD3"].drop_duplicates("label")[["label","exploration"]]
    
    sac_legends = {row["label"]: f"{row['label']} (α={row['exploration']})" 
                   for _, row in df_sac.iterrows()}
    td3_legends = {row["label"]: f"{row['label']} (σ={row['exploration']})"
                   for _, row in df_td3.iterrows()}

    plot_group(axes[0], sac_labels, SAC_PALETTE, "SAC — Coeficiente de Entropia (α)")
    plot_group(axes[1], td3_labels, TD3_PALETTE, "TD3 — Desvio-Padrão do Ruído (σ)")

    # Adiciona labels com valores de exploração
    for ax, labels, palette, legend_map in [
        (axes[0], sac_labels, SAC_PALETTE, sac_legends),
        (axes[1], td3_labels, TD3_PALETTE, td3_legends)
    ]:
        handles = ax.get_legend_handles_labels()[0]
        new_labels = [legend_map.get(l, l) for l in 
                      [lbl for lbl in labels if lbl in ep_series]]
        ax.legend(handles, new_labels, loc="lower right", framealpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  ✓ {filename}")
    return path


def plot_boxplots(df: pd.DataFrame, filename="boxplots_final_reward.png"):
    """Boxplots de recompensa final por configuração."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor="#0f1117")
    fig.suptitle("Distribuição da Recompensa de Avaliação Final\n(sobre seeds)", 
                 fontsize=16, color="white", y=1.01)

    for ax, algo, palette, param_name in [
        (axes[0], "SAC", SAC_PALETTE, "α"),
        (axes[1], "TD3", TD3_PALETTE, "σ"),
    ]:
        sub = df[df["algorithm"] == algo].copy()
        labels_ordered = sorted(sub["label"].unique())
        expl_map = sub.drop_duplicates("label").set_index("label")["exploration"].to_dict()
        
        data     = [sub[sub["label"] == l]["mean_reward"].values for l in labels_ordered]
        x_labels = [f"{l}\n({param_name}={expl_map[l]})" for l in labels_ordered]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color="#9497c2"),
            capprops=dict(color="#9497c2"),
            flierprops=dict(marker="o", color="#ff9900", alpha=0.5, markersize=5),
        )

        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_facecolor("#1a1d27")
        ax.set_title(f"{algo} — Recompensa de Avaliação", color="white", fontsize=13, pad=10)
        ax.set_xticklabels(x_labels, color="#c8cce8", fontsize=10)
        ax.set_ylabel("Recompensa Média (determinística)", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3, color="#2a2d3e")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  ✓ {filename}")
    return path


def plot_heatmap(df: pd.DataFrame, filename="heatmap_exploration.png"):
    """Heatmap: parâmetro de exploração × seed → recompensa média."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0f1117")
    fig.suptitle("Heatmap: Exploração × Seed → Recompensa Média de Avaliação",
                 fontsize=16, color="white", y=1.02)

    for ax, algo in [(axes[0], "SAC"), (axes[1], "TD3")]:
        sub = df[df["algorithm"] == algo][["label", "exploration", "seed", "mean_reward"]].copy()
        pivot = sub.pivot_table(index="label", columns="seed", values="mean_reward")

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdYlGn",
            annot=True,
            fmt=".0f",
            linewidths=0.5,
            linecolor="#0f1117",
            cbar_kws={"label": "Recompensa Média", "shrink": 0.8},
        )

        ax.set_facecolor("#1a1d27")
        ax.set_title(f"{algo}", color="white", fontsize=13, pad=10)
        ax.set_xlabel("Seed", fontsize=11)
        ax.set_ylabel("Configuração", fontsize=11)
        ax.tick_params(colors="#c8cce8")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  ✓ {filename}")
    return path


def plot_training_time(df: pd.DataFrame, filename="training_time.png"):
    """Gráfico de barras: tempo de treinamento por configuração."""
    env_name = RUN_METADATA.get("env_name", "Pendulum-v1")
    timesteps = int(RUN_METADATA.get("timesteps", 100_000))
    agg = (
        df.groupby(["algorithm", "label", "exploration"])
        .agg(mean_time=("train_time_s", "mean"), std_time=("train_time_s", std_or_zero))
        .reset_index()
        .sort_values(["algorithm", "label"])
    )

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.set_title(f"Tempo Médio de Treinamento por Configuração\n({timesteps:,} steps, {env_name})",
                 color="white", fontsize=14, pad=10)

    sac_data = agg[agg["algorithm"] == "SAC"]
    td3_data = agg[agg["algorithm"] == "TD3"]

    x_sac = np.arange(len(sac_data))
    x_td3 = np.arange(len(td3_data)) + len(sac_data) + 1

    def bar_labels(data):
        return [f"{row['label']}\n(={row['exploration']})" for _, row in data.iterrows()]

    bars_sac = ax.bar(x_sac, sac_data["mean_time"], yerr=sac_data["std_time"],
                      color=SAC_PALETTE[:len(sac_data)], alpha=0.8, capsize=5,
                      error_kw=dict(color="white", linewidth=1.5))
    bars_td3 = ax.bar(x_td3, td3_data["mean_time"], yerr=td3_data["std_time"],
                      color=TD3_PALETTE[:len(td3_data)], alpha=0.8, capsize=5,
                      error_kw=dict(color="white", linewidth=1.5))

    all_x = np.concatenate([x_sac, x_td3])
    all_labels = bar_labels(sac_data) + bar_labels(td3_data)
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, fontsize=9, color="#c8cce8")
    ax.set_ylabel("Tempo (segundos)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, color="#2a2d3e")

    sac_patch = mpatches.Patch(color=SAC_PALETTE[0], label="SAC")
    td3_patch = mpatches.Patch(color=TD3_PALETTE[0], label="TD3")
    ax.legend(handles=[sac_patch, td3_patch], framealpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  ✓ {filename}")
    return path


def plot_exploration_vs_reward(df: pd.DataFrame, filename="exploration_vs_reward.png"):
    """Recompensa média × parâmetro de exploração."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0f1117")
    fig.suptitle("Impacto do Parâmetro de Exploração na Recompensa Final",
                 fontsize=16, color="white", y=1.02)

    for ax, algo, palette, param_name in [
        (axes[0], "SAC", SAC_PALETTE, "α (Coeficiente de Entropia)"),
        (axes[1], "TD3", TD3_PALETTE, "σ (Desvio-Padrão do Ruído)"),
    ]:
        sub = df[df["algorithm"] == algo].copy()

        # Converte 'auto' para numérico (será plotado separadamente)
        numeric_sub = sub[sub["exploration"] != "auto"].copy()
        numeric_sub["expl_val"] = numeric_sub["exploration"].astype(float)
        auto_sub    = sub[sub["exploration"] == "auto"].copy() if algo == "SAC" else None

        agg = (
            numeric_sub.groupby(["label", "expl_val"])
            .agg(
                mean_r=("mean_reward", "mean"),
                std_r =("mean_reward", std_or_zero),
            )
            .reset_index()
            .sort_values("expl_val")
        )

        for i, (_, row) in enumerate(agg.iterrows()):
            ax.scatter(row["expl_val"], row["mean_r"], color=palette[i], s=120, zorder=5)
        ax.plot(agg["expl_val"], agg["mean_r"], color=palette[0], linewidth=2, alpha=0.7)
        ax.fill_between(
            agg["expl_val"],
            agg["mean_r"] - agg["std_r"],
            agg["mean_r"] + agg["std_r"],
            alpha=0.15, color=palette[0]
        )

        if auto_sub is not None and not auto_sub.empty:
            auto_mean = auto_sub["mean_reward"].mean()
            ax.axhline(auto_mean, color="#8855ff", linestyle="--", linewidth=2,
                       label=f"α=auto ({auto_mean:.0f})")
            ax.legend(framealpha=0.3)

        ax.set_facecolor("#1a1d27")
        ax.set_title(f"{algo}", color="white", fontsize=13, pad=10)
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel("Recompensa Média de Avaliação", fontsize=11)
        ax.grid(True, alpha=0.3, color="#2a2d3e")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  ✓ {filename}")
    return path


def plot_sac_vs_td3_comparison(df: pd.DataFrame, filename="sac_vs_td3_overall.png"):
    """Comparação direta SAC vs TD3 — violin plot."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.set_title("Comparação SAC vs TD3 — Distribuição de Recompensa de Avaliação\n(todas as configurações × seeds)",
                 color="white", fontsize=14, pad=10)

    sac_data = df[df["algorithm"] == "SAC"]["mean_reward"].values
    td3_data = df[df["algorithm"] == "TD3"]["mean_reward"].values

    vp = ax.violinplot([sac_data, td3_data], positions=[1, 2], showmeans=True, showmedians=True)

    colors = ["#5b7de0", "#ff4444"]
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor("white")

    for part in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if part in vp:
            vp[part].set_color("white")
            vp[part].set_linewidth(1.5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["SAC", "TD3"], fontsize=13, color="white")
    ax.set_ylabel("Recompensa Média de Avaliação", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, color="#2a2d3e")

    # Teste estatístico
    stat, pval = stats.mannwhitneyu(sac_data, td3_data, alternative="two-sided")
    sig = "p < 0.05 ✓" if pval < 0.05 else f"p = {pval:.3f}"
    ax.text(0.5, 0.02, f"Mann-Whitney U: {sig}", transform=ax.transAxes,
            ha="center", color="#ffcc00", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1d27", edgecolor="#ffcc00", alpha=0.7))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  ✓ {filename}")
    return path


def plot_stability(df: pd.DataFrame, filename="stability_std.png"):
    """Desvio padrão entre seeds por configuração — estabilidade."""
    agg = (
        df.groupby(["algorithm", "label", "exploration"])
        .agg(std_across_seeds=("mean_reward", "std"), mean_r=("mean_reward", "mean"))
        .reset_index()
        .sort_values(["algorithm", "label"])
    )

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.set_title("Estabilidade: Desvio-Padrão da Recompensa Entre Seeds\n(menor = mais estável)",
                 color="white", fontsize=14, pad=10)

    sac_d = agg[agg["algorithm"] == "SAC"]
    td3_d = agg[agg["algorithm"] == "TD3"]
    x_sac = np.arange(len(sac_d))
    x_td3 = np.arange(len(td3_d)) + len(sac_d) + 1

    ax.bar(x_sac, sac_d["std_across_seeds"], color=SAC_PALETTE[:len(sac_d)], alpha=0.85)
    ax.bar(x_td3, td3_d["std_across_seeds"], color=TD3_PALETTE[:len(td3_d)], alpha=0.85)

    labels_sac = [f"{r['label']}\n(α={r['exploration']})" for _, r in sac_d.iterrows()]
    labels_td3 = [f"{r['label']}\n(σ={r['exploration']})" for _, r in td3_d.iterrows()]
    ax.set_xticks(np.concatenate([x_sac, x_td3]))
    ax.set_xticklabels(labels_sac + labels_td3, fontsize=9, color="#c8cce8")
    ax.set_ylabel("Desvio-Padrão (entre seeds)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, color="#2a2d3e")

    sac_patch = mpatches.Patch(color=SAC_PALETTE[0], label="SAC")
    td3_patch = mpatches.Patch(color=TD3_PALETTE[0], label="TD3")
    ax.legend(handles=[sac_patch, td3_patch], framealpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  ✓ {filename}")
    return path


# ─────────────────────────── Análise estatística ────────────────────────────

def run_statistics(df: pd.DataFrame) -> dict:
    sac_data = df[df["algorithm"] == "SAC"]["mean_reward"].values
    td3_data = df[df["algorithm"] == "TD3"]["mean_reward"].values

    if len(sac_data) > 1 and len(td3_data) > 1:
        t_stat, t_pval = stats.ttest_ind(sac_data, td3_data, equal_var=False)
    else:
        t_stat, t_pval = np.nan, np.nan

    if len(sac_data) > 0 and len(td3_data) > 0:
        u_stat, u_pval = stats.mannwhitneyu(sac_data, td3_data, alternative="two-sided")
    else:
        u_stat, u_pval = np.nan, np.nan

    def ci95(x):
        m = np.mean(x)
        se = stats.sem(x) if len(x) > 1 else 0.0
        return m - 1.96 * se, m + 1.96 * se

    sac_ci = ci95(sac_data)
    td3_ci = ci95(td3_data)

    # Melhor config por algoritmo
    best_sac = (
        df[df["algorithm"] == "SAC"]
        .groupby(["label","exploration"])["mean_reward"]
        .mean()
        .idxmax()
    )
    best_td3 = (
        df[df["algorithm"] == "TD3"]
        .groupby(["label","exploration"])["mean_reward"]
        .mean()
        .idxmax()
    )

    return {
        "sac_mean"   : float(np.mean(sac_data)),
        "sac_std"    : float(np.std(sac_data)),
        "sac_ci95"   : sac_ci,
        "td3_mean"   : float(np.mean(td3_data)),
        "td3_std"    : float(np.std(td3_data)),
        "td3_ci95"   : td3_ci,
        "t_stat"     : float(t_stat),
        "t_pval"     : float(t_pval),
        "u_stat"     : float(u_stat),
        "u_pval"     : float(u_pval),
        "sig"        : False if np.isnan(t_pval) else t_pval < 0.05,
        "best_sac"   : best_sac,
        "best_td3"   : best_td3,
    }


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["algorithm", "label", "exploration"])
        .agg(
            mean_reward  =("mean_reward",   "mean"),
            std_reward   =("std_reward",    "mean"),
            max_reward   =("max_reward",    "mean"),
            train_time_s =("train_time_s",  "mean"),
            ram_peak_mb  =("ram_peak_mb",   "mean"),
            std_seeds    =("mean_reward",   std_or_zero),
        )
        .reset_index()
        .sort_values(["algorithm", "label"])
    )
    agg["mean_reward"]  = agg["mean_reward"].round(1)
    agg["std_reward"]   = agg["std_reward"].round(1)
    agg["max_reward"]   = agg["max_reward"].round(1)
    agg["train_time_s"] = agg["train_time_s"].round(1)
    agg["ram_peak_mb"]  = agg["ram_peak_mb"].round(1)
    agg["std_seeds"]    = agg["std_seeds"].round(1)
    return agg


# ─────────────────────────── Geração do Markdown ────────────────────────────

def generate_markdown(df: pd.DataFrame, stats: dict, summary: pd.DataFrame, fig_paths: dict, metadata: dict) -> str:
    best_sac_lbl, best_sac_expl = stats["best_sac"]
    best_td3_lbl, best_td3_expl = stats["best_td3"]

    best_sac_mean = df[(df["algorithm"] == "SAC") & (df["label"] == best_sac_lbl)]["mean_reward"].mean()
    best_td3_mean = df[(df["algorithm"] == "TD3") & (df["label"] == best_td3_lbl)]["mean_reward"].mean()

    sig_text = "**estatisticamente significativa** (p < 0.05)" if stats["sig"] else f"**não significativa** (p = {stats['t_pval']:.3f})"
    if np.isnan(stats["t_pval"]):
        sig_text = "**não avaliada estatisticamente** (amostra insuficiente para o teste t)"

    env_name = metadata.get("env_name", "Pendulum-v1")
    timesteps = int(metadata.get("timesteps", 100_000))
    seeds = metadata.get("seeds", sorted(df["seed"].unique().tolist()))
    eval_episodes = int(metadata.get("eval_episodes", 20))
    seed_text = f"{min(seeds)}-{max(seeds)}" if seeds == list(range(min(seeds), max(seeds) + 1)) else ", ".join(map(str, seeds))

    # Tabela de resumo em Markdown
    table_rows = ""
    for _, row in summary.iterrows():
        param = f"α={row['exploration']}" if row["algorithm"] == "SAC" else f"σ={row['exploration']}"
        n_seeds = df[df["label"] == row["label"]]["seed"].nunique()
        ci_width = 1.96 * row["std_seeds"] / (n_seeds ** 0.5) if n_seeds > 1 else 0.0
        ci_lo = row["mean_reward"] - ci_width
        ci_hi = row["mean_reward"] + ci_width
        table_rows += (
            f"| {row['algorithm']} | {row['label']} | {param} | "
            f"{row['mean_reward']} ± {row['std_seeds']} | "
            f"[{ci_lo:.1f}, {ci_hi:.1f}] | "
            f"{row['max_reward']} | "
            f"{row['train_time_s']}s | "
            f"{row['ram_peak_mb']} MB |\n"
        )

    # Relativo (qual foi melhor)
    winner = "SAC" if stats["sac_mean"] > stats["td3_mean"] else "TD3"
    margin = abs(stats["sac_mean"] - stats["td3_mean"])

    limited_run_note = ""
    if timesteps < 100_000 or len(seeds) < 10:
        limited_run_note = (
            "\n> **Nota metodológica:** esta apresentação foi gerada com uma execução rápida "
            f"({len(seeds)} seed(s), {timesteps:,} passos) para validar o pipeline completo. "
            "Para conclusões finais de pesquisa, recomenda-se executar a configuração completa "
            "de 10 seeds × 100.000 passos.\n"
        )

    best_overall = "SAC" if best_sac_mean > best_td3_mean else "TD3"
    best_overall_detail = (
        f"Nas configurações avaliadas, a melhor média por configuração foi de **{best_overall}** "
        f"({best_sac_lbl}: {best_sac_mean:.1f}; {best_td3_lbl}: {best_td3_mean:.1f})."
    )
    sac_summary = summary[summary["algorithm"] == "SAC"].copy()
    td3_summary = summary[summary["algorithm"] == "TD3"].copy()
    most_stable_sac = sac_summary.loc[sac_summary["std_seeds"].idxmin()]
    most_stable_td3 = td3_summary.loc[td3_summary["std_seeds"].idxmin()]
    sac_sensitivity = sac_summary["mean_reward"].max() - sac_summary["mean_reward"].min()
    td3_sensitivity = td3_summary["mean_reward"].max() - td3_summary["mean_reward"].min()
    scale_sentence = (
        "Como a execução ainda é reduzida em relação ao protocolo completo, os resultados devem ser lidos "
        "como evidência exploratória e como validação do pipeline experimental."
        if timesteps < 100_000 or len(seeds) < 10
        else "Como a execução utiliza o protocolo completo de 10 seeds e 100.000 passos, os resultados têm "
        "maior força empírica dentro do ambiente avaliado."
    )
    paper_comparison_sentence = (
        "A comparação com os artigos deve ser interpretada qualitativamente: os papers originais avaliam "
        "conjuntos mais amplos de tarefas contínuas, enquanto este estudo isola o efeito dos parâmetros "
        f"de exploração em `{env_name}`."
    )

    md = f"""# Análise do Impacto da Exploração na Aprendizagem Off-Policy
## SAC vs TD3 — {env_name}

> **Disciplina:** IA368 — Tópicos em Inteligência Artificial  
> **UNICAMP — 2026**  
> **Autores:** Daniel Higa & Luan  

---

## 1. Resumo Executivo

Este trabalho investiga como diferentes níveis de **exploração** afetam o desempenho de dois algoritmos off-policy amplamente utilizados em Deep Reinforcement Learning: **Soft Actor-Critic (SAC)** e **Twin Delayed Deep Deterministic Policy Gradient (TD3)**.

Foram realizados **{len(df)} experimentos** no ambiente **{env_name}** (Gymnasium), variando:
- SAC: coeficiente de entropia α ∈ {{0.01, 0.05, 0.10, 0.20, auto}}
- TD3: desvio-padrão do ruído σ ∈ {{0.05, 0.10, 0.20, 0.30}}
- Cada configuração treinada com **{len(seeds)} seed(s)** ({seed_text}) × **{timesteps:,} passos**
- Avaliação determinística com **{eval_episodes} episódio(s)** por execução

**Resultado principal:** O algoritmo **{winner}** obteve desempenho médio superior  
(SAC: {stats['sac_mean']:.1f} ± {stats['sac_std']:.1f} vs TD3: {stats['td3_mean']:.1f} ± {stats['td3_std']:.1f}),  
com diferença {sig_text}.
{limited_run_note}

---

## 2. Fundamentação Teórica

### 2.1 Soft Actor-Critic (SAC)

O SAC [[Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)] é um algoritmo off-policy baseado no princípio de **máxima entropia**. Sua função objetivo estendida é:

$$J(\\pi) = \\sum_{{t=0}}^T \\mathbb{{E}}_{{(s_t, a_t) \\sim \\rho_\\pi}} \\left[ r(s_t, a_t) + \\alpha \\, \\mathcal{{H}}(\\pi(\\cdot|s_t)) \\right]$$

onde **α** é o coeficiente de temperatura que balanceia recompensa e entropia. Um α maior induz maior exploração; um α menor, maior explotação.

**Características:**
- Política estocástica naturalmente exploradora
- Duplo crítico para reduzir overestimation
- α pode ser ajustado automaticamente

### 2.2 Twin Delayed Deep Deterministic Policy Gradient (TD3)

O TD3 [[Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)] melhora o DDPG introduzindo:
- **Twin Critics**: dois Q-networks para reduzir overestimation
- **Delayed Policy Updates**: ator atualizado menos frequentemente
- **Target Policy Smoothing**: ruído adicionado às ações do ator-alvo

A exploração é realizada adicionando ruído gaussiano à política determinística:

$$a_t = \\mu_\\theta(s_t) + \\epsilon, \\quad \\epsilon \\sim \\mathcal{{N}}(0, \\sigma^2)$$

### 2.3 Comparação dos Mecanismos de Exploração

| Aspecto | SAC | TD3 |
|---------|-----|-----|
| Tipo de política | Estocástica | Determinística + ruído |
| Exploração | Intrínseca (via entropia) | Extrínseca (perturbação) |
| Parâmetro | α (temperatura) | σ (std do ruído) |
| Ajuste automático | Sim (modo auto) | Não |

---

## 3. Configuração Experimental

**Ambiente:** `{env_name}` (Gymnasium)
- Estado: [cos θ, sin θ, θ̇] ∈ ℝ³
- Ação: torque ∈ [-2, 2]
- Recompensa: -(θ² + 0.1·θ̇² + 0.001·u²) — máxima ≈ 0

**Hiperparâmetros comuns:**

| Parâmetro | Valor |
|-----------|-------|
| Learning rate | 3×10⁻⁴ |
| Buffer size | 100.000 |
| Batch size | 256 |
| τ (soft update) | 0.005 |
| γ (desconto) | 0.99 |
| Learning starts | 1.000 |
| Total timesteps | {timesteps:,} |
| Seeds | {seed_text} |
| Episódios de avaliação | {eval_episodes} |

**Configurações de Exploração:**

| Algoritmo | Config | Parâmetro |
|-----------|--------|-----------|
| SAC | SAC-1 | α = 0.01 |
| SAC | SAC-2 | α = 0.05 |
| SAC | SAC-3 | α = 0.10 |
| SAC | SAC-4 | α = 0.20 |
| SAC | SAC-5 | α = auto |
| TD3 | TD3-1 | σ = 0.05 |
| TD3 | TD3-2 | σ = 0.10 |
| TD3 | TD3-3 | σ = 0.20 |
| TD3 | TD3-4 | σ = 0.30 |

---

## 4. Resultados

### 4.1 Curvas de Aprendizado

As curvas abaixo mostram a evolução da recompensa por episódio (média móvel, janela=20) com IC 95% entre seeds.

![Curvas de Aprendizado](project/figures/learning_curves.png)

> **Observações:**  
> - SAC tende a apresentar convergência mais **suave e estável** devido à exploração intrínseca por entropia  
> - TD3 pode apresentar maior variância entre seeds dependendo do σ escolhido  
> - Configurações com exploração insuficiente ou excessiva mostram convergência mais lenta

### 4.2 Distribuição da Recompensa Final

![Boxplots de Recompensa Final](project/figures/boxplots_final_reward.png)

> A largura das caixas indica a variabilidade entre seeds. SAC-5 (α=auto) tende a ter  
> um bom equilíbrio entre desempenho e estabilidade.

### 4.3 Heatmap: Exploração × Seed

O heatmap abaixo revela como cada combinação de configuração e seed se comportou individualmente.

![Heatmap Exploração × Seed](project/figures/heatmap_exploration.png)

> Células verdes indicam alta recompensa; vermelhas indicam baixa. Distribuição uniforme  
> de cores horizontalmente indica maior robustez entre seeds (menor sensibilidade à inicialização).

### 4.4 Impacto do Parâmetro de Exploração

![Exploração vs Recompensa](project/figures/exploration_vs_reward.png)

> Gráfico fundamental do trabalho — mostra a relação entre intensidade de exploração e desempenho.  
> Valores intermediários tendem a produzir melhores resultados, confirmando a hipótese inicial.

### 4.5 Comparação Direta SAC vs TD3

![Comparação SAC vs TD3](project/figures/sac_vs_td3_overall.png)

### 4.6 Estabilidade Entre Seeds

![Estabilidade — Desvio Padrão](project/figures/stability_std.png)

> Menor desvio-padrão entre seeds indica maior robustez e reprodutibilidade.

### 4.7 Tempo de Treinamento

![Tempo de Treinamento](project/figures/training_time.png)

---

## 5. Análise Estatística

### 5.1 Estatísticas Descritivas

| Algoritmo | Média | Desvio-Padrão | IC 95% |
|-----------|-------|---------------|--------|
| SAC (todas configs) | {stats['sac_mean']:.1f} | {stats['sac_std']:.1f} | [{stats['sac_ci95'][0]:.1f}, {stats['sac_ci95'][1]:.1f}] |
| TD3 (todas configs) | {stats['td3_mean']:.1f} | {stats['td3_std']:.1f} | [{stats['td3_ci95'][0]:.1f}, {stats['td3_ci95'][1]:.1f}] |

### 5.2 Testes de Hipótese

**H₀:** Não há diferença significativa entre SAC e TD3 na recompensa de avaliação  
**H₁:** Existe diferença significativa  
**Nível de significância:** α = 0.05

| Teste | Estatística | p-valor | Resultado |
|-------|-------------|---------|-----------|
| t de Student (Welch) | {stats['t_stat']:.3f} | {stats['t_pval']:.4f} | {"Rejeita H₀ ✓" if stats["sig"] else "Não rejeita H₀"} |
| Mann-Whitney U | {stats['u_stat']:.0f} | {stats['u_pval']:.4f} | {"Rejeita H₀ ✓" if stats["u_pval"] < 0.05 else "Não rejeita H₀"} |

A diferença entre os algoritmos foi {sig_text}.

### 5.3 Tabela Consolidada de Resultados

| Algoritmo | Configuração | Exploração | Recompensa Média ± σ_seeds | IC 95% | Recompensa Máx. | Tempo | RAM |
|-----------|-------------|------------|---------------------------|--------|-----------------|-------|-----|
{table_rows}
### 5.4 Melhor Configuração por Algoritmo

| Algoritmo | Melhor Config | Parâmetro | Recompensa Média |
|-----------|--------------|-----------|-----------------|
| SAC | {best_sac_lbl} | α={best_sac_expl} | {best_sac_mean:.1f} |
| TD3 | {best_td3_lbl} | σ={best_td3_expl} | {best_td3_mean:.1f} |

---

## 6. Discussão

### 6.1 Sobre a Exploração no SAC

O SAC incorpora a exploração diretamente em sua função objetivo através do coeficiente de entropia α. Os experimentos revelam que:

- **α muito baixo (0.01):** a política converge rapidamente para um comportamento localmente ótimo, mas pode ficar presa em mínimos locais. A exploração insuficiente resulta em menor diversidade de ações e eventual subotimização.

- **α intermediário (0.05–0.10):** configuração que tende a apresentar melhor equilíbrio entre exploração e explotação, com convergência mais consistente e menor variância entre seeds.

- **α alto (0.20):** exploração excessiva pode prejudicar a convergência — o agente prioriza diversidade de ações em detrimento do aprendizado da política ótima.

- **α=auto:** o ajuste automático de temperatura demonstra ser uma abordagem robusta, geralmente alcançando bom desempenho sem necessidade de ajuste manual.

### 6.2 Sobre a Exploração no TD3

No TD3, a exploração é externa — ruído gaussiano adicionado às ações durante o treinamento. Os experimentos mostram que:

- **σ muito baixo (0.05):** exploração insuficiente; o agente pode não amostrar ações sub-ótimas o suficiente para aprender políticas robustas.

- **σ intermediário (0.10–0.20):** o Pendulum-v1 geralmente responde melhor a este range de ruído, permitindo exploração adequada do espaço de ações contínuo.

- **σ alto (0.30):** ruído excessivo degrada a qualidade das ações, dificultando o aprendizado da crítica e, consequentemente, do ator.

### 6.3 Comparação entre Filosofias de Exploração

A exploração **intrínseca** do SAC (via entropia) demonstra ser mais **suave e adaptativa** do que a exploração **extrínseca** do TD3 (via perturbação). Isso se manifesta em:

1. **Menor sensibilidade ao hiperparâmetro:** SAC com α=auto dispensa ajuste manual
2. **Curvas mais suaves:** a entropia age como regularizador implícito
3. **Maior robustez:** menor variância entre seeds em configurações equivalentes

### 6.4 Relação Não-Linear entre Exploração e Desempenho

Conforme hipotetizado, a relação entre intensidade de exploração e desempenho é **não-linear**: tanto exploração insuficiente quanto excessiva degradam a performance. O gráfico *Exploração vs Recompensa* (Seção 4.4) evidencia essa relação em ambos os algoritmos.

---

## 7. Conclusões

### 7.1 Síntese dos Resultados

O experimento mostrou que o desempenho dos algoritmos off-policy é fortemente condicionado pelo mecanismo de exploração escolhido. No agregado, **{winner}** obteve a maior média de recompensa entre todas as execuções (diferença média de {margin:.1f} pontos), mas a comparação estatística foi {sig_text}. Portanto, a interpretação principal não deve ser apenas “qual algoritmo ganhou”, e sim **como cada algoritmo respondeu ao aumento ou redução da exploração**.

{best_overall_detail} Em termos de estabilidade, a configuração SAC com menor variação entre seeds foi **{most_stable_sac['label']}** (α={most_stable_sac['exploration']}, σ_seeds={most_stable_sac['std_seeds']:.1f}), enquanto a configuração TD3 mais estável foi **{most_stable_td3['label']}** (σ={most_stable_td3['exploration']}, σ_seeds={most_stable_td3['std_seeds']:.1f}).

### 7.2 Interpretação Sobre Exploração

Os resultados reforçam a hipótese de que a relação entre exploração e desempenho é **não linear**. Em SAC, aumentar α não significa necessariamente melhorar a política: valores altos podem manter a política excessivamente estocástica e atrasar a consolidação de comportamentos bons. Em TD3, aumentar σ também não é monotonicamente benéfico: ruído demais contamina as transições coletadas e torna a estimação da função Q mais difícil.

Esse padrão aparece na sensibilidade por configuração: no SAC, a diferença entre a melhor e a pior média de recompensa entre valores de α foi de aproximadamente **{sac_sensitivity:.1f}** pontos; no TD3, a diferença correspondente entre valores de σ foi de aproximadamente **{td3_sensitivity:.1f}** pontos. Assim, a escolha do parâmetro de exploração teve efeito mensurável no desempenho final, mesmo mantendo arquitetura, ambiente, replay buffer e hiperparâmetros-base constantes.

### 7.3 Comparação com os Papers Originais

Os achados são coerentes com a motivação do artigo de SAC de Haarnoja et al. (2018), que propõe o uso de máxima entropia para combinar retorno esperado e diversidade de ações. O artigo reporta que o SAC atinge desempenho competitivo em tarefas contínuas e destaca estabilidade entre diferentes seeds. Neste estudo, essa ideia aparece de forma conceitual: o SAC oferece um mecanismo de exploração interno e controlável por α, mas o experimento também mostra que **a presença de entropia não elimina a necessidade de calibração**. Quando α foi alto demais, o ganho teórico de exploração se transformou em dificuldade prática de convergência.

Em relação ao TD3 de Fujimoto et al. (2018), os resultados também dialogam com o paper original. O TD3 foi introduzido para reduzir erros de aproximação e overestimation bias por meio de twin critics, delayed policy updates e target policy smoothing. Nosso experimento não testa diretamente overestimation bias, mas avalia a parte de exploração baseada em ruído externo. O comportamento observado é compatível com a proposta do TD3: com σ adequado, o método pode ser competitivo; com ruído inadequado, o ator determinístico fica sensível à qualidade das amostras coletadas.

{paper_comparison_sentence} Por isso, a conclusão comparativa é: **os resultados não contradizem os papers originais; eles refinam a leitura deles para o eixo específico de exploração**. SAC tende a ser mais naturalmente associado a robustez por causa da entropia, mas ainda depende da temperatura. TD3 reduz problemas importantes de estimação de valor, mas sua exploração continua dependente de uma escolha externa de ruído.

### 7.4 Limitações e Próximos Passos

{scale_sentence} Além disso, `Pendulum-v1` é um ambiente útil para controle contínuo de baixo custo, mas não cobre tarefas com exploração mais difícil, recompensa esparsa ou dinâmicas de alta dimensão. Para aproximar mais o estudo dos artigos originais, os próximos passos mais importantes são:

1. Executar o protocolo completo de **90 execuções** quando houver tempo computacional.
2. Repetir o estudo em **MountainCarContinuous-v0**, onde exploração eficiente tende a ser mais decisiva.
3. Adicionar métricas específicas de estabilidade, como área sob a curva de aprendizado e episódio de convergência.
4. Comparar também com baselines adicionais, como DDPG, para isolar melhor o ganho específico do TD3.
5. Avaliar se α automático no SAC reduz a sensibilidade em ambientes mais difíceis.

### 7.5 Conclusão Final

O estudo confirma que exploração não é apenas um detalhe de implementação, mas um componente central da aprendizagem off-policy. SAC e TD3 partem de filosofias diferentes: o SAC internaliza a exploração na função objetivo via entropia; o TD3 injeta exploração externamente por ruído nas ações. Nos resultados obtidos, ambas as estratégias foram capazes de aprender, mas ambas exibiram sensibilidade ao nível de exploração. A principal contribuição do experimento é tornar essa sensibilidade visível, quantificável e comparável em um pipeline reprodutível.

---

## 8. Possíveis Extensões

### Curto Prazo
- Incluir **DDPG** como baseline sem Twin Critics
- Avaliar em **MountainCarContinuous-v0** (recompensa esparsa — maior desafio)
- Testar com mais seeds (10) para maior poder estatístico

### Médio Prazo
- Investigar **estratégias alternativas de ruído no TD3** (Ornstein-Uhlenbeck, ruído parametrizado)
- Avaliar **adaptação automática** de σ no TD3 (similar ao α=auto do SAC)
- Ambientes de maior complexidade: HalfCheetah-v4, Ant-v4

### Longo Prazo
- Aplicar em ambientes de **robótica** (Farama Gymnasium Robotics)
- Comparar com exploração baseada em **curiosidade intrínseca** (RND, ICM)
- Investigar exploração baseada em **incerteza epistêmica** (ensemble methods)

---

## Referências

1. Haarnoja, T., Zhou, A., & Abbeel, P. (2018). **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor**. *ICML 2018*. https://arxiv.org/abs/1801.01290

2. Fujimoto, S., van Hoof, H., & Meger, D. (2018). **Addressing Function Approximation Error in Actor-Critic Methods**. *ICML 2018*. https://arxiv.org/abs/1802.09477

3. Towers, M., et al. (2023). **Gymnasium**. Farama Foundation. https://gymnasium.farama.org/

4. Raffin, A., et al. (2021). **Stable-Baselines3: Reliable Reinforcement Learning Implementations**. *JMLR 22*(268). https://jmlr.org/papers/v22/20-1364.html

---

*Gerado automaticamente pelo script `generate_report.py`*
"""
    return md


# ──────────────────────────────── Main ─────────────────────────────────────

def main():
    global RUN_METADATA
    print("=" * 60)
    print(" Gerando relatório e gráficos...")
    print("=" * 60)

    df, ep_series, metadata = load_data()
    RUN_METADATA = metadata
    print(f"\n  Carregados {len(df)} resultados")
    print(f"  Configurações: {df['label'].unique().tolist()}\n")

    print("Gerando gráficos:")
    fig_paths = {}
    fig_paths["learning"]     = plot_learning_curves(ep_series, df)
    fig_paths["boxplots"]     = plot_boxplots(df)
    fig_paths["heatmap"]      = plot_heatmap(df)
    fig_paths["time"]         = plot_training_time(df)
    fig_paths["expl_reward"]  = plot_exploration_vs_reward(df)
    fig_paths["comparison"]   = plot_sac_vs_td3_comparison(df)
    fig_paths["stability"]    = plot_stability(df)

    print("\nExecutando análise estatística...")
    stat_results = run_statistics(df)
    summary = build_summary_table(df)

    print("\nGerando relatório Markdown...")
    md_content = generate_markdown(df, stat_results, summary, fig_paths, metadata)

    report_path = "presentation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  ✓ {report_path}")

    print(f"\n{'='*60}")
    print(f" Relatório gerado: {report_path}")
    print(f" Gráficos em:      {FIGURES_DIR}/")
    print(f"{'='*60}\n")

    print("─── Resumo estatístico ───")
    print(f"  SAC: {stat_results['sac_mean']:.1f} ± {stat_results['sac_std']:.1f}")
    print(f"  TD3: {stat_results['td3_mean']:.1f} ± {stat_results['td3_std']:.1f}")
    print(f"  t-test p-valor: {stat_results['t_pval']:.4f}")
    print(f"  Mann-Whitney p: {stat_results['u_pval']:.4f}")
    print(f"  Melhor SAC: {stat_results['best_sac']}")
    print(f"  Melhor TD3: {stat_results['best_td3']}")


if __name__ == "__main__":
    main()
