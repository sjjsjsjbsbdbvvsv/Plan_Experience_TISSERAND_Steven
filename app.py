import io
import json
import re
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from itertools import product
from scipy.linalg import hadamard
from scipy.stats import qmc


# ============================================================
# DOE PARAMETRABLE — V4 (Python 3.13 compatible)
# - Plans: Full factorial 2-level, Fractional (simple generator),
#          Screening (Hadamard), RSM (CCD, Box-Behnken), LHS
# - 100% in-app: run sheet + results + analysis + optimization
# - Project save/load JSON
# - Auto-reset when new plan is generated
# ============================================================

st.set_page_config(page_title="DOE complet (V4)", layout="wide")
st.title("DOE complet — Plans + Saisie + Analyse + Optimisation (V4)")


# ----------------------------
# Data models
# ----------------------------
@dataclass
class Factor:
    name: str
    kind: str  # "quant" or "cat"
    low: Optional[float] = None
    high: Optional[float] = None
    step: float = 0.0
    levels: Optional[List[str]] = None  # for cat (2 levels)


@dataclass
class DesignConfig:
    design_type: str
    random_seed: int = 42
    replicates: int = 1
    center_points: int = 0
    n_blocks: int = 1
    randomize_within_block: bool = True
    randomize_global: bool = True
    # Fractionnaire
    frac_generator: str = "a b c ab ac"
    # CCD
    ccd_alpha: str = "rotatable"  # "rotatable" or "face-centered"
    # LHS
    lhs_samples: int = 20


# ----------------------------
# Session state
# ----------------------------
def ensure_state():
    defaults = {
        "plan_id": 0,
        "factors": [],
        "design_cfg": None,
        "doe_coded": None,
        "doe_real": None,
        "results": None,
        "y_cols": ["Y"],
        "analysis_cache": None,
        "analysis_cache_key": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ensure_state()


def reset_everything_for_new_plan(df_real: pd.DataFrame, df_coded: pd.DataFrame, cfg: DesignConfig, factors: List[Factor]):
    """Auto-reset results + analysis when generating a new plan (prevents mixing designs)."""
    st.session_state.plan_id += 1
    st.session_state.design_cfg = cfg
    st.session_state.factors = factors
    st.session_state.doe_real = df_real
    st.session_state.doe_coded = df_coded

    # Reset responses
    st.session_state.y_cols = ["Y"]
    res = df_real.copy()
    res["Done"] = False
    res["Comment"] = ""
    res["Y"] = np.nan
    st.session_state.results = res

    # Reset analysis cache
    st.session_state.analysis_cache = None
    st.session_state.analysis_cache_key = None


# ----------------------------
# Utils
# ----------------------------
def sanitize_name(name: str, default: str = "X") -> str:
    name = (name or "").strip()
    name = name.replace(" ", "_")
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    return name if name else default


def round_to_step(values: np.ndarray, step: float) -> np.ndarray:
    if step is None or step == 0:
        return values
    return np.round(values / step) * step


def stable_hash_df(df: pd.DataFrame) -> str:
    b = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(b).hexdigest()


def add_runorder(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df.insert(0, "RunOrder", np.arange(1, len(df) + 1))
    return df


def repeat_df(df: pd.DataFrame, reps: int) -> pd.DataFrame:
    if reps <= 1:
        return df.reset_index(drop=True)
    return pd.concat([df] * reps, ignore_index=True)


def apply_blocking(df_real: pd.DataFrame, df_coded: pd.DataFrame, n_blocks: int, seed: int, randomize_within_block: bool):
    """Add Block and randomize within each block if requested."""
    df_real = df_real.copy()
    df_coded = df_coded.copy()

    if n_blocks <= 1:
        df_real["Block"] = 1
        df_coded["Block"] = 1
        return df_real, df_coded

    n = len(df_real)
    blocks = np.tile(np.arange(1, n_blocks + 1), int(np.ceil(n / n_blocks)))[:n]
    df_real["Block"] = blocks
    df_coded["Block"] = blocks

    if randomize_within_block:
        rng = np.random.RandomState(seed)
        idx = []
        for b in range(1, n_blocks + 1):
            idx_b = df_real.index[df_real["Block"] == b].to_list()
            rng.shuffle(idx_b)
            idx.extend(idx_b)
        df_real = df_real.loc[idx].reset_index(drop=True)
        df_coded = df_coded.loc[idx].reset_index(drop=True)

    return df_real, df_coded


def coded_to_real_quant(coded: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map coded -> real (supports coded beyond [-1,1] for CCD axial points)."""
    return low + (coded + 1.0) * (high - low) / 2.0


def build_real_from_coded(df_coded: pd.DataFrame, factors: List[Factor]) -> pd.DataFrame:
    """Convert coded design to real values."""
    df_real = df_coded.copy()
    for f in factors:
        if f.kind == "quant":
            x = coded_to_real_quant(df_coded[f.name].astype(float).values, float(f.low), float(f.high))
            x = round_to_step(x, float(f.step))
            df_real[f.name] = x
        else:
            # categorical (2 levels): -1 => levels[0], +1/0 => levels[1] (0 treated as high level for simplicity)
            lv = f.levels or ["A", "B"]
            if len(lv) < 2:
                lv = lv + ["B"]
            df_real[f.name] = np.where(df_coded[f.name].astype(float).values >= 0, lv[1], lv[0])
    return df_real


def make_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="DOE")
    return buffer.getvalue()


def is_two_level_only(df_coded: pd.DataFrame, factor_cols: List[str]) -> bool:
    """True if all factor columns only contain -1 and +1 (no 0, no alpha)."""
    for c in factor_cols:
        vals = set(pd.Series(df_coded[c]).dropna().unique().tolist())
        if not vals.issubset({-1.0, 1.0}):
            return False
    return True


# ----------------------------
# Design generation (coded)
# ----------------------------
def _ff2n(k: int) -> np.ndarray:
    return np.array(list(product([-1.0, 1.0], repeat=k)), dtype=float)


def _fracfact_simple(generator_terms: List[str]) -> np.ndarray:
    """
    Simple fractional factorial from terms like: a b c ab ac
    - Determine base letters from union of letters in terms
    - Generate full factorial on base letters
    - Each term = product of corresponding base columns
    """
    terms = [t.strip().lower() for t in generator_terms if t.strip()]
    if not terms:
        raise ValueError("Generator vide. Exemple: 'a b c ab ac'")

    letters = sorted(set("".join(terms)))
    letters = [ch for ch in letters if ch.isalpha()]
    p = len(letters)
    if p == 0:
        raise ValueError("Generator invalide (pas de lettres).")

    base = _ff2n(p)
    base_df = pd.DataFrame(base, columns=letters)

    cols = []
    for t in terms:
        t = "".join([ch for ch in t if ch.isalpha()])
        v = np.ones(len(base_df), dtype=float)
        for ch in t:
            v *= base_df[ch].values
        cols.append(v)

    return np.column_stack(cols).astype(float)


def _screening_hadamard(k: int) -> np.ndarray:
    """
    Screening design based on Hadamard:
    - N = smallest power of 2 >= k+1
    - Use columns 1..k (skip first all-ones column)
    """
    N = 1
    while N < (k + 1):
        N *= 2
    H = hadamard(N).astype(float)
    return H[:, 1:k+1]


def _ccd_design(k: int, center_points: int = 0, alpha_mode: str = "rotatable") -> np.ndarray:
    """
    CCD coded design:
    - factorial points: +/-1 (2^k)
    - axial points: +/-alpha on each axis (2k)
    - center points: 0
    alpha rotatable ~ sqrt(k), face-centered alpha=1
    """
    alpha = 1.0 if alpha_mode == "face-centered" else float(np.sqrt(k))

    factorial = _ff2n(k)
    axial = []
    for i in range(k):
        v = np.zeros(k, dtype=float)
        v[i] = alpha
        axial.append(v.copy())
        v[i] = -alpha
        axial.append(v.copy())
    axial = np.array(axial, dtype=float)

    center = np.zeros((int(center_points), k), dtype=float) if center_points > 0 else np.zeros((0, k), dtype=float)
    return np.vstack([factorial, axial, center])


def _bb_design(k: int, center_points: int = 0) -> np.ndarray:
    """
    Box–Behnken coded design:
    - For each pair (i,j): 4 points (±1,±1) on i,j, others 0
    - + center points
    Requires k >= 3
    """
    if k < 3:
        raise ValueError("Box–Behnken nécessite au moins 3 facteurs.")
    runs = []
    for i in range(k):
        for j in range(i + 1, k):
            for si in [-1.0, 1.0]:
                for sj in [-1.0, 1.0]:
                    v = np.zeros(k, dtype=float)
                    v[i] = si
                    v[j] = sj
                    runs.append(v)
    X = np.array(runs, dtype=float)
    if center_points > 0:
        X = np.vstack([X, np.zeros((int(center_points), k), dtype=float)])
    return X


def _lhs_design(k: int, samples: int = 20, seed: int = 42) -> np.ndarray:
    """Latin Hypercube in [-1, 1]."""
    sampler = qmc.LatinHypercube(d=k, seed=seed)
    X01 = sampler.random(n=int(samples))
    return (2 * X01 - 1).astype(float)


def generate_coded_matrix(design_type: str, factor_names: List[str], cfg: DesignConfig) -> pd.DataFrame:
    k = len(factor_names)

    if design_type == "Factoriel complet (2 niveaux)":
        X = _ff2n(k)
        return pd.DataFrame(X, columns=factor_names)

    if design_type == "Fractionnaire (2 niveaux)":
        gen = (cfg.frac_generator or "").strip()
        if not gen:
            raise ValueError("Generator fractionnaire vide. Exemple: 'a b c ab ac'.")
        terms = gen.split()
        X = _fracfact_simple(terms)
        if X.shape[1] != k:
            raise ValueError(f"Le generator crée {X.shape[1]} colonnes, mais tu as {k} facteurs.")
        return pd.DataFrame(X, columns=factor_names)

    if design_type == "Screening (Hadamard)":
        X = _screening_hadamard(k)
        return pd.DataFrame(X, columns=factor_names)

    if design_type == "CCD (RSM)":
        X = _ccd_design(k, center_points=cfg.center_points, alpha_mode=cfg.ccd_alpha)
        return pd.DataFrame(X, columns=factor_names)

    if design_type == "Box–Behnken (RSM)":
        X = _bb_design(k, center_points=cfg.center_points)
        return pd.DataFrame(X, columns=factor_names)

    if design_type == "LHS (exploration)":
        X = _lhs_design(k, samples=cfg.lhs_samples, seed=cfg.random_seed)
        return pd.DataFrame(X, columns=factor_names)

    raise ValueError("Type de plan non supporté.")


# ----------------------------
# Analysis helpers
# ----------------------------
def compute_effects_two_level(df_coded: pd.DataFrame, y: pd.Series, include_interactions: bool = True) -> pd.DataFrame:
    """DOE effects (difference of means) for 2-level designs only."""
    tmp = df_coded.copy()
    tmp["_Y_"] = y.values

    cols = [c for c in df_coded.columns if c not in ["RunOrder", "Block"]]
    effects = []

    for col in cols:
        hi = tmp.loc[tmp[col] == 1, "_Y_"].mean()
        lo = tmp.loc[tmp[col] == -1, "_Y_"].mean()
        effects.append((col, hi - lo))

    if include_interactions and len(cols) >= 2:
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                name = f"{cols[i]}:{cols[j]}"
                inter = tmp[cols[i]] * tmp[cols[j]]
                hi = tmp.loc[inter == 1, "_Y_"].mean()
                lo = tmp.loc[inter == -1, "_Y_"].mean()
                effects.append((name, hi - lo))

    eff = pd.DataFrame(effects, columns=["Terme", "Effet"])
    eff["|Effet|"] = eff["Effet"].abs()
    return eff.sort_values("|Effet|", ascending=False).reset_index(drop=True)


def standardized_coeff_pareto(model) -> pd.DataFrame:
    """
    For non-2level designs (RSM), we provide a pareto-like ranking using
    |t| or |coef/std_err| (t-values) from the fitted model.
    """
    tvals = model.tvalues.copy()
    tvals = tvals.drop(labels=["Intercept"], errors="ignore")
    df = pd.DataFrame({"Terme": tvals.index, "|t|": np.abs(tvals.values)})
    return df.sort_values("|t|", ascending=False).reset_index(drop=True)


def build_formula(y_col: str, factors: List[Factor], include_inter: bool, include_quad: bool, include_block: bool) -> str:
    x_terms = []
    quant = []
    for f in factors:
        if f.kind == "quant":
            x_terms.append(f.name)
            quant.append(f.name)
        else:
            x_terms.append(f"C({f.name})")

    base = " + ".join(x_terms) if x_terms else "1"
    formula = f"{y_col} ~ {base}"

    # interactions among quantitative factors only (stable)
    if include_inter and len(quant) >= 2:
        inter_terms = []
        for i in range(len(quant)):
            for j in range(i + 1, len(quant)):
                inter_terms.append(f"{quant[i]}:{quant[j]}")
        if inter_terms:
            formula += " + " + " + ".join(inter_terms)

    # quadratic for quantitative factors
    if include_quad and len(quant) >= 1:
        quad_terms = [f"I({q}**2)" for q in quant]
        formula += " + " + " + ".join(quad_terms)

    if include_block:
        formula += " + C(Block)"

    return formula


# ----------------------------
# Optimization (desirability)
# ----------------------------
def desirability_single(y_pred: float, goal: str, low: float, high: float, target: Optional[float] = None) -> float:
    if goal == "Maximiser":
        if y_pred <= low: return 0.0
        if y_pred >= high: return 1.0
        return (y_pred - low) / (high - low)
    if goal == "Minimiser":
        if y_pred >= high: return 0.0
        if y_pred <= low: return 1.0
        return (high - y_pred) / (high - low)

    # Cibler
    if target is None:
        target = (low + high) / 2
    if y_pred <= low or y_pred >= high:
        return 0.0
    if y_pred == target:
        return 1.0
    if y_pred < target:
        return (y_pred - low) / (target - low)
    return (high - y_pred) / (high - target)


def predict_from_model(model, x_dict: Dict[str, float]) -> float:
    df = pd.DataFrame([x_dict])
    return float(model.predict(df)[0])


def optimize_desirability(models: Dict[str, object], factors: List[Factor], goals: Dict[str, dict], n_samples: int = 2000, seed: int = 42):
    rng = np.random.RandomState(seed)
    quant = [f for f in factors if f.kind == "quant"]
    if not quant:
        return None

    lows = np.array([float(f.low) for f in quant])
    highs = np.array([float(f.high) for f in quant])

    bestD = -1.0
    best_x = None
    best_preds = None

    X = rng.uniform(lows, highs, size=(int(n_samples), len(quant)))
    for row in X:
        x = {quant[i].name: float(row[i]) for i in range(len(quant))}
        d_list = []
        preds = {}
        for yname, model in models.items():
            yp = predict_from_model(model, x)
            preds[yname] = yp
            g = goals[yname]
            d = desirability_single(yp, g["goal"], g["low"], g["high"], g.get("target"))
            d_list.append(d ** g.get("weight", 1.0))
        D = float(np.prod(d_list) ** (1.0 / max(1, len(d_list))))
        if D > bestD:
            bestD = D
            best_x = x
            best_preds = preds

    return bestD, best_x, best_preds


# ============================================================
# UI NAV
# ============================================================
st.sidebar.markdown("## Navigation")
step = st.sidebar.radio(
    "Étapes",
    ["Projet", "Facteurs", "Plan", "Exécution", "Analyse", "Optimisation", "Rapport"],
    index=2
)
st.sidebar.markdown("---")
st.sidebar.caption("Nouveau plan ⇒ reset Exécution + Analyse (automatique).")


# ============================================================
# 0) PROJET
# ============================================================
if step == "Projet":
    st.subheader("Projet — sauvegarder / charger")

    if st.session_state.results is not None and st.session_state.doe_real is not None:
        project = {
            "factors": [asdict(f) for f in st.session_state.factors],
            "design_cfg": asdict(st.session_state.design_cfg) if st.session_state.design_cfg else None,
            "doe_real": st.session_state.doe_real.to_dict(orient="list"),
            "doe_coded": st.session_state.doe_coded.to_dict(orient="list") if st.session_state.doe_coded is not None else None,
            "results": st.session_state.results.to_dict(orient="list"),
            "y_cols": st.session_state.y_cols,
        }
        b = json.dumps(project, ensure_ascii=False).encode("utf-8")
        st.download_button("Télécharger le projet (.json)", b, file_name="doe_project.json", mime="application/json")
    else:
        st.info("Génère un plan (Facteurs → Plan) pour pouvoir sauvegarder.")

    st.markdown("---")
    up = st.file_uploader("Charger un projet (.json)", type=["json"])
    if up is not None:
        try:
            proj = json.loads(up.read().decode("utf-8"))
            st.session_state.factors = [Factor(**f) for f in proj.get("factors", [])]
            dc = proj.get("design_cfg")
            st.session_state.design_cfg = DesignConfig(**dc) if dc else None
            st.session_state.doe_real = pd.DataFrame(proj.get("doe_real"))
            st.session_state.doe_coded = pd.DataFrame(proj.get("doe_coded")) if proj.get("doe_coded") else None
            st.session_state.results = pd.DataFrame(proj.get("results"))
            st.session_state.y_cols = proj.get("y_cols", ["Y"])
            st.session_state.analysis_cache = None
            st.session_state.analysis_cache_key = None
            st.success("Projet chargé ✅")
        except Exception as e:
            st.error(f"Erreur chargement projet: {e}")


# ============================================================
# 1) FACTEURS
# ============================================================
if step == "Facteurs":
    st.subheader("Facteurs")
    st.write("Définis tes facteurs. Quantitatif (bornes + pas) ou Qualitatif (2 niveaux).")

    n_factors = st.number_input("Nombre de facteurs", 1, 25, 3, 1)

    factors: List[Factor] = []
    any_issue = False

    for i in range(int(n_factors)):
        st.markdown(f"### Facteur {i+1}")
        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            name = sanitize_name(st.text_input("Nom", value=f"X{i+1}", key=f"fx_name_{i}"), default=f"X{i+1}")
        with c2:
            kind = st.selectbox("Type", ["quant", "cat"], index=0, key=f"fx_kind_{i}")
        with c3:
            stepv = float(st.number_input("Pas/arrondi (quant)", value=0.0, min_value=0.0, key=f"fx_step_{i}"))

        if kind == "quant":
            c4, c5 = st.columns([1, 1])
            with c4:
                low = float(st.number_input("Bas", value=0.0, key=f"fx_low_{i}"))
            with c5:
                high = float(st.number_input("Haut", value=1.0, key=f"fx_high_{i}"))
            if high == low:
                any_issue = True
                st.warning("⚠️ Bas = Haut (corrige).")
            factors.append(Factor(name=name, kind="quant", low=low, high=high, step=stepv))
        else:
            lv1 = st.text_input("Niveau bas (-1)", value="A", key=f"fx_lv1_{i}")
            lv2 = st.text_input("Niveau haut (+1)", value="B", key=f"fx_lv2_{i}")
            factors.append(Factor(name=name, kind="cat", levels=[lv1, lv2]))

    if any_issue:
        st.error("Corrige d’abord les facteurs quantitatifs où Bas = Haut.")
    else:
        st.session_state.factors = factors
        st.success("Facteurs enregistrés ✅ (passe à Plan)")


# ============================================================
# 2) PLAN
# ============================================================
if step == "Plan":
    st.subheader("Plan d'expérience")
    st.write("Choisis le type de plan. L’app génère le DOE, applique blocs/randomisation, puis reset Exécution+Analyse.")

    if not st.session_state.factors:
        st.warning("Définis d’abord les facteurs (étape Facteurs).")
        st.stop()

    factors = st.session_state.factors
    factor_names = [f.name for f in factors]
    k = len(factor_names)

    left, right = st.columns([1, 2])
    with left:
        design_type = st.selectbox(
            "Type de plan",
            [
                "Factoriel complet (2 niveaux)",
                "Fractionnaire (2 niveaux)",
                "Screening (Hadamard)",
                "CCD (RSM)",
                "Box–Behnken (RSM)",
                "LHS (exploration)",
            ],
            index=0
        )

        seed = st.number_input("Graine (random)", 0, 10000, 42, 1)
        replicates = st.number_input("Réplicats", 1, 50, 1, 1)
        center_points = st.number_input("Points centraux (selon plan)", 0, 50, 0, 1)

        st.markdown("### Blocage (Blocks)")
        n_blocks = st.number_input("Nombre de blocs", 1, 30, 1, 1)
        randomize_within_block = st.checkbox("Randomiser dans chaque bloc", value=True)
        randomize_global = st.checkbox("Randomiser global (si pas de blocs)", value=True)

        frac_gen = "a b c ab ac"
        ccd_alpha = "rotatable"
        lhs_samples = 20

        if design_type == "Fractionnaire (2 niveaux)":
            st.caption("Exemple (5 facteurs): `a b c ab ac` (doit produire exactement k colonnes).")
            frac_gen = st.text_input("Generator", value="a b c ab ac")

        if design_type == "CCD (RSM)":
            ccd_alpha = st.selectbox("Alpha", ["rotatable", "face-centered"], index=0)

        if design_type == "LHS (exploration)":
            lhs_samples = st.number_input("Nb d'échantillons", 5, 500, 20, 1)

    with right:
        # quick estimates (informative)
        if design_type == "Factoriel complet (2 niveaux)":
            est = (2 ** k + center_points) * replicates
        elif design_type == "Screening (Hadamard)":
            N = 1
            while N < (k + 1):
                N *= 2
            est = f"{N} essais (puissance de 2)"
        elif design_type == "CCD (RSM)":
            est = f"{2**k + 2*k + center_points} (approx)"
        elif design_type == "Box–Behnken (RSM)":
            est = f"{2*k*(k-1) + center_points} (approx)"
        elif design_type == "LHS (exploration)":
            est = lhs_samples * replicates
        else:
            est = "selon generator"
        st.info(f"Estimation essais: {est}")

        cfg = DesignConfig(
            design_type=design_type,
            random_seed=int(seed),
            replicates=int(replicates),
            center_points=int(center_points),
            n_blocks=int(n_blocks),
            randomize_within_block=bool(randomize_within_block),
            randomize_global=bool(randomize_global),
            frac_generator=frac_gen,
            ccd_alpha=ccd_alpha,
            lhs_samples=int(lhs_samples),
        )

        if st.button("✅ Générer le plan (reset Exécution+Analyse)"):
            try:
                df_coded = generate_coded_matrix(design_type, factor_names, cfg)
            except Exception as e:
                st.error(f"Erreur génération plan: {e}")
                st.stop()

            # Add extra center points ONLY for plans that do not already contain centers structurally
            if design_type in ["Factoriel complet (2 niveaux)", "Fractionnaire (2 niveaux)", "Screening (Hadamard)", "LHS (exploration)"]:
                if cfg.center_points > 0:
                    centers = pd.DataFrame({name: 0.0 for name in factor_names}, index=range(cfg.center_points))
                    df_coded = pd.concat([df_coded, centers], ignore_index=True)

            # Replicates
            df_coded = repeat_df(df_coded, cfg.replicates)

            # Convert to real
            df_real = build_real_from_coded(df_coded, factors)

            # Blocking
            df_real, df_coded = apply_blocking(df_real, df_coded, cfg.n_blocks, cfg.random_seed, cfg.randomize_within_block)

            # Randomize global if no blocks
            if cfg.n_blocks <= 1 and cfg.randomize_global:
                df_real = df_real.sample(frac=1, random_state=cfg.random_seed).reset_index(drop=True)
                df_coded = df_coded.loc[df_real.index].reset_index(drop=True)

            # RunOrder
            df_real = add_runorder(df_real)
            df_coded = add_runorder(df_coded)

            reset_everything_for_new_plan(df_real, df_coded, cfg, factors)

            st.success("Plan généré ✅ Va à Exécution.")
            st.dataframe(st.session_state.doe_real, use_container_width=True)

            st.download_button("Télécharger CSV (optionnel)", st.session_state.doe_real.to_csv(index=False).encode("utf-8"),
                               file_name="plan.csv", mime="text/csv")
            st.download_button("Télécharger Excel (optionnel)", make_excel_bytes(st.session_state.doe_real),
                               file_name="plan.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ============================================================
# 3) EXECUTION
# ============================================================
if step == "Exécution":
    st.subheader("Exécution — saisie des mesures dans l’app")
    st.write("Remplis Done / Comment, et ajoute une ou plusieurs réponses Y.")

    if st.session_state.results is None:
        st.warning("Génère un plan d’abord (Facteurs → Plan).")
        st.stop()

    c1, c2 = st.columns([2, 1])
    with c2:
        st.markdown("### Réponses (Y)")
        new_y = st.text_input("Ajouter une réponse", placeholder="Ex: Poids, Retrait, Y2", value="")
        if st.button("➕ Ajouter"):
            ny = sanitize_name(new_y, default="")
            if not ny:
                st.info("Nom vide.")
            elif ny in st.session_state.y_cols:
                st.info("Existe déjà.")
            else:
                st.session_state.y_cols.append(ny)
                st.session_state.results[ny] = np.nan
                st.session_state.analysis_cache = None
                st.session_state.analysis_cache_key = None
                st.success(f"Ajouté: {ny}")

        if st.button("🧹 Vider toutes les Y"):
            for y in st.session_state.y_cols:
                st.session_state.results[y] = np.nan
            st.session_state.results["Done"] = False
            st.session_state.results["Comment"] = ""
            st.session_state.analysis_cache = None
            st.session_state.analysis_cache_key = None
            st.success("Réponses vidées.")

        st.markdown("---")
        st.caption("Remplissage:")
        for y in st.session_state.y_cols:
            st.write(f"- {y}: {int(st.session_state.results[y].notna().sum())}/{len(st.session_state.results)}")

    with c1:
        edited = st.data_editor(
            st.session_state.results,
            use_container_width=True,
            num_rows="fixed",
            key=f"exec_editor_{st.session_state.plan_id}",
        )
        st.session_state.results = edited


# ============================================================
# 4) ANALYSE
# ============================================================
if step == "Analyse":
    st.subheader("Analyse — modèle, ANOVA, Pareto, diagnostics")
    st.write("Choisis une réponse Y et un modèle. L’app bloque si le modèle est trop riche pour N.")

    if st.session_state.results is None or st.session_state.doe_coded is None:
        st.warning("Génère un plan et saisis des résultats d’abord.")
        st.stop()

    factors = st.session_state.factors
    df_res = st.session_state.results.copy()
    df_coded = st.session_state.doe_coded.copy()

    y_candidates = [y for y in st.session_state.y_cols if y in df_res.columns]
    y_col = st.selectbox("Réponse (Y)", options=y_candidates)

    include_inter = st.checkbox("Interactions (quant-quant)", value=True)
    include_quad = st.checkbox("Quadratique (RSM)", value=False)
    include_block = st.checkbox("Inclure Block", value=True)

    mask = df_res[y_col].notna()
    n_obs = int(mask.sum())
    if n_obs == 0:
        st.warning("Aucune valeur Y remplie. Va à Exécution.")
        st.stop()

    df_model = df_res.loc[mask].copy()
    if "Block" not in df_model.columns:
        df_model["Block"] = 1

    formula = build_formula(y_col, factors, include_inter, include_quad, include_block)
    st.code(formula)

    # Guardrail using a quick fit to estimate parameters
    try:
        tmp_model = smf.ols(formula=formula, data=df_model).fit()
        p = int(tmp_model.df_model) + 1
    except Exception:
        p = 10

    if n_obs <= p:
        st.error(f"Pas assez d’essais pour ce modèle: N={n_obs} <= paramètres~{p}. Réduis le modèle ou ajoute essais.")
        st.stop()

    cache_key = stable_hash_df(df_model[["RunOrder", "Block"] + [f.name for f in factors] + [y_col]]) + f"|{formula}|plan={st.session_state.plan_id}"

    if st.button("🚀 Lancer l’analyse"):
        if st.session_state.analysis_cache_key == cache_key and st.session_state.analysis_cache is not None:
            st.info("Analyse déjà calculée (cache) ✅")
        else:
            try:
                model = smf.ols(formula=formula, data=df_model).fit()
                try:
                    anova_tbl = anova_lm(model, typ=2)
                except Exception:
                    anova_tbl = None

                factor_cols = [f.name for f in factors]
                coded_cols = [c for c in df_coded.columns if c not in ["RunOrder", "Block"]]
                two_level = is_two_level_only(df_coded.loc[mask.values].reset_index(drop=True), coded_cols)

                if two_level:
                    eff = compute_effects_two_level(df_coded.loc[mask.values, coded_cols].reset_index(drop=True),
                                                    df_res.loc[mask, y_col].reset_index(drop=True),
                                                    include_interactions=include_inter)
                else:
                    eff = None

                pareto_std = standardized_coeff_pareto(model)

                st.session_state.analysis_cache = {
                    "model": model,
                    "anova": anova_tbl,
                    "effects": eff,
                    "pareto_std": pareto_std,
                    "two_level": two_level
                }
                st.session_state.analysis_cache_key = cache_key
                st.success("Analyse calculée ✅")
            except Exception as e:
                st.error(f"Erreur analyse: {e}")

    if st.session_state.analysis_cache is None:
        st.info("Clique sur **Lancer l’analyse**.")
        st.stop()

    model = st.session_state.analysis_cache["model"]
    anova_tbl = st.session_state.analysis_cache["anova"]
    eff = st.session_state.analysis_cache["effects"]
    pareto_std = st.session_state.analysis_cache["pareto_std"]
    two_level = st.session_state.analysis_cache["two_level"]

    st.markdown("## Coefficients")
    coef = pd.DataFrame({"coef": model.params, "std_err": model.bse, "t": model.tvalues, "p_value": model.pvalues})
    st.dataframe(coef, use_container_width=True)

    st.markdown("## Qualité d’ajustement")
    fit_stats = pd.DataFrame({
        "R²": [model.rsquared],
        "R² ajusté": [model.rsquared_adj],
        "AIC": [model.aic],
        "BIC": [model.bic],
        "N": [int(model.nobs)]
    })
    st.dataframe(fit_stats, use_container_width=True)

    st.markdown("## ANOVA")
    if anova_tbl is None:
        st.warning("ANOVA indisponible pour ce modèle/données.")
    else:
        st.dataframe(anova_tbl, use_container_width=True)

    st.markdown("## Pareto")
    if two_level and eff is not None:
        st.caption("Design 2 niveaux détecté → Pareto sur effets (différence de moyennes).")
        st.dataframe(eff, use_container_width=True)
        figp = plt.figure()
        plt.bar(range(len(eff)), eff["|Effet|"].values)
        plt.xticks(range(len(eff)), eff["Terme"].values, rotation=90)
        plt.ylabel("|Effet|")
        plt.title("Pareto des effets (DOE)")
        plt.tight_layout()
        st.pyplot(figp)
    else:
        st.caption("Design non 2 niveaux (ex: RSM) → Pareto basé sur |t| (coefficients standardisés).")
        st.dataframe(pareto_std, use_container_width=True)
        figp = plt.figure()
        plt.bar(range(len(pareto_std)), pareto_std["|t|"].values)
        plt.xticks(range(len(pareto_std)), pareto_std["Terme"].values, rotation=90)
        plt.ylabel("|t|")
        plt.title("Pareto (|t| des coefficients)")
        plt.tight_layout()
        st.pyplot(figp)

    st.markdown("## Diagnostics")
    resid = model.resid
    fitted = model.fittedvalues

    f1 = plt.figure()
    plt.scatter(fitted, resid)
    plt.axhline(0)
    plt.xlabel("Ajustés")
    plt.ylabel("Résidus")
    plt.title("Résidus vs Ajustés")
    st.pyplot(f1)

    f2 = plt.figure()
    sm.qqplot(resid, line="45", fit=True)
    plt.title("QQ-plot résidus")
    st.pyplot(f2)

    f3 = plt.figure()
    plt.scatter(df_model["RunOrder"], df_model[y_col])
    plt.xlabel("RunOrder")
    plt.ylabel(y_col)
    plt.title("Réponse vs ordre (dérive ?)")
    st.pyplot(f3)


# ============================================================
# 5) OPTIMISATION
# ============================================================
if step == "Optimisation":
    st.subheader("Optimisation — 1 ou multi-réponses (désirabilité)")
    st.write("L’optimisation utilise des modèles ajustés. Elle optimise uniquement les facteurs quantitatifs.")

    if st.session_state.results is None:
        st.warning("Génère un plan, saisis des données, puis analyse.")
        st.stop()

    factors = st.session_state.factors
    df_res = st.session_state.results.copy()

    y_candidates = [y for y in st.session_state.y_cols if y in df_res.columns]
    selected_y = st.multiselect("Réponses à optimiser", y_candidates, default=y_candidates[:1] if y_candidates else [])
    if not selected_y:
        st.info("Choisis au moins une réponse.")
        st.stop()

    include_inter = st.checkbox("Modèle: interactions (quant-quant)", value=True)
    include_quad = st.checkbox("Modèle: quadratique (RSM)", value=False)
    include_block = st.checkbox("Inclure Block", value=True)

    st.markdown("### Objectifs / bornes désirées")
    goals = {}
    for y in selected_y:
        st.markdown(f"**{y}**")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            goal = st.selectbox(f"Objectif ({y})", ["Maximiser", "Minimiser", "Cibler"], key=f"goal_{y}")
        with c2:
            low = float(st.number_input(f"Seuil bas (D=0) {y}", value=0.0, key=f"low_{y}"))
        with c3:
            high = float(st.number_input(f"Seuil haut (D=1) {y}", value=1.0, key=f"high_{y}"))
        with c4:
            weight = float(st.number_input(f"Poids {y} (>=1)", value=1.0, min_value=1.0, key=f"w_{y}"))
        target = None
        if goal == "Cibler":
            target = float(st.number_input(f"Cible {y}", value=(low + high) / 2, key=f"t_{y}"))
        goals[y] = {"goal": goal, "low": low, "high": high, "target": target, "weight": weight}

    # Fit models for selected Y
    models = {}
    for y in selected_y:
        mask = df_res[y].notna()
        if int(mask.sum()) < 6:
            st.warning(f"Pas assez de points pour {y} (>=6 recommandé).")
            st.stop()
        df_model = df_res.loc[mask].copy()
        if "Block" not in df_model.columns:
            df_model["Block"] = 1
        formula = build_formula(y, factors, include_inter, include_quad, include_block)
        try:
            models[y] = smf.ols(formula=formula, data=df_model).fit()
        except Exception as e:
            st.error(f"Impossible d’ajuster le modèle pour {y}: {e}")
            st.stop()

    n_samples = st.slider("Nb d'essais virtuels", 200, 20000, 2000, 200)

    if st.button("🎯 Optimiser"):
        out = optimize_desirability(models, factors, goals, n_samples=int(n_samples), seed=42)
        if out is None:
            st.error("Aucun facteur quantitatif à optimiser (cat seulement).")
            st.stop()
        bestD, best_x, best_preds = out
        st.success(f"Désirabilité globale = {bestD:.3f}")
        st.markdown("#### Réglages recommandés (quantitatifs)")
        st.dataframe(pd.DataFrame([best_x]), use_container_width=True)
        st.markdown("#### Prédictions")
        st.dataframe(pd.DataFrame([best_preds]), use_container_width=True)


# ============================================================
# 6) RAPPORT
# ============================================================
if step == "Rapport":
    st.subheader("Rapport (résumé)")
    if st.session_state.doe_real is None:
        st.info("Génère un plan d’abord.")
        st.stop()

    lines = []
    lines.append("# Rapport DOE\n")
    lines.append(f"- Plan ID: {st.session_state.plan_id}\n")
    if st.session_state.design_cfg:
        lines.append(f"- Type: {st.session_state.design_cfg.design_type}\n")
        lines.append(f"- Réplicats: {st.session_state.design_cfg.replicates}\n")
        lines.append(f"- Points centraux: {st.session_state.design_cfg.center_points}\n")
        lines.append(f"- Blocs: {st.session_state.design_cfg.n_blocks}\n")

    lines.append("\n## Facteurs\n")
    for f in st.session_state.factors:
        if f.kind == "quant":
            lines.append(f"- {f.name}: [{f.low}, {f.high}], pas={f.step}\n")
        else:
            lines.append(f"- {f.name}: cat {f.levels}\n")

    lines.append("\n## Aperçu plan (10 lignes)\n")
    lines.append(st.session_state.doe_real.head(10).to_markdown(index=False))

    if st.session_state.results is not None:
        lines.append("\n\n## Remplissage réponses\n")
        for y in st.session_state.y_cols:
            if y in st.session_state.results.columns:
                lines.append(f"- {y}: {int(st.session_state.results[y].notna().sum())}/{len(st.session_state.results)}\n")

    md = "\n".join(lines).encode("utf-8")
    st.download_button("Télécharger rapport (MD)", md, file_name="rapport_doe.md", mime="text/markdown")

