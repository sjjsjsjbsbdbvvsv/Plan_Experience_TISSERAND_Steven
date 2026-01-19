# ============================================================
# DOE PARAMETRABLE — V4 "niveau Minitab +" (Streamlit)
# - Plans: Factoriel complet 2 niveaux, Fractionnaire (fracfact),
#          Plackett–Burman (PB), RSM: CCD, Box–Behnken, LHS
# - 100% dans l'app: saisie Y, analyse (ANOVA/diagnostics/effets),
#   optimisation 1 ou multi-réponses (désirabilité)
# - Projet: sauvegarde/chargement JSON
# - Garde-fous: reset auto si nouveau plan, blocage (blocks), contrôle N vs modèle
# ============================================================

import json
import io
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
from scipy import optimize

from doepy import build



# --------------------------
# UI CONFIG
# --------------------------
st.set_page_config(page_title="DOE complet (V4)", layout="wide")
st.title("DOE complet — Plans + Saisie + Analyse + Optimisation (V4)")


# --------------------------
# DATA MODELS
# --------------------------
@dataclass
class Factor:
    name: str
    kind: str  # "quant" or "cat"
    low: Optional[float] = None
    high: Optional[float] = None
    step: float = 0.0
    levels: Optional[List[str]] = None  # for cat


@dataclass
class DesignConfig:
    design_type: str
    random_seed: int
    replicates: int
    center_points: int
    n_blocks: int
    randomize_within_block: bool
    randomize_global: bool
    # Fractionnaire
    frac_generator: str = ""  # ex: "a b c ab ac"
    # CCD / RSM
    ccd_center: int = 4
    ccd_alpha: str = "rotatable"  # rotatable / face-centered
    ccd_face: str = "circumscribed"  # circumscribed / inscribed / faced
    # LHS
    lhs_samples: int = 20


# --------------------------
# STATE
# --------------------------
def ensure_state():
    defaults = {
        "plan_id": 0,
        "factors": [],
        "design_cfg": None,
        "doe_coded": None,
        "doe_real": None,
        "results": None,            # includes Y columns + status
        "y_cols": ["Y"],
        "analysis_cache": None,
        "analysis_cache_key": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_everything_for_new_plan(df_real: pd.DataFrame, df_coded: pd.DataFrame):
    """Reset complet (résultats + analyse) quand un nouveau plan est généré."""
    st.session_state.plan_id += 1
    st.session_state.doe_real = df_real
    st.session_state.doe_coded = df_coded

    # Reset réponses
    st.session_state.y_cols = ["Y"]

    res = df_real.copy()
    res["Done"] = False
    res["Comment"] = ""
    res["Y"] = np.nan
    st.session_state.results = res

    # Reset analyse
    st.session_state.analysis_cache = None
    st.session_state.analysis_cache_key = None


ensure_state()


# --------------------------
# UTILS
# --------------------------
def sanitize_name(name: str, default: str = "X") -> str:
    name = (name or "").strip()
    name = name.replace(" ", "_")
    # évite caractères bizarres
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
    """Ajoute Block et randomise dans chaque bloc si demandé."""
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
        idx_parts = []
        for b in range(1, n_blocks + 1):
            idx_b = df_real.index[df_real["Block"] == b].to_list()
            rng.shuffle(idx_b)
            idx_parts.extend(idx_b)
        df_real = df_real.loc[idx_parts].reset_index(drop=True)
        df_coded = df_coded.loc[idx_parts].reset_index(drop=True)

    return df_real, df_coded


def coded_to_real_quant(coded: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map codé -> réel via interpolation linéaire. Fonctionne aussi si codé dépasse [-1,1] (CCD)."""
    # coded=-1 -> low, coded=+1 -> high
    return low + (coded + 1.0) * (high - low) / 2.0


def build_real_from_coded(df_coded: pd.DataFrame, factors: List[Factor]) -> pd.DataFrame:
    """Convertit colonnes codées en valeurs réelles (quant) + labels (cat)."""
    df_real = df_coded.copy()

    for f in factors:
        if f.kind == "quant":
            col = f.name
            df_real[col] = coded_to_real_quant(df_coded[col].values.astype(float), float(f.low), float(f.high))
            df_real[col] = round_to_step(df_real[col].values.astype(float), float(f.step))
        else:
            # Catégoriel: pour l’instant support 2 niveaux (PB/factoriel/etc)
            # codé -1/+1 => levels[0]/levels[1]
            col = f.name
            levels = f.levels or ["A", "B"]
            if len(levels) < 2:
                levels = levels + ["B"]
            df_real[col] = np.where(df_coded[col].values.astype(float) >= 0, levels[1], levels[0])

    return df_real


def make_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="DOE")
    return buffer.getvalue()


# --------------------------
# DOE GENERATION (CODED)
# --------------------------
def generate_coded_matrix(design_type: str, factor_names: List[str], cfg: DesignConfig) -> pd.DataFrame:
    k = len(factor_names)

    if design_type == "Factoriel complet (2 niveaux)":
        X = ff2n(k)  # -1/+1
        return pd.DataFrame(X, columns=factor_names)

    if design_type == "Fractionnaire (2 niveaux)":
        # cfg.frac_generator doit contenir une formule fracfact
        # Ex: "a b c ab ac" -> 5 colonnes, mais on gère k colonnes => il faut fournir k termes
        gen = (cfg.frac_generator or "").strip()
        if not gen:
            raise ValueError("Generator fractionnaire vide. Exemple: 'a b c ab ac'.")
        X = fracfact(gen)  # -1/+1
        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(X.shape[1])])
        if X.shape[1] != k:
            raise ValueError(f"Le generator crée {X.shape[1]} colonnes, mais tu as {k} facteurs.")
        df.columns = factor_names
        return df

    if design_type == "Plackett–Burman (screening)":
        X = pbdesign(k)  # -1/+1, N = multiple of 4
        return pd.DataFrame(X, columns=factor_names)

    if design_type == "CCD (RSM)":
        # ccdesign: renvoie niveaux codés incluant alpha, 0, +/-1
        # center=(nc, nc) par défaut: points centraux (factoriels, axiaux)
        alpha_map = {"rotatable": "rotatable", "face-centered": "faced"}
        alpha = alpha_map.get(cfg.ccd_alpha, "rotatable")
        # ccdesign uses alpha parameter values: 'rotatable', 'orthogonal', or numeric
        # for face-centered we can set alpha=1 and face='faced'
        if cfg.ccd_alpha == "face-centered":
            X = ccdesign(k, center=(cfg.center_points, cfg.center_points), alpha=1, face="faced")
        else:
            X = ccdesign(k, center=(cfg.center_points, cfg.center_points), alpha="rotatable", face=cfg.ccd_face)
        return pd.DataFrame(X, columns=factor_names)

    if design_type == "Box–Behnken (RSM)":
        X = bbdesign(k, center=cfg.center_points)  # -1/0/+1
        return pd.DataFrame(X, columns=factor_names)

    if design_type == "LHS (exploration)":
        X = lhs(k, samples=cfg.lhs_samples, criterion="center")  # in [0,1]
        # Convert [0,1] -> [-1,1] for consistent mapping
        X = 2 * X - 1
        return pd.DataFrame(X, columns=factor_names)

    raise ValueError("Type de plan non supporté.")


# --------------------------
# EFFECTS / PARETO
# --------------------------
def compute_effects(df_coded: pd.DataFrame, y: pd.Series, include_interactions: bool = True) -> pd.DataFrame:
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
    eff = eff.sort_values("|Effet|", ascending=False).reset_index(drop=True)
    return eff


# --------------------------
# MODEL BUILDING
# --------------------------
def build_formula(y_col: str, factors: List[Factor], include_interactions: bool, include_quadratic: bool, include_block: bool) -> str:
    # quant factors appear as numeric; categorical as C(name)
    x_terms = []
    quant_names = []
    for f in factors:
        if f.kind == "quant":
            x_terms.append(f.name)
            quant_names.append(f.name)
        else:
            x_terms.append(f"C({f.name})")

    base = " + ".join(x_terms) if x_terms else "1"
    formula = f"{y_col} ~ {base}"

    # interactions: only among quant for now (stable + performant)
    if include_interactions and len(quant_names) >= 2:
        inter = " + ".join([f"{quant_names[i]}:{quant_names[j]}" for i in range(len(quant_names)) for j in range(i+1, len(quant_names))])
        formula += " + " + inter

    # quadratic: only quant (RSM)
    if include_quadratic and len(quant_names) >= 1:
        quad = " + ".join([f"I({q}**2)" for q in quant_names])
        formula += " + " + quad

    if include_block:
        formula += " + C(Block)"

    return formula


def backward_elimination(model, alpha=0.10):
    """Simple backward elimination: drop the worst p-value term iteratively (except intercept)."""
    current = model
    while True:
        pvals = current.pvalues.drop(labels=["Intercept"], errors="ignore")
        if len(pvals) == 0:
            break
        worst_term = pvals.idxmax()
        worst_p = pvals.max()
        if worst_p <= alpha:
            break
        # rebuild formula without worst_term
        # WARNING: simple string removal can be tricky; on reste prudent:
        formula = current.model.formula
        # remove " + worst_term" or worst at ends
        formula2 = re.sub(rf"\s*\+\s*{re.escape(worst_term)}\s*", " + ", formula)
        formula2 = re.sub(r"\+\s*\+\s*", "+", formula2)
        formula2 = formula2.replace("~ +", "~")
        formula2 = re.sub(r"\s+", " ", formula2).strip()
        try:
            current = smf.ols(formula=formula2, data=current.model.data.frame).fit()
        except Exception:
            break
    return current


# --------------------------
# OPTIMIZATION (DESIRABILITY)
# --------------------------
def desirability_single(y_pred: float, goal: str, low: float, high: float, target: float = None) -> float:
    """Desirability 0..1."""
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
    """
    Random search (robuste) + local polish (optionnel).
    - factors: quant only for optimization (on ignore cat ici)
    """
    rng = np.random.RandomState(seed)
    quant = [f for f in factors if f.kind == "quant"]
    if not quant:
        return None

    lows = np.array([float(f.low) for f in quant])
    highs = np.array([float(f.high) for f in quant])

    best = None
    bestD = -1

    # random search
    X = rng.uniform(lows, highs, size=(n_samples, len(quant)))
    for row in X:
        x = {quant[i].name: float(row[i]) for i in range(len(quant))}
        D_list = []
        for yname, model in models.items():
            yp = predict_from_model(model, x)
            g = goals[yname]
            d = desirability_single(yp, g["goal"], g["low"], g["high"], g.get("target"))
            D_list.append(d ** g.get("weight", 1.0))
        D = float(np.prod(D_list) ** (1.0 / max(1, len(D_list))))
        if D > bestD:
            bestD = D
            best = (x, {yname: predict_from_model(models[yname], x) for yname in models.keys()})

    return bestD, best


# ============================================================
# SIDEBAR — NAV (UX)
# ============================================================
st.sidebar.markdown("## Navigation")
step = st.sidebar.radio(
    "Étapes",
    ["Projet", "Facteurs", "Plan", "Exécution", "Analyse", "Optimisation", "Rapport"],
    index=2
)

st.sidebar.markdown("---")
st.sidebar.caption("Astuce: l'app *réinitialise* résultats+analyse à chaque nouveau plan pour éviter les mélanges.")


# ============================================================
# 0) PROJET (save/load)
# ============================================================
if step == "Projet":
    st.subheader("Projet")
    st.write("Ici tu peux **sauvegarder** l’état du projet (facteurs, plan, résultats) et **recharger** plus tard.")

    # Save
    if st.session_state.results is not None and st.session_state.doe_real is not None:
        project = {
            "factors": [asdict(f) for f in st.session_state.factors],
            "design_cfg": asdict(st.session_state.design_cfg) if st.session_state.design_cfg else None,
            "doe_real": st.session_state.doe_real.to_dict(orient="list"),
            "doe_coded": st.session_state.doe_coded.to_dict(orient="list") if st.session_state.doe_coded is not None else None,
            "results": st.session_state.results.to_dict(orient="list"),
            "y_cols": st.session_state.y_cols,
        }
        bytes_json = json.dumps(project, ensure_ascii=False).encode("utf-8")
        st.download_button("Télécharger le projet (.json)", bytes_json, file_name="doe_project.json", mime="application/json")
    else:
        st.info("Génère un plan d’abord (étapes Facteurs + Plan), puis tu pourras sauvegarder.")

    st.markdown("---")

    # Load
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

            # reset caches (sécurité)
            st.session_state.analysis_cache = None
            st.session_state.analysis_cache_key = None
            st.success("Projet chargé ✅")
        except Exception as e:
            st.error(f"Impossible de charger ce projet: {e}")


# ============================================================
# 1) FACTEURS
# ============================================================
if step == "Facteurs":
    st.subheader("Facteurs")
    st.write("Définis tes facteurs. **Quantitatif** (bornes + pas) ou **Qualitatif (2 niveaux)**.")

    colA, colB = st.columns([1, 2])
    with colA:
        n_factors = st.number_input("Nombre de facteurs", 1, 20, 3, 1)
        st.caption("Conseil: pour les qualitatifs, commence simple (2 niveaux).")

    factors: List[Factor] = []
    issues = False
    with colB:
        for i in range(int(n_factors)):
            st.markdown(f"### Facteur {i+1}")
            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                name = sanitize_name(st.text_input("Nom", value=f"X{i+1}", key=f"fx_name_{i}"), default=f"X{i+1}")
            with c2:
                kind = st.selectbox("Type", ["quant", "cat"], index=0, key=f"fx_kind_{i}")
            with c3:
                step = float(st.number_input("Pas/arrondi (quant)", value=0.0, min_value=0.0, key=f"fx_step_{i}"))

            if kind == "quant":
                c4, c5 = st.columns([1, 1])
                with c4:
                    low = float(st.number_input("Bas", value=0.0, key=f"fx_low_{i}"))
                with c5:
                    high = float(st.number_input("Haut", value=1.0, key=f"fx_high_{i}"))
                if high == low:
                    issues = True
                    st.warning("Bas = Haut (corrige).")
                factors.append(Factor(name=name, kind="quant", low=low, high=high, step=step))
            else:
                lv1 = st.text_input("Niveau bas (-1)", value="A", key=f"fx_lv1_{i}")
                lv2 = st.text_input("Niveau haut (+1)", value="B", key=f"fx_lv2_{i}")
                factors.append(Factor(name=name, kind="cat", levels=[lv1, lv2]))

    st.session_state.factors = factors
    st.success("Facteurs enregistrés ✅ (passe à l’étape Plan)")


# ============================================================
# 2) PLAN
# ============================================================
if step == "Plan":
    st.subheader("Plan d’expérience")
    st.write("Choisis le type de plan. L’app génère la matrice, applique réplicats/points centraux/blocs, puis réinitialise Exécution+Analyse.")

    if not st.session_state.factors:
        st.warning("Définis d’abord les facteurs (étape Facteurs).")
        st.stop()

    factors = st.session_state.factors
    factor_names = [f.name for f in factors]
    k = len(factors)

    left, right = st.columns([1, 2])

    with left:
        design_type = st.selectbox(
            "Type de plan",
            [
                "Factoriel complet (2 niveaux)",
                "Fractionnaire (2 niveaux)",
                "Plackett–Burman (screening)",
                "CCD (RSM)",
                "Box–Behnken (RSM)",
                "LHS (exploration)",
            ],
            index=0
        )

        random_seed = st.number_input("Graine (random)", 0, 10000, 42, 1)
        replicates = st.number_input("Réplicats", 1, 50, 1, 1)
        center_points = st.number_input("Points centraux", 0, 50, 0, 1)

        st.markdown("### Blocage (Blocks)")
        n_blocks = st.number_input("Nombre de blocs", 1, 20, 1, 1)
        randomize_within_block = st.checkbox("Randomiser dans chaque bloc", value=True)
        randomize_global = st.checkbox("Randomiser global (si pas de blocs)", value=True)

        frac_generator = ""
        ccd_alpha = "rotatable"
        ccd_face = "circumscribed"
        lhs_samples = 20

        if design_type == "Fractionnaire (2 niveaux)":
            st.markdown("### Fractionnaire — generator")
            st.caption("Exemple (5 facteurs): `a b c ab ac`  (doit produire exactement k colonnes).")
            frac_generator = st.text_input("Generator fracfact", value="a b c ab ac", help="Doit créer k colonnes.")

        if design_type == "CCD (RSM)":
            st.markdown("### CCD — options")
            ccd_alpha = st.selectbox("Alpha", ["rotatable", "face-centered"], index=0)
            ccd_face = st.selectbox("Face", ["circumscribed", "inscribed", "faced"], index=0)

        if design_type == "LHS (exploration)":
            lhs_samples = st.number_input("Nb d'échantillons", 5, 500, 20, 1)

    # preview runs
    with right:
        # Estimate runs quickly (rough)
        est = "?"
        if design_type == "Factoriel complet (2 niveaux)":
            est = (2**k + center_points) * replicates
        elif design_type == "Plackett–Burman (screening)":
            # pbdesign gives N multiple of 4 >= k+1 (roughly)
            est = "≈ multiple de 4"
        elif design_type == "Box–Behnken (RSM)":
            est = "BB = 2k(k-1)+center"
        elif design_type == "CCD (RSM)":
            est = "CCD ≈ 2^k + 2k + centers"
        elif design_type == "LHS (exploration)":
            est = lhs_samples * replicates
        st.info(f"Estimation essais: {est}")

        st.caption("Garde-fou: un nouveau plan ⇒ reset Exécution/Analyse automatiquement.")

        cfg = DesignConfig(
            design_type=design_type,
            random_seed=int(random_seed),
            replicates=int(replicates),
            center_points=int(center_points),
            n_blocks=int(n_blocks),
            randomize_within_block=bool(randomize_within_block),
            randomize_global=bool(randomize_global),
            frac_generator=frac_generator,
            ccd_alpha=ccd_alpha,
            ccd_face=ccd_face,
            lhs_samples=int(lhs_samples),
        )

        if st.button("✅ Générer le plan (reset Exécution+Analyse)"):
            # Build coded matrix
            try:
                df_coded = generate_coded_matrix(design_type, factor_names, cfg)
            except Exception as e:
                st.error(f"Erreur génération plan: {e}")
                st.stop()

            # Add optional center points (for plans that don't already include them)
            # For CCD/BB: they already include center; but we allow extra centers by repeating rows at 0.
            if design_type in ["Factoriel complet (2 niveaux)", "Fractionnaire (2 niveaux)", "Plackett–Burman (screening)", "LHS (exploration)"]:
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

            # Save config + reset
            st.session_state.design_cfg = cfg
            reset_everything_for_new_plan(df_real, df_coded)

            st.success("Plan généré ✅ (va à Exécution pour saisir les mesures)")
            st.dataframe(st.session_state.doe_real, use_container_width=True)

            st.download_button("Télécharger CSV (optionnel)", st.session_state.doe_real.to_csv(index=False).encode("utf-8"),
                               file_name="plan.csv", mime="text/csv")
            st.download_button("Télécharger Excel (optionnel)", make_excel_bytes(st.session_state.doe_real),
                               file_name="plan.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ============================================================
# 3) EXECUTION (saisie dans l'app)
# ============================================================
if step == "Exécution":
    st.subheader("Exécution (Run Sheet) — saisie des mesures dans l’app")
    st.write(
        "Tu peux cocher **Done**, ajouter un **Comment**, et saisir tes réponses **Y** directement. "
        "Aucun Excel nécessaire."
    )

    if st.session_state.results is None:
        st.warning("Génère un plan d’abord (étape Plan).")
        st.stop()

    c1, c2 = st.columns([2, 1])

    with c2:
        st.markdown("### Réponses (Y)")
        new_y = st.text_input("Ajouter une réponse", placeholder="Ex: Poids, Retrait, Y2", value="")
        if st.button("➕ Ajouter une réponse"):
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
        st.caption("Conseil: remplis au moins une réponse Y pour lancer l’analyse.")

        y_fill = {y: int(st.session_state.results[y].notna().sum()) for y in st.session_state.y_cols}
        st.write("Remplissage Y:", y_fill)

    with c1:
        st.markdown("### Tableau d’exécution (éditable)")
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
    st.subheader("Analyse — coefficients, ANOVA, effets, diagnostics")
    st.write(
        "Choisis une réponse Y, un modèle (main / interactions / quadratique), puis lance l’analyse. "
        "L’app bloque automatiquement si le modèle est trop riche pour le nombre d’essais."
    )

    if st.session_state.results is None or st.session_state.doe_coded is None:
        st.warning("Génère un plan et saisis des résultats d’abord.")
        st.stop()

    factors = st.session_state.factors
    df_res = st.session_state.results.copy()
    df_coded = st.session_state.doe_coded.copy()

    y_candidates = [y for y in st.session_state.y_cols if y in df_res.columns]
    y_col = st.selectbox("Réponse (Y)", options=y_candidates)

    include_inter = st.checkbox("Inclure interactions (2 facteurs)", value=True)
    include_quad = st.checkbox("Inclure termes quadratiques (RSM)", value=False)
    include_block = st.checkbox("Inclure Block dans le modèle", value=True)

    use_stepwise = st.checkbox("Simplifier le modèle (backward elimination)", value=False)
    alpha_step = st.slider("Seuil p-value (si stepwise)", 0.01, 0.30, 0.10, 0.01)

    # Keep rows where Y not null
    mask = df_res[y_col].notna()
    n_obs = int(mask.sum())
    if n_obs == 0:
        st.warning("Aucune valeur Y renseignée. Va à Exécution.")
        st.stop()

    df_model = df_res.loc[mask].copy()

    # Ensure Block exists
    if "Block" not in df_model.columns:
        df_model["Block"] = 1

    formula = build_formula(y_col, factors, include_inter, include_quad, include_block)
    st.code(formula)

    # Guardrail: N vs parameters (approx)
    # (We keep it simple: if N <= number of columns in design matrix, block)
    try:
        # quick estimate using patsy design matrix via statsmodels
        model_tmp = smf.ols(formula=formula, data=df_model).fit()
        p = int(model_tmp.df_model) + 1
    except Exception:
        p = 1 + len([f for f in factors if f.kind == "quant"]) * (3 if include_inter else 1)

    if n_obs <= p:
        st.error(f"Pas assez d’essais pour ce modèle: N={n_obs} <= paramètres~{p}. Réduis le modèle ou ajoute essais.")
        st.stop()

    # Cache key
    cache_key = (
        stable_hash_df(df_model[["RunOrder", "Block"] + [f.name for f in factors] + [y_col]].copy())
        + f"|Y={y_col}|inter={include_inter}|quad={include_quad}|block={include_block}|step={use_stepwise}|a={alpha_step}"
        + f"|plan_id={st.session_state.plan_id}"
    )

    if st.button("🚀 Lancer l’analyse"):
        if st.session_state.analysis_cache_key == cache_key and st.session_state.analysis_cache is not None:
            st.info("Analyse déjà calculée (cache) ✅")
        else:
            try:
                model = smf.ols(formula=formula, data=df_model).fit()
                if use_stepwise:
                    model = backward_elimination(model, alpha=float(alpha_step))

                try:
                    anova_tbl = anova_lm(model, typ=2)
                except Exception:
                    anova_tbl = None

                # Effects on coded (only numeric coded columns)
                coded_cols = [c for c in df_coded.columns if c not in ["RunOrder", "Block"]]
                df_coded_used = df_coded.loc[mask.values, coded_cols].reset_index(drop=True)
                y_used = df_res.loc[mask, y_col].reset_index(drop=True)
                eff = compute_effects(df_coded_used, y_used, include_interactions=include_inter)

                st.session_state.analysis_cache = {"model": model, "anova": anova_tbl, "effects": eff}
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

    # Results
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

    st.markdown("## Effets + Pareto")
    st.dataframe(eff, use_container_width=True)

    figp = plt.figure()
    plt.bar(range(len(eff)), eff["|Effet|"].values)
    plt.xticks(range(len(eff)), eff["Terme"].values, rotation=90)
    plt.ylabel("|Effet|")
    plt.title("Pareto des effets")
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
# 5) OPTIMISATION (1 ou multi Y)
# ============================================================
if step == "Optimisation":
    st.subheader("Optimisation — 1 réponse ou multi-réponses (désirabilité)")
    st.write(
        "L’optimisation utilise le(s) modèle(s) ajusté(s). "
        "Tu définis un objectif (Max/Min/Cible) et des bornes désirées."
    )

    if st.session_state.results is None:
        st.warning("Génère un plan, saisis des résultats, fais une analyse d’abord.")
        st.stop()

    # On entraîne des modèles pour les Y choisies (sur le même schéma de formule)
    factors = st.session_state.factors
    df_res = st.session_state.results.copy()

    # Only keep rows with at least one Y filled for those selected
    y_candidates = [y for y in st.session_state.y_cols if y in df_res.columns]
    selected_y = st.multiselect("Réponses à optimiser", y_candidates, default=y_candidates[:1] if y_candidates else [])

    if not selected_y:
        st.info("Choisis au moins une réponse.")
        st.stop()

    include_inter = st.checkbox("Modèle: interactions", value=True)
    include_quad = st.checkbox("Modèle: quadratique (RSM)", value=False)
    include_block = st.checkbox("Inclure Block", value=True)

    # Goals for each Y
    st.markdown("### Objectifs")
    goals = {}
    for y in selected_y:
        st.markdown(f"**{y}**")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            goal = st.selectbox(f"Objectif ({y})", ["Maximiser", "Minimiser", "Cibler"], key=f"goal_{y}")
        with c2:
            low = float(st.number_input(f"Seuil bas (désirabilité=0) {y}", value=0.0, key=f"low_{y}"))
        with c3:
            high = float(st.number_input(f"Seuil haut (désirabilité=1) {y}", value=1.0, key=f"high_{y}"))
        with c4:
            weight = float(st.number_input(f"Poids {y} (>=1)", value=1.0, min_value=1.0, key=f"w_{y}"))
        target = None
        if goal == "Cibler":
            target = float(st.number_input(f"Cible {y}", value=(low+high)/2, key=f"t_{y}"))
        goals[y] = {"goal": goal, "low": low, "high": high, "target": target, "weight": weight}

    # Fit models
    models = {}
    for y in selected_y:
        mask = df_res[y].notna()
        if mask.sum() < 6:
            st.warning(f"Pas assez de données pour {y} (>=6 recommandé).")
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

    st.markdown("### Recherche du meilleur réglage (random search)")
    n_samples = st.slider("Nb d'essais virtuels (plus = mieux)", 200, 10000, 2000, 200)

    if st.button("🎯 Optimiser"):
        out = optimize_desirability(models, factors, goals, n_samples=int(n_samples), seed=42)
        if out is None:
            st.error("Aucun facteur quantitatif à optimiser (cat seulement).")
            st.stop()

        bestD, best = out
        x_best, y_pred = best

        st.success(f"Désirabilité globale = {bestD:.3f}")
        st.markdown("#### Réglages recommandés (facteurs quantitatifs)")
        st.dataframe(pd.DataFrame([x_best]), use_container_width=True)

        st.markdown("#### Prédictions")
        st.dataframe(pd.DataFrame([y_pred]), use_container_width=True)


# ============================================================
# 6) RAPPORT
# ============================================================
if step == "Rapport":
    st.subheader("Rapport (simple)")
    st.write("Génère un résumé téléchargeable (utile pour partager).")

    if st.session_state.doe_real is None:
        st.info("Génère un plan d’abord.")
        st.stop()

    summary = []
    summary.append("# Rapport DOE\n")
    summary.append("## Plan\n")
    summary.append(f"- Plan ID: {st.session_state.plan_id}\n")
    if st.session_state.design_cfg:
        summary.append(f"- Type: {st.session_state.design_cfg.design_type}\n")
        summary.append(f"- Réplicats: {st.session_state.design_cfg.replicates}\n")
        summary.append(f"- Points centraux: {st.session_state.design_cfg.center_points}\n")
        summary.append(f"- Blocs: {st.session_state.design_cfg.n_blocks}\n")

    summary.append("\n## Facteurs\n")
    for f in st.session_state.factors:
        if f.kind == "quant":
            summary.append(f"- {f.name}: [{f.low}, {f.high}] pas={f.step}\n")
        else:
            summary.append(f"- {f.name}: cat {f.levels}\n")

    summary.append("\n## Aperçu plan (10 premières lignes)\n")
    summary.append(st.session_state.doe_real.head(10).to_markdown(index=False))
    summary.append("\n")

    if st.session_state.results is not None:
        summary.append("\n## Remplissage réponses\n")
        for y in st.session_state.y_cols:
            if y in st.session_state.results.columns:
                summary.append(f"- {y}: {int(st.session_state.results[y].notna().sum())}/{len(st.session_state.results)}\n")

    md = "\n".join(summary).encode("utf-8")
    st.download_button("Télécharger rapport (MD)", md, file_name="rapport_doe.md", mime="text/markdown")
