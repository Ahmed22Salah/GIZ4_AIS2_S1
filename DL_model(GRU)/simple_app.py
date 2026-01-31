import os, re, glob, math, tempfile, unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import joblib
import requests
import streamlit as st
import tensorflow as tf
from tensorflow import keras


# =============================================================================
# Page + Styles
# =============================================================================
st.set_page_config(page_title="Matchday Predictor (Fixed)", layout="wide")

st.markdown(
    """
<style>
.stApp {
  background: radial-gradient(1200px 800px at 20% 10%, #1f8f3b 0%, #0b5d26 45%, #063d1a 100%);
  color: #eaf6ee;
}
section[data-testid="stSidebar"] {
  background: rgba(0,0,0,0.25);
  border-right: 1px solid rgba(255,255,255,0.08);
}
.tile {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px;
}
.teamname { font-size: 18px; font-weight: 800; }
.vs {
  font-size: 16px; font-weight: 900; letter-spacing: 2px;
  padding: 8px 14px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(0,0,0,0.18);
  display: inline-block;
}
.badgebox {
  width: 70px; height: 70px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(0,0,0,0.25);
  display: flex; align-items: center; justify-content: center;
  overflow: hidden;
}
.small { opacity: 0.85; font-size: 13px; }
.probrow { display:grid; grid-template-columns:120px 1fr 70px; gap:12px; align-items:center; margin:8px 0; }
.barwrap {
  background: rgba(255,255,255,0.10);
  border-radius: 999px;
  height: 14px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.10);
}
.barfill { height: 100%; border-radius: 999px; }
.pct { text-align: right; font-variant-numeric: tabular-nums; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='tile'>"
    "<h2 style='margin:0'>Matchday Predictor (Fixed UI + Scoreline Heuristic)</h2>"
    "<div class='small'>Fixes the bug where Streamlit prints <code>st.image({ ... })</code> or API docs by never printing Streamlit objects/functions.</div>"
    "</div>",
    unsafe_allow_html=True,
)


# =============================================================================
# Custom losses (for loading models that were saved with custom losses)
# =============================================================================
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=3)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * y_true_one_hot * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed

def focal_loss_with_label_smoothing(gamma=2.5, alpha=0.25, smoothing=0.1):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=3)
        if smoothing > 0:
            y_true_one_hot = y_true_one_hot * (1 - smoothing) + smoothing / 3.0
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss_fn

CUSTOM_OBJECTS = {
    "focal_loss_fixed": focal_loss(gamma=2.0, alpha=0.25),
    "loss_fn": focal_loss_with_label_smoothing(gamma=2.5, alpha=0.25, smoothing=0.1),
}


# =============================================================================
# Helpers (file, data, rendering)
# =============================================================================
def list_files(patterns: List[str]) -> List[str]:
    out = []
    for p in patterns:
        out.extend(glob.glob(p))
    return sorted(set(out), key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)

def save_uploaded_to_temp(uploaded) -> str:
    suffix = os.path.splitext(uploaded.name)[1]
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.getbuffer())
    return path

def safe_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    p = np.clip(p, 1e-12, 1.0)
    s = float(p.sum())
    return p / s if s > 0 else np.ones_like(p) / len(p)

def normalize_matches_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    need = {"home_team", "away_team", "home_score", "away_score"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int)
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int)

    return df.sort_values(["date"], na_position="last").reset_index(drop=True)

def render_prob_bars(probs: Dict[str, float]) -> None:
    colors = {"Home win": "#6ef3a3", "Draw": "#f6d365", "Away win": "#7ab8ff"}
    html = ""
    for k, v in probs.items():
        pct = float(v) * 100.0
        color = colors.get(k, "#a7ff83")
        html += f"""
        <div class="probrow">
          <div>{k}</div>
          <div class="barwrap"><div class="barfill" style="width:{pct:.1f}%; background:{color};"></div></div>
          <div class="pct">{pct:5.1f}%</div>
        </div>
        """
    st.markdown(f"<div class='tile'>{html}</div>", unsafe_allow_html=True)


# =============================================================================
# Badges (IMPORTANT: renders without ever printing Streamlit objects)
# =============================================================================
BADGES_DIR = "badges"

def ensure_badges_dir():
    os.makedirs(BADGES_DIR, exist_ok=True)

def canonical(s: str) -> str:
    s = (s or "").strip().lower().replace("&", " and ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def slugify(team: str) -> str:
    s = canonical(team)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_") or "team"

def local_badge_path(team: str) -> Optional[str]:
    ensure_badges_dir()
    base = os.path.join(BADGES_DIR, slugify(team))
    for ext in ["png", "jpg", "jpeg"]:
        p = f"{base}.{ext}"
        if os.path.exists(p):
            return p
    return None

@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=10, headers={"User-Agent": "matchday-app/1.0"})
    r.raise_for_status()
    return r.content

TEAM_ISO2 = {canonical(k): v for k, v in [
    ("Egypt","eg"), ("South Africa","za"), ("Nigeria","ng"), ("Ghana","gh"), ("Senegal","sn"),
    ("Cameroon","cm"), ("Algeria","dz"), ("Morocco","ma"), ("Tunisia","tn"),
    ("Ivory Coast","ci"), ("Cote d'Ivoire","ci"), ("Côte d'Ivoire","ci"),
    ("DR Congo","cd"), ("Democratic Republic of the Congo","cd"),
    ("Congo","cg"), ("Cape Verde","cv"), ("Cabo Verde","cv"),
    ("Saudi Arabia","sa"), ("United Arab Emirates","ae"), ("Qatar","qa"),
    ("Iraq","iq"), ("Jordan","jo"), ("Palestine","ps"), ("Syria","sy"),
    ("Lebanon","lb"), ("Kuwait","kw"), ("Bahrain","bh"), ("Oman","om"), ("Yemen","ye"),
]}

def iso2_for_team(team: str) -> Optional[str]:
    c = canonical(team)
    if c in TEAM_ISO2:
        return TEAM_ISO2[c]
    if "cote" in c and "ivoire" in c:
        return "ci"
    if c in ("dr congo", "congo dr", "congo kinshasa"):
        return "cd"
    return None

def flagcdn_url(team: str, width: int) -> Optional[str]:
    code = iso2_for_team(team)
    return f"https://flagcdn.com/w{width}/{code}.png" if code else None

def maybe_cache_flag(team: str, img: bytes):
    try:
        ensure_badges_dir()
        p = os.path.join(BADGES_DIR, f"{slugify(team)}.png")
        if not os.path.exists(p):
            with open(p, "wb") as f:
                f.write(img)
    except Exception:
        pass

def get_badge_bytes(team: str, use_online: bool, width: int, cache_to_disk: bool) -> Optional[bytes]:
    local = local_badge_path(team)
    if local:
        try:
            with open(local, "rb") as f:
                return f.read()
        except Exception:
            pass

    if not use_online:
        return None

    url = flagcdn_url(team, width)
    if not url:
        return None

    try:
        img = fetch_bytes(url)
        if cache_to_disk:
            maybe_cache_flag(team, img)
        return img
    except Exception:
        return None

def render_badge(team: str, use_online: bool, width: int, cache_to_disk: bool) -> None:
    """
    IMPORTANT: does not return anything and does not print Streamlit objects.
    """
    img = get_badge_bytes(team, use_online, width, cache_to_disk)
    if img:
        st.image(img, width=70)
    else:
        letters = re.findall(r"[A-Za-z]", team)
        abbr = ("".join(letters[:2]) if letters else team[:2]).upper()
        abbr = (abbr or "??")[:2]
        st.markdown(
            f"<div class='badgebox' style='font-weight:900; font-size:18px; color: rgba(255,255,255,0.92);'>{abbr}</div>",
            unsafe_allow_html=True
        )


# =============================================================================
# Feature snapshot (31-feature fixture builder)
# =============================================================================
BALANCED_FEATURE_COLUMNS = [
    'home_team_id', 'away_team_id',
    'home_elo', 'away_elo', 'elo_diff',
    'home_goals_avg_5', 'home_conceded_avg_5', 'home_win_rate_5',
    'away_goals_avg_5', 'away_conceded_avg_5', 'away_win_rate_5',
    'home_goals_avg_10', 'home_conceded_avg_10', 'home_win_rate_10',
    'away_goals_avg_10', 'away_conceded_avg_10', 'away_win_rate_10',
    'home_recent_form_3', 'away_recent_form_3',
    'home_advantage', 'tournament_weight',
    'is_afcon', 'is_qualifier', 'is_friendly',
    'home_matches', 'away_matches',
    'form_diff', 'attack_defense_home', 'attack_defense_away',
    'strength_diff', 'recent_momentum_diff'
]

@dataclass
class TeamState:
    gf: List[int]
    ga: List[int]
    res: List[float]
    m: int

def build_states_elo(df: pd.DataFrame, k: float = 35.0, initial: float = 1500.0):
    ratings: Dict[str, float] = {}
    states: Dict[str, TeamState] = {}

    def ensure(t):
        if t not in states:
            states[t] = TeamState([], [], [], 0)

    def rating(t):
        ratings.setdefault(t, float(initial))
        return ratings[t]

    def exp(ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    for r in df.itertuples(index=False):
        h, a = str(r.home_team).strip(), str(r.away_team).strip()
        hs, as_ = int(r.home_score), int(r.away_score)
        ensure(h); ensure(a)

        sh = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        sa = 1.0 - sh if sh != 0.5 else 0.5

        states[h].gf.append(hs); states[h].ga.append(as_); states[h].res.append(sh); states[h].m += 1
        states[a].gf.append(as_); states[a].ga.append(hs); states[a].res.append(sa); states[a].m += 1

        rh, ra = rating(h), rating(a)
        eh = exp(rh, ra)
        ratings[h] = rh + k * (sh - eh)
        ratings[a] = ra + k * (sa - (1.0 - eh))

    return states, ratings

def fixture_feature_dict(
    df_matches: pd.DataFrame,
    team_encoder,
    home: str,
    away: str,
    is_afcon: int,
    is_qualifier: int,
    is_friendly: int,
    neutral: bool
) -> Dict[str, float]:
    df = normalize_matches_df(df_matches)
    states, ratings = build_states_elo(df)

    def stats(team: str, w: int):
        s = states.get(team)
        if not s or s.m == 0:
            return 0.0, 0.0, 0.0
        gf = float(np.mean(s.gf[-w:])) if s.gf else 0.0
        ga = float(np.mean(s.ga[-w:])) if s.ga else 0.0
        wr = float(np.mean(s.res[-w:])) if s.res else 0.0
        return gf, ga, wr

    def form(team: str, w: int):
        s = states.get(team)
        return float(np.mean(s.res[-w:])) if s and s.res else 0.0

    home, away = home.strip(), away.strip()
    hid = int(team_encoder.transform([home])[0])
    aid = int(team_encoder.transform([away])[0])

    helo = float(ratings.get(home, 1500.0))
    aelo = float(ratings.get(away, 1500.0))
    elo_diff = helo - aelo

    hg5, hc5, hw5 = stats(home, 5)
    ag5, ac5, aw5 = stats(away, 5)
    hg10, hc10, hw10 = stats(home, 10)
    ag10, ac10, aw10 = stats(away, 10)

    hform3 = form(home, 3)
    aform3 = form(away, 3)

    hm = float(states.get(home, TeamState([], [], [], 0)).m)
    am = float(states.get(away, TeamState([], [], [], 0)).m)

    home_adv = 0 if neutral else 1
    tw = 1.0 + 0.5 * int(is_afcon) + 0.3 * int(is_qualifier)

    return {
        'home_team_id': hid, 'away_team_id': aid,
        'home_elo': helo, 'away_elo': aelo, 'elo_diff': elo_diff,
        'home_goals_avg_5': hg5, 'home_conceded_avg_5': hc5, 'home_win_rate_5': hw5,
        'away_goals_avg_5': ag5, 'away_conceded_avg_5': ac5, 'away_win_rate_5': aw5,
        'home_goals_avg_10': hg10, 'home_conceded_avg_10': hc10, 'home_win_rate_10': hw10,
        'away_goals_avg_10': ag10, 'away_conceded_avg_10': ac10, 'away_win_rate_10': aw10,
        'home_recent_form_3': hform3, 'away_recent_form_3': aform3,
        'home_advantage': home_adv, 'tournament_weight': tw,
        'is_afcon': int(is_afcon), 'is_qualifier': int(is_qualifier), 'is_friendly': int(is_friendly),
        'home_matches': hm, 'away_matches': am,
        'form_diff': hw5 - aw5,
        'attack_defense_home': hg5 - ac5, 'attack_defense_away': ag5 - hc5,
        'strength_diff': elo_diff / 100.0,
        'recent_momentum_diff': hform3 - aform3
    }


# =============================================================================
# Scoreline heuristic (post-processing, no retraining)
# =============================================================================
def poisson_pmf(k: int, lam: float) -> float:
    lam = max(1e-6, float(lam))
    return math.exp(-lam) * (lam ** int(k)) / math.factorial(int(k))

def infer_lambdas(feat: Dict[str, float]) -> Tuple[float, float]:
    home_attack = 0.65 * feat.get("home_goals_avg_5", 0.0) + 0.35 * feat.get("home_goals_avg_10", 0.0)
    away_def = 0.65 * feat.get("away_conceded_avg_5", 0.0) + 0.35 * feat.get("away_conceded_avg_10", 0.0)
    away_attack = 0.65 * feat.get("away_goals_avg_5", 0.0) + 0.35 * feat.get("away_goals_avg_10", 0.0)
    home_def = 0.65 * feat.get("home_conceded_avg_5", 0.0) + 0.35 * feat.get("home_conceded_avg_10", 0.0)

    lam_h = 0.5 * (home_attack + away_def)
    lam_a = 0.5 * (away_attack + home_def)

    adv = float(feat.get("home_advantage", 1.0))
    lam_h += 0.15 * adv
    lam_a -= 0.05 * adv

    lam_h = float(np.clip(lam_h, 0.15, 3.50))
    lam_a = float(np.clip(lam_a, 0.15, 3.50))
    return lam_h, lam_a

def scoreline_grid(
    feat: Dict[str, float],
    pH: float,
    pD: float,
    pA: float,
    max_goals: int = 6,
    reweight_to_hda: bool = True
) -> Tuple[Tuple[float, float], List[Tuple[int, int, float]]]:
    lam_h, lam_a = infer_lambdas(feat)
    grid_raw = []
    for h in range(max_goals + 1):
        ph = poisson_pmf(h, lam_h)
        for a in range(max_goals + 1):
            pa = poisson_pmf(a, lam_a)
            prob = ph * pa
            if reweight_to_hda:
                prob *= (pH if h > a else (pD if h == a else pA))
            grid_raw.append((h, a, float(prob)))

    total = float(sum(p for _, _, p in grid_raw))
    if total <= 0:
        return (lam_h, lam_a), []

    grid = [(h, a, p / total) for (h, a, p) in grid_raw]
    return (lam_h, lam_a), grid

def top_scorelines(grid: List[Tuple[int, int, float]], top_k: int) -> List[Tuple[str, float]]:
    grid_sorted = sorted(grid, key=lambda x: x[2], reverse=True)[:top_k]
    return [(f"{h}-{a}", float(p)) for (h, a, p) in grid_sorted]

def quick_markets(grid: List[Tuple[int, int, float]]) -> Dict[str, float]:
    if not grid:
        return {}
    p_over25 = sum(p for h, a, p in grid if (h + a) >= 3)
    p_btts = sum(p for h, a, p in grid if (h >= 1 and a >= 1))
    p_home_cs = sum(p for h, a, p in grid if a == 0)
    p_away_cs = sum(p for h, a, p in grid if h == 0)
    exp_total = sum((h + a) * p for h, a, p in grid)
    return {
        "Over 2.5": float(p_over25),
        "Under 2.5": float(1.0 - p_over25),
        "BTTS": float(p_btts),
        "Home clean sheet": float(p_home_cs),
        "Away clean sheet": float(p_away_cs),
        "Expected total goals": float(exp_total),
    }


# =============================================================================
# Model + preprocessors
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model_file(path: str):
    return keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)

@st.cache_resource(show_spinner=False)
def load_pkl(path: str):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def probe_shape(path: str) -> Tuple[int, int]:
    m = load_model_file(path)
    return int(m.input_shape[1]), int(m.input_shape[2])

def pick_best_model_smart(models: List[str]) -> Optional[str]:
    if not models:
        return None
    scored = []
    for p in models[:12]:
        try:
            _, nf = probe_shape(p)
            score = 1000 if nf == len(BALANCED_FEATURE_COLUMNS) else 0
            b = os.path.basename(p).lower()
            score += 50 if "optimized" in b else 0
            score += 30 if "enhanced" in b else 0
            score += 10 if "improved" in b else 0
            score += int(os.path.getmtime(p) / 1e6)
            scored.append((score, p))
        except Exception:
            scored.append((int(os.path.getmtime(p) / 1e6), p))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]

def pick_first_by_substring(paths: List[str], key: str) -> Optional[str]:
    key = key.lower()
    for p in paths:
        if key in os.path.basename(p).lower():
            return p
    return None


# =============================================================================
# Sidebar: Load files
# =============================================================================
if "loaded" not in st.session_state:
    st.session_state.loaded = None
if "matches_df_raw" not in st.session_state:
    st.session_state.matches_df_raw = None

with st.sidebar:
    st.markdown("### Load files")
    models = list_files(["*.keras", "*.h5"])
    pkls = list_files(["*.pkl"])
    csvs = list_files(["*.csv"])

    autodetect = st.toggle("Auto-detect best model", value=True)

    if autodetect:
        auto_model = pick_best_model_smart(models)
        st.caption(f"Auto model: {os.path.basename(auto_model) if auto_model else 'none'}")

        if st.button("Load auto", type="primary", use_container_width=True):
            if not auto_model:
                st.error("No model found in the folder.")
            else:
                try:
                    model = load_model_file(auto_model)
                    scaler_p = pick_first_by_substring(pkls, "scaler")
                    label_p = pick_first_by_substring(pkls, "label_encoder")
                    team_p = pick_first_by_substring(pkls, "team_encoder")
                    csv_p = next((c for c in csvs if os.path.basename(c) == "all_matches.csv"), (csvs[0] if csvs else None))

                    scaler = load_pkl(scaler_p) if scaler_p else None
                    label = load_pkl(label_p) if label_p else None
                    team = load_pkl(team_p) if team_p else None

                    st.session_state.loaded = (model, scaler, label, team, auto_model, scaler_p, label_p, team_p, csv_p)
                    if csv_p:
                        st.session_state.matches_df_raw = pd.read_csv(csv_p)
                    st.success("Loaded.")
                except Exception as e:
                    st.session_state.loaded = None
                    st.error(f"Load failed: {e}")

    st.divider()
    st.markdown("### Manual load")
    model_name = st.selectbox("Model", [""] + [os.path.basename(p) for p in models])
    p_models = {os.path.basename(p): p for p in models}
    model_path = p_models.get(model_name) if model_name else None

    p_pkls = {os.path.basename(p): p for p in pkls}
    scaler_name = st.selectbox("Scaler (.pkl)", [""] + list(p_pkls.keys()))
    label_name = st.selectbox("Label encoder (.pkl)", [""] + list(p_pkls.keys()))
    team_name = st.selectbox("Team encoder (.pkl)", [""] + list(p_pkls.keys()))

    p_csvs = {os.path.basename(p): p for p in csvs}
    csv_name = st.selectbox("Matches CSV", [""] + list(p_csvs.keys()))

    if st.button("Load selected", use_container_width=True):
        if not model_path:
            st.error("Select a model.")
        else:
            try:
                model = load_model_file(model_path)
                scaler = load_pkl(p_pkls[scaler_name]) if scaler_name else None
                label = load_pkl(p_pkls[label_name]) if label_name else None
                team = load_pkl(p_pkls[team_name]) if team_name else None
                csv_path = p_csvs[csv_name] if csv_name else None

                st.session_state.loaded = (
                    model, scaler, label, team,
                    model_path,
                    p_pkls.get(scaler_name) if scaler_name else None,
                    p_pkls.get(label_name) if label_name else None,
                    p_pkls.get(team_name) if team_name else None,
                    csv_path
                )

                if csv_path:
                    st.session_state.matches_df_raw = pd.read_csv(csv_path)
                st.success("Loaded.")
            except Exception as e:
                st.session_state.loaded = None
                st.error(f"Load failed: {e}")

    st.divider()
    st.markdown("### Badges")
    use_online_badges = st.toggle("Use online flags if no local badge", value=True)
    cache_flags_to_disk = st.toggle("Cache downloaded flags to ./badges", value=True)
    flag_width = st.select_slider("Flag quality", [80, 120, 160, 240], value=160)


# =============================================================================
# Guard
# =============================================================================
loaded = st.session_state.loaded
if not loaded:
    st.info("Load a model in the sidebar to begin.")
    st.stop()

model, scaler, label_encoder, team_encoder, mp, sp, lp, tp, cp = loaded
seq_len, n_feat = int(model.input_shape[1]), int(model.input_shape[2])

matches_df = None
if st.session_state.matches_df_raw is not None:
    try:
        matches_df = normalize_matches_df(st.session_state.matches_df_raw)
    except Exception as e:
        st.error(f"Matches CSV loaded but invalid: {e}")
        matches_df = None

# Header
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"<div class='tile'><b>Model</b><br/>{os.path.basename(mp)}<br/><span class='small'>shape={model.input_shape}</span></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='tile'><b>seq_len</b><br/>{seq_len}</div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='tile'><b>n_features</b><br/>{n_feat}</div>", unsafe_allow_html=True)

tabs = st.tabs(["Fixture mode", "Custom input"])


# =============================================================================
# Fixture mode
# =============================================================================
with tabs[0]:
    if team_encoder is None:
        st.error("Fixture mode requires a team_encoder.pkl.")
        st.stop()
    if matches_df is None:
        st.error("Fixture mode requires a matches CSV (e.g., all_matches.csv).")
        st.stop()

    teams = list(team_encoder.classes_)

    a, b, c = st.columns([2.2, 0.8, 2.2])
    with a:
        home = st.selectbox("Home team", teams, index=0)
    with b:
        st.markdown("<div style='text-align:center; padding-top:34px;'><span class='vs'>VS</span></div>", unsafe_allow_html=True)
    with c:
        away = st.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)

    o1, o2, o3, o4 = st.columns(4)
    with o1: neutral = st.toggle("Neutral venue", value=False)
    with o2: is_afcon = st.toggle("AFCON / Africa Cup", value=False)
    with o3: is_qual = st.toggle("Qualifier", value=False)
    with o4: is_friendly = st.toggle("Friendly", value=False)

    st.markdown("<div class='tile'>", unsafe_allow_html=True)
    st.write("**Scoreline options (heuristic)**")
    s1, s2, s3 = st.columns(3)
    with s1:
        show_scorelines = st.toggle("Show scoreline suggestions", value=True)
        show_markets = st.toggle("Show quick markets", value=True)
    with s2:
        top_k = st.select_slider("Top scorelines", options=[3, 5, 10], value=5)
    with s3:
        max_goals = st.select_slider("Max goals grid (0..N)", options=[5, 6, 7, 8], value=6)
    st.caption("Scorelines/markets are post-processing estimates; the NN predicts only H/D/A.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Alignment strategy (AutoFix)
    scaler_feature_names = None
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        try:
            if len(scaler.feature_names_in_) == n_feat:
                scaler_feature_names = list(scaler.feature_names_in_)
        except Exception:
            scaler_feature_names = None

    allow_fill_missing = st.checkbox("Allow missing features to be filled with 0 (if needed)", value=False)
    assume_prefix = st.checkbox("Fallback: assume model uses prefix of 31 balanced features", value=False)

    if n_feat == len(BALANCED_FEATURE_COLUMNS):
        strategy = "balanced_31"
        feature_list = BALANCED_FEATURE_COLUMNS
    elif scaler_feature_names is not None:
        strategy = "scaler_feature_names_in"
        feature_list = scaler_feature_names
    elif assume_prefix and n_feat <= len(BALANCED_FEATURE_COLUMNS):
        strategy = "balanced_prefix"
        feature_list = BALANCED_FEATURE_COLUMNS[:n_feat]
    else:
        strategy = "blocked"
        feature_list = None

    st.markdown("<div class='tile'>", unsafe_allow_html=True)
    st.write("**Preflight**")
    st.write(f"- strategy: `{strategy}`")
    st.write(f"- model n_features: `{n_feat}`")
    st.write(f"- fixture base features: `{len(BALANCED_FEATURE_COLUMNS)}`")
    st.write(f"- scaler.feature_names_in_: `{bool(scaler_feature_names)}`")
    st.markdown("</div>", unsafe_allow_html=True)

    # Team cards (FIXED: no printing of st.image)
    L, M, R = st.columns([3, 1, 3])
    with L:
        st.markdown("<div class='tile'>", unsafe_allow_html=True)
        lx, ly = st.columns([1, 3])
        with lx:
            render_badge(home, use_online_badges, flag_width, cache_flags_to_disk)
        with ly:
            st.markdown(f"<div class='teamname'>{home}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with M:
        st.markdown("<div class='tile' style='text-align:center;'><span class='vs'>VS</span></div>", unsafe_allow_html=True)

    with R:
        st.markdown("<div class='tile'>", unsafe_allow_html=True)
        rx, ry = st.columns([3, 1])
        with rx:
            st.markdown(f"<div class='teamname' style='text-align:right;'>{away}</div>", unsafe_allow_html=True)
        with ry:
            render_badge(away, use_online_badges, flag_width, cache_flags_to_disk)
        st.markdown("</div>", unsafe_allow_html=True)

    # Actual result lookup
    st.markdown("<div class='tile'>", unsafe_allow_html=True)
    st.write("**Actual result (from CSV, if present)**")
    cand = matches_df[(matches_df["home_team"] == home) & (matches_df["away_team"] == away)].copy()
    if cand.empty:
        st.caption("No exact home/away match found in CSV (future fixture or teams swapped).")
    else:
        if "date" in cand.columns and cand["date"].notna().any():
            cand = cand.sort_values("date")
            labels = cand["date"].dt.date.astype(str) + " | " + cand["home_team"] + " vs " + cand["away_team"]
            pick = st.selectbox("Select match instance", list(range(len(cand))), format_func=lambda i: labels.iloc[i])
            row = cand.iloc[pick]
        else:
            row = cand.iloc[-1]
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        st.metric("Final score", f"{home} {hs} – {as_} {away}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Predict button ALWAYS visible
    do_predict = st.button("Predict fixture", type="primary", use_container_width=True)

    if do_predict:
        if home == away:
            st.error("Home and away teams must be different.")
        elif strategy == "blocked":
            st.error(
                "Blocked: cannot align Fixture features to this model safely.\n\n"
                "Fix options:\n"
                "- Load a 31-feature model (optimized/enhanced)\n"
                "- Or load a scaler that has feature_names_in_\n"
                "- Or enable the prefix fallback only if your model really used the first N balanced features"
            )
        else:
            try:
                feat_dict = fixture_feature_dict(
                    matches_df, team_encoder, home, away,
                    int(is_afcon), int(is_qual), int(is_friendly), bool(neutral)
                )

                missing = [f for f in feature_list if f not in feat_dict]
                if missing and not allow_fill_missing:
                    st.error(
                        "Model expects features this fixture builder can't compute:\n"
                        + "\n".join([f"- {m}" for m in missing[:30]])
                        + ("\n- ..." if len(missing) > 30 else "")
                        + "\n\nEnable 'Allow missing features to be filled with 0' to proceed anyway."
                    )
                else:
                    vec = np.array([feat_dict.get(name, 0.0) for name in feature_list], dtype=np.float32)
                    X = np.repeat(vec.reshape(1, 1, -1), repeats=seq_len, axis=1)

                    if scaler is not None:
                        X2d = X.reshape(seq_len, -1)
                        X2d = scaler.transform(X2d)
                        X = X2d.reshape(1, seq_len, -1)

                    probs_raw = safe_probs(model.predict(X, verbose=0)[0])

                    if label_encoder is not None and hasattr(label_encoder, "classes_"):
                        cls = list(label_encoder.classes_)
                    else:
                        cls = ["H", "D", "A"]

                    def p_for(label: str) -> float:
                        return float(probs_raw[cls.index(label)]) if label in cls else 0.0

                    pH, pD, pA = p_for("H"), p_for("D"), p_for("A")
                    probs = {"Home win": pH, "Draw": pD, "Away win": pA}

                    st.markdown("<div class='tile'><b>Predicted outcome (model)</b></div>", unsafe_allow_html=True)
                    render_prob_bars(probs)

                    best = max(probs, key=probs.get)
                    st.markdown("<div class='tile'>", unsafe_allow_html=True)
                    st.metric("Most likely", f"{best}  ({probs[best]*100:.1f}%)")
                    if missing:
                        st.caption(f"Note: {len(missing)} feature(s) were filled with 0.")
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Scoreline heuristic
                    if show_scorelines or show_markets:
                        pH2, pD2, pA2 = max(pH, 1e-6), max(pD, 1e-6), max(pA, 1e-6)
                        (lam_h, lam_a), grid = scoreline_grid(
                            feat_dict, pH2, pD2, pA2, max_goals=max_goals, reweight_to_hda=True
                        )

                    if show_scorelines:
                        tops = top_scorelines(grid, top_k=top_k)
                        st.markdown("<div class='tile'>", unsafe_allow_html=True)
                        st.write("**Recommended scoreline (heuristic)**")
                        if tops:
                            top1, top1p = tops[0]
                            k1, k2, k3 = st.columns(3)
                            with k1: st.metric("Top scoreline", top1)
                            with k2: st.metric("Scoreline confidence", f"{top1p*100:.1f}%")
                            with k3: st.metric("Expected goals (λ)", f"{lam_h:.2f}–{lam_a:.2f}")
                            st.write("**Top scorelines:**")
                            for s, pr in tops:
                                st.write(f"- {s}  ({pr*100:.1f}%)")
                        else:
                            st.caption("Could not generate scorelines.")
                        st.caption("Scorelines are post-processing; the NN output is the H/D/A probabilities above.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    if show_markets:
                        mk = quick_markets(grid)
                        st.markdown("<div class='tile'>", unsafe_allow_html=True)
                        st.write("**Quick markets (heuristic)**")
                        if mk:
                            a1, a2, a3 = st.columns(3)
                            with a1:
                                st.metric("Over 2.5", f"{mk['Over 2.5']*100:.1f}%")
                                st.metric("Under 2.5", f"{mk['Under 2.5']*100:.1f}%")
                            with a2:
                                st.metric("BTTS", f"{mk['BTTS']*100:.1f}%")
                                st.metric("Home clean sheet", f"{mk['Home clean sheet']*100:.1f}%")
                            with a3:
                                st.metric("Away clean sheet", f"{mk['Away clean sheet']*100:.1f}%")
                                st.metric("Exp total goals", f"{mk['Expected total goals']:.2f}")
                            st.caption(
                                f"Computed from a truncated 0..{max_goals} score grid (approximation). "
                                "Increase max goals for more tail coverage."
                            )
                        else:
                            st.caption("Could not compute markets.")
                        st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# =============================================================================
# Custom input mode
# =============================================================================
with tabs[1]:
    st.markdown(
        f"<div class='tile'><b>Custom input</b><br/><span class='small'>Upload a CSV/NPY of shape ({seq_len}, {n_feat}).</span></div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        up_csv = st.file_uploader("Sequence CSV", type=["csv"], key="seqcsv")
    with c2:
        up_npy = st.file_uploader("Sequence NPY", type=["npy"], key="seqnpy")

    X_seq = None
    try:
        if up_npy is not None:
            p = save_uploaded_to_temp(up_npy)
            arr = np.asarray(np.load(p), dtype=np.float32)
            if arr.shape != (seq_len, n_feat):
                st.error(f"NPY shape is {arr.shape}, expected {(seq_len, n_feat)}.")
            else:
                X_seq = arr
        elif up_csv is not None:
            arr = pd.read_csv(up_csv).values.astype(np.float32)
            if arr.shape != (seq_len, n_feat):
                st.error(f"CSV shape is {arr.shape}, expected {(seq_len, n_feat)}.")
            else:
                X_seq = arr
    except Exception as e:
        st.error(f"Could not read input: {e}")

    if X_seq is not None:
        st.markdown("<div class='tile'>", unsafe_allow_html=True)
        st.write("**Preview**")
        st.dataframe(pd.DataFrame(X_seq).head(10), use_container_width=True, height=240)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Predict custom sequence", type="primary", use_container_width=True):
            try:
                X = X_seq.copy()
                if scaler is not None:
                    X = scaler.transform(X)
                X = X.reshape(1, seq_len, n_feat)
                probs_raw = safe_probs(model.predict(X, verbose=0)[0])

                if label_encoder is not None and hasattr(label_encoder, "classes_"):
                    cls = list(label_encoder.classes_)
                else:
                    cls = ["H", "D", "A"]

                def p_for(label: str) -> float:
                    return float(probs_raw[cls.index(label)]) if label in cls else 0.0

                probs = {"Home win": p_for("H"), "Draw": p_for("D"), "Away win": p_for("A")}
                st.markdown("<div class='tile'><b>Predicted outcome</b></div>", unsafe_allow_html=True)
                render_prob_bars(probs)
            except Exception as e:
                st.error(f"Prediction failed: {e}")