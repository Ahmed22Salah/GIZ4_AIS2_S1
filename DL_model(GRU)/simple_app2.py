import os
import re
import glob
import tempfile
import unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import joblib
import requests
import tensorflow as tf
from tensorflow import keras
import streamlit as st

# =============================================================================
# Page config + Matchday theme
# =============================================================================
st.set_page_config(page_title="Matchday Predictor", layout="wide")

MATCHDAY_CSS = """
<style>
.stApp {
  background: radial-gradient(1200px 800px at 20% 10%, #1f8f3b 0%, #0b5d26 45%, #063d1a 100%);
  color: #eaf6ee;
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(0,0,0,0.40), rgba(0,0,0,0.14));
  border-right: 1px solid rgba(255,255,255,0.08);
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(0,0,0,0.22);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 12px 40px rgba(0,0,0,0.25);
}
.scoreboard {
  background: linear-gradient(135deg, rgba(8,15,20,0.88), rgba(10,25,18,0.70));
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 18px 20px;
  margin-bottom: 14px;
  position: relative;
  overflow: hidden;
}
.scoreboard:before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(to right, rgba(255,255,255,0.08) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(255,255,255,0.08) 1px, transparent 1px);
  background-size: 80px 80px;
  opacity: 0.18;
  pointer-events: none;
}
.scoreboard h1 { margin: 0; font-size: 30px; letter-spacing: 0.6px; }
.scoreboard .sub { margin-top: 6px; opacity: 0.9; }

.tile {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px;
}

.badge {
  width: 58px; height: 58px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(0,0,0,0.25);
  display: flex; align-items: center; justify-content: center;
  overflow: hidden;
}
.badge img { width: 100%; height: 100%; object-fit: cover; }

.teamname { font-size: 18px; font-weight: 700; }
.vs {
  font-size: 16px; font-weight: 800; letter-spacing: 2px;
  opacity: 0.95;
  padding: 8px 14px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(0,0,0,0.18);
}

.formstrip { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.chip {
  display:inline-flex; align-items:center; justify-content:center;
  width: 28px; height: 22px;
  border-radius: 7px;
  font-weight: 800;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,0.14);
}
.W { background: rgba(110,243,163,0.20); color: #bfffe0; }
.D { background: rgba(246,211,101,0.20); color: #fff1b5; }
.L { background: rgba(255,122,122,0.18); color: #ffd1d1; }
.N { background: rgba(255,255,255,0.10); color: rgba(255,255,255,0.85); }

.probrow {
  display: grid;
  grid-template-columns: 120px 1fr 70px;
  gap: 12px;
  align-items: center;
  margin: 8px 0;
}
.barwrap {
  background: rgba(255,255,255,0.10);
  border-radius: 999px;
  height: 14px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.10);
}
.barfill { height: 100%; border-radius: 999px; width: 0%; }
.pct { text-align: right; font-variant-numeric: tabular-nums; opacity: 0.95; }

.smallnote { opacity: 0.85; font-size: 13px; }

.stButton button {
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(0,0,0,0.25);
}
.stButton button:hover {
  border: 1px solid rgba(255,255,255,0.25);
  background: rgba(0,0,0,0.35);
}
</style>
"""
st.markdown(MATCHDAY_CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="scoreboard">
  <h1>⚽ Matchday Football Predictor</h1>
  <div class="sub">Online flags (auto), form strips, head-to-head, and model probabilities.</div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Custom loss functions (for loading focal-loss-trained models)
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
# Helpers: file discovery + formatting
# =============================================================================
def list_files(patterns: List[str]) -> List[str]:
    out = []
    for p in patterns:
        out.extend(glob.glob(p))
    return sorted(set(out), key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)

def fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

def file_meta(path: str) -> str:
    try:
        s = os.path.getsize(path)
        m = os.path.getmtime(path)
        return f"{fmt_bytes(s)} | mtime={pd.to_datetime(m, unit='s')}"
    except Exception:
        return "Unknown"

def save_uploaded_to_temp(uploaded) -> str:
    suffix = os.path.splitext(uploaded.name)[1]
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.getbuffer())
    return path

def safe_softmax_probs(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    pred = np.clip(pred, 1e-12, 1.0)
    return pred / pred.sum()

def infer_family_hint(model_name: str) -> str:
    name = (model_name or "").lower()
    if "optimized" in name:
        return "optimized"
    if "enhanced" in name:
        return "enhanced"
    if "ensemble" in name:
        return "ensemble"
    if "improved" in name:
        return "improved"
    return "generic"

def pick_best_model(models: List[str]) -> Optional[str]:
    """FIXED: Prioritize optimized/enhanced models that work with 31-feature Fixture mode"""
    if not models:
        return None
    # Changed order: optimized/enhanced first (these use 31 features)
    for preferred in [
        "best_optimized_model.keras",
        "best_enhanced_model.keras",
        "african_football_ensemble.keras",
        "best_model_improved.keras",
        "african_football_improved_final.keras",
    ]:
        if preferred in models:
            return preferred
    return models[0]  # newest

def pick_best_pkl(pkls: List[str], kind: str, hint: str) -> Optional[str]:
    if not pkls:
        return None
    preferred = []
    if hint != "generic":
        preferred += [f"{kind}_{hint}.pkl", f"{kind}_{hint}_v2.pkl"]
    preferred += [
        f"{kind}_improved.pkl",
        f"{kind}_optimized.pkl",
        f"{kind}_ensemble.pkl",
        f"{kind}_enhanced.pkl",
        f"{kind}.pkl",
    ]
    for p in preferred:
        if p in pkls:
            return p
    for p in pkls:
        if kind in os.path.basename(p).lower():
            return p
    return pkls[0]

def pick_best_csv(csvs: List[str]) -> Optional[str]:
    if not csvs:
        return None
    if "all_matches.csv" in csvs:
        return "all_matches.csv"
    return csvs[0]

@st.cache_resource(show_spinner=False)
def load_model_and_preprocessors(model_file: str,
                                 scaler_file: Optional[str],
                                 label_file: Optional[str],
                                 team_file: Optional[str]):
    model = keras.models.load_model(model_file, custom_objects=CUSTOM_OBJECTS, compile=False)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    scaler = joblib.load(scaler_file) if scaler_file else None
    label_encoder = joblib.load(label_file) if label_file else None
    team_encoder = joblib.load(team_file) if team_file else None
    return model, scaler, label_encoder, team_encoder

# =============================================================================
# Badges: local + online fallback (FlagCDN) + strong normalization
# =============================================================================
BADGES_DIR = "badges"

def ensure_badges_dir():
    os.makedirs(BADGES_DIR, exist_ok=True)

def slugify_team(team: str) -> str:
    s = team.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "team"

def badge_candidates(team: str) -> List[str]:
    ensure_badges_dir()
    slug = slugify_team(team)
    patterns = [
        os.path.join(BADGES_DIR, f"{slug}.png"),
        os.path.join(BADGES_DIR, f"{slug}.jpg"),
        os.path.join(BADGES_DIR, f"{slug}.jpeg"),
        os.path.join(BADGES_DIR, f"{team}.png"),
        os.path.join(BADGES_DIR, f"{team}.jpg"),
        os.path.join(BADGES_DIR, f"{team}.jpeg"),
    ]
    return patterns

def get_badge_path(team: str) -> Optional[str]:
    for p in badge_candidates(team):
        if os.path.exists(p):
            return p
    return None

def save_badge(team: str, uploaded, force_png_name: bool = True) -> Optional[str]:
    if uploaded is None:
        return None
    ensure_badges_dir()
    ext = os.path.splitext(uploaded.name)[1].lower()
    ext = ext if ext in [".png", ".jpg", ".jpeg"] else ".png"
    fname = f"{slugify_team(team)}.png" if force_png_name else f"{slugify_team(team)}{ext}"
    path = os.path.join(BADGES_DIR, fname)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path

def canonical_team_name(s: str) -> str:
    """
    Aggressive normalizer:
    - lower
    - remove accents
    - replace "&" with "and"
    - drop punctuation
    - collapse spaces
    """
    s = (s or "").strip().lower()
    s = s.replace("&", " and ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Minimal mapping + aliases. Canonicalization handles many variants automatically.
TEAM_ISO2_PAIRS = [
    # Africa
    ("Egypt", "eg"), ("Nigeria", "ng"), ("Cameroon", "cm"), ("Senegal", "sn"), ("Ghana", "gh"),
    ("Algeria", "dz"), ("Morocco", "ma"), ("Tunisia", "tn"), ("Ivory Coast", "ci"),
    ("Cote dIvoire", "ci"), ("Côte d'Ivoire", "ci"),
    ("Mali", "ml"), ("Burkina Faso", "bf"), ("South Africa", "za"), ("DR Congo", "cd"),
    ("Democratic Republic of the Congo", "cd"), ("Congo DR", "cd"),
    ("Guinea", "gn"), ("Zambia", "zm"), ("Uganda", "ug"), ("Kenya", "ke"), ("Ethiopia", "et"),
    ("Tanzania", "tz"), ("Zimbabwe", "zw"), ("Angola", "ao"), ("Benin", "bj"), ("Gabon", "ga"),
    ("Equatorial Guinea", "gq"), ("Mozambique", "mz"), ("Cape Verde", "cv"),
    ("Cape Verde Islands", "cv"), ("Cabo Verde", "cv"),
    ("Mauritania", "mr"), ("Comoros", "km"), ("Madagascar", "mg"), ("Central African Republic", "cf"),
    ("Congo", "cg"), ("Republic of the Congo", "cg"),
    ("Botswana", "bw"), ("Namibia", "na"), ("Libya", "ly"), ("Sudan", "sd"), ("Rwanda", "rw"),
    ("Togo", "tg"), ("Niger", "ne"), ("Sierra Leone", "sl"), ("Malawi", "mw"), ("Chad", "td"),
    ("Burundi", "bi"), ("Liberia", "lr"), ("Lesotho", "ls"), ("Mauritius", "mu"), ("Seychelles", "sc"),
    ("Djibouti", "dj"), ("Eritrea", "er"), ("Somalia", "so"), ("South Sudan", "ss"), ("Eswatini", "sz"),
    ("Swaziland", "sz"), ("Gambia", "gm"), ("Guinea-Bissau", "gw"),
    ("Sao Tome and Principe", "st"), ("São Tomé and Príncipe", "st"),

    # Arab (non-Africa included in your list)
    ("Saudi Arabia", "sa"), ("United Arab Emirates", "ae"), ("Qatar", "qa"), ("Iraq", "iq"),
    ("Jordan", "jo"), ("Palestine", "ps"), ("Syria", "sy"), ("Lebanon", "lb"), ("Kuwait", "kw"),
    ("Bahrain", "bh"), ("Oman", "om"), ("Yemen", "ye"),
]

TEAM_TO_ISO2_CANON: Dict[str, str] = {canonical_team_name(k): v for k, v in TEAM_ISO2_PAIRS}

def iso2_for_team(team: str) -> Optional[str]:
    c = canonical_team_name(team)

    # direct match
    if c in TEAM_TO_ISO2_CANON:
        return TEAM_TO_ISO2_CANON[c]

    # a few extra heuristic fallbacks
    if "sao tome" in c:
        return "st"
    if "cote" in c and "ivoire" in c:
        return "ci"
    if c in ("d r congo", "dr congo", "congo dr", "congo kinshasa"):
        return "cd"
    if c in ("congo", "republic of congo", "congo brazzaville"):
        return "cg"

    return None

def flagcdn_url(team: str, width: int = 160) -> Optional[str]:
    code = iso2_for_team(team)
    if not code:
        return None
    return f"https://flagcdn.com/w{width}/{code.lower()}.png"

@st.cache_data(show_spinner=False)
def fetch_image_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=10, headers={"User-Agent": "matchday-app/1.0"})
    r.raise_for_status()
    return r.content

def maybe_cache_to_disk(team: str, img_bytes: bytes) -> Optional[str]:
    """
    Save online badge to ./badges/ so it persists between runs.
    Returns path.
    """
    try:
        ensure_badges_dir()
        path = os.path.join(BADGES_DIR, f"{slugify_team(team)}.png")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(img_bytes)
        return path
    except Exception:
        return None

def get_badge_bytes(team: str,
                    use_online: bool = True,
                    width: int = 160,
                    cache_downloads_to_disk: bool = True) -> Optional[bytes]:
    # 1) local badge if exists
    local = get_badge_path(team)
    if local and os.path.exists(local):
        try:
            with open(local, "rb") as f:
                return f.read()
        except Exception:
            pass

    # 2) online flag fallback
    if not use_online:
        return None

    url = flagcdn_url(team, width=width)
    if not url:
        return None

    try:
        img = fetch_image_bytes(url)
        if cache_downloads_to_disk:
            maybe_cache_to_disk(team, img)
        return img
    except Exception:
        return None

def badge_placeholder_html(team: str) -> str:
    letters = re.findall(r"[A-Za-z]", team)
    abbr = ("".join(letters[:2]) if letters else team[:2]).upper()
    abbr = (abbr or "??")[:2]
    return f"<div class='badge' style='font-weight:900; font-size:18px; color: rgba(255,255,255,0.92);'>{abbr}</div>"

# =============================================================================
# Match analytics (form strip, head-to-head)
# =============================================================================
def normalize_matches_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    required = {"home_team", "away_team", "home_score", "away_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int)
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int)

    df = df.sort_values(["date"], na_position="last").reset_index(drop=True)
    return df

def team_form_strip(df: pd.DataFrame, team: str, n: int = 5) -> Tuple[List[str], Dict[str, float], pd.DataFrame]:
    t = team.strip()
    home_mask = df["home_team"] == t
    away_mask = df["away_team"] == t
    recent = df[home_mask | away_mask].copy()
    if recent.empty:
        return ["N"] * n, {"ppg": 0.0, "gf": 0.0, "ga": 0.0}, pd.DataFrame()

    recent = recent.tail(n).copy()

    strip = []
    gf_total = 0
    ga_total = 0
    pts = 0
    rows = []

    for r in recent.itertuples(index=False):
        home = r.home_team
        away = r.away_team
        hs = int(r.home_score)
        a_s = int(r.away_score)
        d = getattr(r, "date", pd.NaT)

        if home == t:
            gf, ga = hs, a_s
            if hs > a_s:
                res, p = "W", 3
            elif hs == a_s:
                res, p = "D", 1
            else:
                res, p = "L", 0
            opp = away
            venue = "H"
        else:
            gf, ga = a_s, hs
            if a_s > hs:
                res, p = "W", 3
            elif a_s == hs:
                res, p = "D", 1
            else:
                res, p = "L", 0
            opp = home
            venue = "A"

        gf_total += gf
        ga_total += ga
        pts += p
        strip.append(res)
        rows.append({
            "date": d.date().isoformat() if pd.notna(d) else "",
            "venue": venue,
            "opponent": opp,
            "score": f"{gf}-{ga}",
            "result": res
        })

    if len(strip) < n:
        strip = (["N"] * (n - len(strip))) + strip

    summary = {
        "ppg": float(pts) / max(1, len(recent)),
        "gf": float(gf_total) / max(1, len(recent)),
        "ga": float(ga_total) / max(1, len(recent)),
    }
    return strip, summary, pd.DataFrame(rows)

def head_to_head(df: pd.DataFrame, home_team: str, away_team: str, n: int = 5) -> pd.DataFrame:
    h = home_team.strip()
    a = away_team.strip()
    m1 = (df["home_team"] == h) & (df["away_team"] == a)
    m2 = (df["home_team"] == a) & (df["away_team"] == h)
    h2h = df[m1 | m2].copy()
    if h2h.empty:
        return pd.DataFrame()
    h2h = h2h.tail(n).copy()

    rows = []
    for r in h2h.itertuples(index=False):
        d = getattr(r, "date", pd.NaT)
        hs = int(r.home_score)
        a_s = int(r.away_score)
        rows.append({
            "date": d.date().isoformat() if pd.notna(d) else "",
            "home": r.home_team,
            "away": r.away_team,
            "score": f"{hs}-{a_s}"
        })
    return pd.DataFrame(rows)

def render_form_strip(strip: List[str]):
    chips = ""
    for c in strip:
        cls = c if c in ["W", "D", "L"] else "N"
        chips += f"<span class='chip {cls}'>{c}</span>"
    st.markdown(f"<div class='formstrip'>{chips}</div>", unsafe_allow_html=True)

def render_prob_bars(probs: Dict[str, float], colors: Dict[str, str]):
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
# Fixture snapshot feature builder (balanced 31 features)
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
    goals_for: List[int]
    goals_against: List[int]
    results: List[float]
    matches: int

def build_team_states_and_elo(df: pd.DataFrame, k_factor: float = 35.0, initial: float = 1500.0
                             ) -> Tuple[Dict[str, TeamState], Dict[str, float]]:
    ratings: Dict[str, float] = {}
    states: Dict[str, TeamState] = {}

    def get_rating(team: str) -> float:
        if team not in ratings:
            ratings[team] = float(initial)
        return ratings[team]

    def expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def ensure(team: str):
        if team not in states:
            states[team] = TeamState([], [], [], 0)

    for row in df.itertuples(index=False):
        home = str(getattr(row, "home_team")).strip()
        away = str(getattr(row, "away_team")).strip()
        hs = int(getattr(row, "home_score"))
        a_s = int(getattr(row, "away_score"))

        ensure(home); ensure(away)

        if hs > a_s:
            score_home = 1.0
        elif hs < a_s:
            score_home = 0.0
        else:
            score_home = 0.5

        score_away = 1.0 - score_home if score_home != 0.5 else 0.5

        states[home].goals_for.append(hs)
        states[home].goals_against.append(a_s)
        states[home].results.append(score_home)
        states[home].matches += 1

        states[away].goals_for.append(a_s)
        states[away].goals_against.append(hs)
        states[away].results.append(score_away)
        states[away].matches += 1

        rh = get_rating(home); ra = get_rating(away)
        eh = expected(rh, ra); ea = 1.0 - eh
        ratings[home] = rh + k_factor * (score_home - eh)
        ratings[away] = ra + k_factor * (score_away - ea)

    return states, ratings

def snapshot_features_for_fixture(
    df_matches: pd.DataFrame,
    team_encoder,
    home_team: str,
    away_team: str,
    is_afcon: int,
    is_qualifier: int,
    is_friendly: int,
    neutral: bool,
) -> np.ndarray:
    df = normalize_matches_df(df_matches)
    states, ratings = build_team_states_and_elo(df)

    def stats(team: str, w: int) -> Tuple[float, float, float]:
        s = states.get(team)
        if not s or s.matches == 0:
            return 0.0, 0.0, 0.0
        gf = float(np.mean(s.goals_for[-w:])) if s.goals_for else 0.0
        ga = float(np.mean(s.goals_against[-w:])) if s.goals_against else 0.0
        wr = float(np.mean(s.results[-w:])) if s.results else 0.0
        return gf, ga, wr

    def form(team: str, w: int) -> float:
        s = states.get(team)
        return float(np.mean(s.results[-w:])) if s and s.results else 0.0

    home_team = str(home_team).strip()
    away_team = str(away_team).strip()

    home_id = int(team_encoder.transform([home_team])[0])
    away_id = int(team_encoder.transform([away_team])[0])

    home_elo = float(ratings.get(home_team, 1500.0))
    away_elo = float(ratings.get(away_team, 1500.0))
    elo_diff = home_elo - away_elo

    hg5, hc5, hw5 = stats(home_team, 5)
    ag5, ac5, aw5 = stats(away_team, 5)
    hg10, hc10, hw10 = stats(home_team, 10)
    ag10, ac10, aw10 = stats(away_team, 10)

    hform3 = form(home_team, 3)
    aform3 = form(away_team, 3)

    home_matches = float(states.get(home_team, TeamState([], [], [], 0)).matches)
    away_matches = float(states.get(away_team, TeamState([], [], [], 0)).matches)

    home_adv = 0 if neutral else 1
    tournament_weight = 1.0 + 0.5 * is_afcon + 0.3 * is_qualifier

    vec = {
        'home_team_id': home_id,
        'away_team_id': away_id,
        'home_elo': home_elo,
        'away_elo': away_elo,
        'elo_diff': elo_diff,
        'home_goals_avg_5': hg5,
        'home_conceded_avg_5': hc5,
        'home_win_rate_5': hw5,
        'away_goals_avg_5': ag5,
        'away_conceded_avg_5': ac5,
        'away_win_rate_5': aw5,
        'home_goals_avg_10': hg10,
        'home_conceded_avg_10': hc10,
        'home_win_rate_10': hw10,
        'away_goals_avg_10': ag10,
        'away_conceded_avg_10': ac10,
        'away_win_rate_10': aw10,
        'home_recent_form_3': hform3,
        'away_recent_form_3': aform3,
        'home_advantage': home_adv,
        'tournament_weight': tournament_weight,
        'is_afcon': int(is_afcon),
        'is_qualifier': int(is_qualifier),
        'is_friendly': int(is_friendly),
        'home_matches': home_matches,
        'away_matches': away_matches,
        'form_diff': hw5 - aw5,
        'attack_defense_home': hg5 - ac5,
        'attack_defense_away': ag5 - hc5,
        'strength_diff': elo_diff / 100.0,
        'recent_momentum_diff': hform3 - aform3
    }

    return np.array([vec[c] for c in BALANCED_FEATURE_COLUMNS], dtype=np.float32)

# =============================================================================
# Session state
# =============================================================================
if "artifacts" not in st.session_state:
    st.session_state.artifacts = None
if "matches_df" not in st.session_state:
    st.session_state.matches_df = None
if "autodetect" not in st.session_state:
    st.session_state.autodetect = True

# =============================================================================
# Sidebar: file loading + badge settings
# =============================================================================
with st.sidebar:
    st.markdown("### Files")
    st.session_state.autodetect = st.toggle("Auto-detect best files in this folder", value=st.session_state.autodetect)

    local_models = list_files(["*.keras", "*.h5"])
    local_pkls = list_files(["*.pkl"])
    local_csvs = list_files(["*.csv"])

    auto = {"model": None, "scaler": None, "label": None, "team": None, "csv": None}
    if st.session_state.autodetect:
        m = pick_best_model(local_models)
        hint = infer_family_hint(m or "")
        auto["model"] = m
        auto["scaler"] = pick_best_pkl(local_pkls, "scaler", hint)
        auto["label"] = pick_best_pkl(local_pkls, "label_encoder", hint)
        auto["team"] = pick_best_pkl(local_pkls, "team_encoder", hint)
        auto["csv"] = pick_best_csv(local_csvs)

        st.markdown("#### 🎯 Auto-detected")
        for k, v in auto.items():
            status = "✅" if v else "❌"
            st.caption(f"{status} {k}: {v if v else 'not found'}")

        if st.button("⚡ Load auto-detected", type="primary", use_container_width=True):
            if not auto["model"]:
                st.error("No model file found in this folder.")
            else:
                try:
                    with st.spinner("Loading model and preprocessors..."):
                        model, scaler, label_encoder, team_encoder = load_model_and_preprocessors(
                            auto["model"], auto["scaler"], auto["label"], auto["team"]
                        )
                        st.session_state.artifacts = (
                            model, scaler, label_encoder, team_encoder,
                            auto["model"], auto["scaler"], auto["label"], auto["team"], auto["csv"]
                        )
                        if auto["csv"]:
                            st.session_state.matches_df = pd.read_csv(auto["csv"])
                    st.success("✅ Loaded successfully!")
                except Exception as e:
                    st.session_state.artifacts = None
                    st.error(f"❌ Load failed: {e}")

    st.divider()
    st.markdown("### 🔧 Manual load (this folder)")
    model_path = st.selectbox("Model", options=[""] + sorted(local_models))
    scaler_path = st.selectbox("Scaler (.pkl)", options=[""] + sorted(local_pkls))
    label_path = st.selectbox("Label encoder (.pkl)", options=[""] + sorted(local_pkls))
    team_path = st.selectbox("Team encoder (.pkl)", options=[""] + sorted(local_pkls))
    csv_path = st.selectbox("Matches CSV (optional)", options=[""] + sorted(local_csvs))

    if st.button("📥 Load selected", use_container_width=True):
        if not model_path:
            st.error("Select a model file.")
        else:
            try:
                with st.spinner("Loading model and preprocessors..."):
                    model, scaler, label_encoder, team_encoder = load_model_and_preprocessors(
                        model_path,
                        scaler_path or None,
                        label_path or None,
                        team_path or None,
                    )
                    st.session_state.artifacts = (
                        model, scaler, label_encoder, team_encoder,
                        model_path, scaler_path or None, label_path or None, team_path or None,
                        csv_path or None
                    )
                    if csv_path:
                        st.session_state.matches_df = pd.read_csv(csv_path)
                st.success("✅ Loaded successfully!")
            except Exception as e:
                st.session_state.artifacts = None
                st.error(f"❌ Load failed: {e}")

    st.divider()
    st.markdown("### 🏴 Badges (online fallback)")
    use_online_badges = st.toggle("Use online flags when no local badge", value=True)
    cache_flags_to_disk = st.toggle("Cache downloaded flags to ./badges", value=True)
    flag_width = st.select_slider("Online flag quality", options=[80, 120, 160, 240], value=160)
    st.caption("Online flags use FlagCDN. Local uploads still work and override online.")
    st.caption(f"📂 Badge folder: {os.path.abspath(BADGES_DIR)}")

# =============================================================================
# Main content guard
# =============================================================================
art = st.session_state.artifacts
if not art:
    st.markdown(
        "<div class='tile'><b>Kickoff:</b> Load a model + preprocessors in the sidebar. "
        "Then open Fixture mode for the matchday view.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

model, scaler, label_encoder, team_encoder, mp, sp, lp, tp, cp = art
seq_len = int(model.input_shape[1])
n_feat = int(model.input_shape[2])

matches_df = None
if st.session_state.matches_df is not None:
    try:
        matches_df = normalize_matches_df(st.session_state.matches_df)
    except Exception as e:
        st.warning(f"Matches CSV loaded but not usable: {e}")
        matches_df = None

# Model info tiles
a, b, c, d = st.columns(4)
with a:
    st.markdown("<div class='tile'>", unsafe_allow_html=True)
    st.write("**Model**")
    st.caption(os.path.basename(mp))
    st.caption(file_meta(mp))
    st.markdown("</div>", unsafe_allow_html=True)
with b:
    st.markdown("<div class='tile'>", unsafe_allow_html=True)
    st.write("**Input**")
    st.code(str(model.input_shape), language="text")
    st.markdown("</div>", unsafe_allow_html=True)
with c:
    st.markdown("<div class='tile'>", unsafe_allow_html=True)
    st.write("**Sequence length**")
    st.metric("seq_length", seq_len)
    st.markdown("</div>", unsafe_allow_html=True)
with d:
    st.markdown("<div class='tile'>", unsafe_allow_html=True)
    st.write("**Features**")
    st.metric("n_features", n_feat)
    # Add feature compatibility check
    if n_feat == 31:
        st.success("✅ Compatible with Fixture mode")
    else:
        st.warning(f"⚠️ Use Custom input mode")
    st.markdown("</div>", unsafe_allow_html=True)

tabs = st.tabs(["⚽ Fixture mode (Matchday)", "🔧 Custom input mode", "🏆 Locker Room"])

# =============================================================================
# Tab 1: Fixture mode with online badges + form + H2H
# =============================================================================
with tabs[0]:
    st.markdown(
        f"""
<div class="tile">
<b>⚽ Matchday Fixture mode</b> shows online flags, recent form, head-to-head, and probabilities.
<br/><span class="smallnote">
This mode builds a balanced-feature snapshot (31 features) and repeats it across timesteps.
Designed for models trained with the balanced/optimized feature set.
</span>
</div>
""",
        unsafe_allow_html=True,
    )

    if team_encoder is None:
        st.warning("Team encoder is not loaded. Fixture mode requires team_encoder.pkl.")
        st.stop()

    teams = list(team_encoder.classes_)
    colL, colM, colR = st.columns([2.2, 0.8, 2.2])
    with colL:
        home_team = st.selectbox("🏠 Home team", teams, index=0)
    with colM:
        st.markdown("<div style='text-align:center; padding-top: 34px; font-weight: 800;'>VS</div>", unsafe_allow_html=True)
    with colR:
        away_team = st.selectbox("✈️ Away team", teams, index=1 if len(teams) > 1 else 0)

    # Optional: upload badges anyway (still supported)
    u1, u2 = st.columns(2)
    with u1:
        up_home_badge = st.file_uploader(f"Upload local badge for {home_team}", type=["png", "jpg", "jpeg"], key="homebadge")
        if up_home_badge is not None:
            try:
                save_badge(home_team, up_home_badge, force_png_name=True)
                st.success("✅ Saved to ./badges/")
            except Exception as e:
                st.error(f"Could not save: {e}")
    with u2:
        up_away_badge = st.file_uploader(f"Upload local badge for {away_team}", type=["png", "jpg", "jpeg"], key="awaybadge")
        if up_away_badge is not None:
            try:
                save_badge(away_team, up_away_badge, force_png_name=True)
                st.success("✅ Saved to ./badges/")
            except Exception as e:
                st.error(f"Could not save: {e}")

    o1, o2, o3, o4 = st.columns(4)
    with o1: neutral = st.toggle("Neutral venue", value=False)
    with o2: is_afcon = st.toggle("AFCON / Africa Cup", value=False)
    with o3: is_qualifier = st.toggle("Qualifier", value=False)
    with o4: is_friendly = st.toggle("Friendly", value=False)

    formN = st.slider("Form window (last N matches)", min_value=3, max_value=10, value=5, step=1)
    h2hN = st.slider("Head-to-head window (last N meetings)", min_value=3, max_value=10, value=5, step=1)

    # Scoreboard header (badges + names + form)
    left, mid, right = st.columns([3, 1, 3])

    with left:
        st.markdown("<div class='tile'>", unsafe_allow_html=True)
        ll1, ll2 = st.columns([1, 3])
        with ll1:
            badge = get_badge_bytes(home_team, use_online=use_online_badges, width=flag_width, cache_downloads_to_disk=cache_flags_to_disk)
            if badge:
                st.image(badge, width=70)
            else:
                st.markdown(badge_placeholder_html(home_team), unsafe_allow_html=True)
        with ll2:
            st.markdown(f"<div class='teamname'>{home_team}</div>", unsafe_allow_html=True)
            if matches_df is not None:
                strip, summ, _tbl = team_form_strip(matches_df, home_team, n=formN)
                render_form_strip(strip)
                st.caption(f"PPG: {summ['ppg']:.2f} | GF: {summ['gf']:.2f} | GA: {summ['ga']:.2f}")
            else:
                st.caption("Load all_matches.csv to show form and head-to-head.")
        st.markdown("</div>", unsafe_allow_html=True)

    with mid:
        st.markdown("<div class='tile' style='text-align:center;'>", unsafe_allow_html=True)
        st.markdown("<div class='vs'>VS</div>", unsafe_allow_html=True)
        st.caption("Matchday view")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='tile'>", unsafe_allow_html=True)
        rr1, rr2 = st.columns([3, 1])
        with rr1:
            st.markdown(f"<div class='teamname' style='text-align:right;'>{away_team}</div>", unsafe_allow_html=True)
            if matches_df is not None:
                strip, summ, _tbl = team_form_strip(matches_df, away_team, n=formN)
                render_form_strip(strip)
                st.caption(f"PPG: {summ['ppg']:.2f} | GF: {summ['gf']:.2f} | GA: {summ['ga']:.2f}")
            else:
                st.caption("Load all_matches.csv to show form and head-to-head.")
        with rr2:
            badge = get_badge_bytes(away_team, use_online=use_online_badges, width=flag_width, cache_downloads_to_disk=cache_flags_to_disk)
            if badge:
                st.image(badge, width=70)
            else:
                st.markdown(badge_placeholder_html(away_team), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if n_feat != len(BALANCED_FEATURE_COLUMNS):
        st.error(
            f"""
            ⚠️ **Feature Mismatch Detected**
            
            Your model expects **{n_feat} features**, but Fixture mode builds **{len(BALANCED_FEATURE_COLUMNS)} features**.
            
            **Solutions:**
            1. ✅ **Use Custom input mode** (second tab) with this model
            2. 🔄 **Load a compatible model** in the sidebar:
               - Try: `best_optimized_model.keras` or `best_enhanced_model.keras`
               - These models expect 31 features and work with Fixture mode
            3. 📝 **Current model**: `{os.path.basename(mp)}` is trained with a different feature set
            
            The prediction is blocked to prevent errors.
            """
        )
        st.stop()

    if matches_df is None:
        st.warning("⚠️ Fixture mode needs all_matches.csv to compute form and snapshot features. Load it in the sidebar.")
        st.stop()

    do_predict = st.button("🎯 Predict fixture", type="primary", use_container_width=True)

    if do_predict:
        if home_team == away_team:
            st.error("❌ Home and Away teams must be different.")
        else:
            try:
                with st.spinner("🔮 Making prediction..."):
                    vec = snapshot_features_for_fixture(
                        matches_df,
                        team_encoder,
                        home_team, away_team,
                        int(is_afcon), int(is_qualifier), int(is_friendly),
                        bool(neutral)
                    )

                    X = np.repeat(vec.reshape(1, 1, -1), repeats=seq_len, axis=1)

                    if scaler is not None:
                        X2d = X.reshape(seq_len, -1)
                        X2d = scaler.transform(X2d)
                        X = X2d.reshape(1, seq_len, -1)

                    probs_raw = safe_softmax_probs(model.predict(X, verbose=0)[0])

                if label_encoder is not None and hasattr(label_encoder, "classes_"):
                    classes = list(label_encoder.classes_)
                else:
                    classes = [f"class_{i}" for i in range(len(probs_raw))]

                def pretty(c):
                    if c == "H": return "Home win"
                    if c == "D": return "Draw"
                    if c == "A": return "Away win"
                    return str(c)

                probs = {pretty(classes[i]): float(probs_raw[i]) for i in range(len(probs_raw))}
                colors = {"Home win": "#6ef3a3", "Draw": "#f6d365", "Away win": "#7ab8ff"}

                st.markdown(
                    "<div class='scoreboard'><h1>🎯 Outcome probabilities</h1>"
                    "<div class='sub'>Model output for this fixture (snapshot repeated across timesteps).</div></div>",
                    unsafe_allow_html=True,
                )
                render_prob_bars(probs, colors)

                best = max(probs, key=probs.get)
                conf = probs[best] * 100.0

                k1, k2, k3 = st.columns(3)
                with k1:
                    st.markdown("<div class='tile'>", unsafe_allow_html=True)
                    st.write("**🏆 Most likely**")
                    st.metric(best, f"{conf:.1f}%")
                    
                    # Betting recommendation
                    if conf > 70:
                        st.success("💰 STRONG BET - High confidence")
                    elif conf > 60:
                        st.info("⚖️ MODERATE BET - Reasonable confidence")
                    elif conf > 50:
                        st.warning("⚠️ WEAK BET - Low confidence")
                    else:
                        st.error("❌ AVOID - Very uncertain")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with k2:
                    st.markdown("<div class='tile'>", unsafe_allow_html=True)
                    st.write("**📋 Context**")
                    st.caption(f"Neutral: {'✅ yes' if neutral else '❌ no'}")
                    st.caption(f"AFCON: {'✅ yes' if is_afcon else '❌ no'}")
                    st.caption(f"Qualifier: {'✅ yes' if is_qualifier else '❌ no'}")
                    st.caption(f"Friendly: {'✅ yes' if is_friendly else '❌ no'}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with k3:
                    st.markdown("<div class='tile'>", unsafe_allow_html=True)
                    st.write("**🔧 Input shape used**")
                    st.code(f"(1, {seq_len}, {n_feat})", language="text")
                    st.caption("Snapshot is repeated across time steps.")
                    st.caption(f"Model: {os.path.basename(mp)}")
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                import traceback
                with st.expander("🔍 Debug info"):
                    st.code(traceback.format_exc())

    st.divider()
    leftpane, rightpane = st.columns(2)

    with leftpane:
        st.markdown("<div class='tile'>", unsafe_allow_html=True)
        st.write("**Head-to-head (last N meetings)**")
        h2h = head_to_head(matches_df, home_team, away_team, n=h2hN)
        if h2h.empty:
            st.caption("No head-to-head matches found in the loaded CSV.")
        else:
            st.dataframe(h2h.iloc[::-1], use_container_width=True, height=260)
        st.markdown("</div>", unsafe_allow_html=True)

    with rightpane:
        st.markdown("<div class='tile'>", unsafe_allow_html=True)
        st.write("**Recent matches (last N per team)**")
        t1, t2 = st.columns(2)
        with t1:
            st.caption(home_team)
            _strip, _summ, tbl = team_form_strip(matches_df, home_team, n=formN)
            if tbl.empty:
                st.caption("No matches found.")
            else:
                st.dataframe(tbl.iloc[::-1], use_container_width=True, height=220)
        with t2:
            st.caption(away_team)
            _strip, _summ, tbl = team_form_strip(matches_df, away_team, n=formN)
            if tbl.empty:
                st.caption("No matches found.")
            else:
                st.dataframe(tbl.iloc[::-1], use_container_width=True, height=220)
        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# Tab 2: Custom input mode
# =============================================================================
with tabs[1]:
    st.markdown(
        f"""
<div class="tile">
<b>Custom input mode</b> works for any model: provide the exact sequence your model expects.
<br/><span class="smallnote">Expected shape: ({seq_len}, {n_feat}). CSV rows=timesteps, columns=features. NPY must match shape exactly.</span>
</div>
""",
        unsafe_allow_html=True,
    )

    ca, cb = st.columns(2)
    with ca:
        up_csv = st.file_uploader("Upload sequence CSV", type=["csv"], key="seqcsv")
    with cb:
        up_npy = st.file_uploader("Upload sequence NPY", type=["npy"], key="seqnpy")

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
            df_seq = pd.read_csv(up_csv)
            arr = df_seq.values.astype(np.float32)
            if arr.shape != (seq_len, n_feat):
                st.error(f"CSV shape is {arr.shape}, expected {(seq_len, n_feat)}.")
            else:
                X_seq = arr
    except Exception as e:
        st.error(f"Could not read input: {e}")

    if X_seq is not None:
        st.markdown("<div class='tile'>", unsafe_allow_html=True)
        st.write("**Input preview**")
        st.dataframe(pd.DataFrame(X_seq).head(10), use_container_width=True, height=260)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Predict from custom sequence", type="primary", use_container_width=True):
            try:
                X = X_seq.copy()
                if scaler is not None:
                    X = scaler.transform(X)

                X = X.reshape(1, seq_len, n_feat)
                probs_raw = safe_softmax_probs(model.predict(X, verbose=0)[0])

                if label_encoder is not None and hasattr(label_encoder, "classes_"):
                    classes = list(label_encoder.classes_)
                else:
                    classes = [f"class_{i}" for i in range(len(probs_raw))]

                def pretty(c):
                    if c == "H": return "Home win"
                    if c == "D": return "Draw"
                    if c == "A": return "Away win"
                    return str(c)

                probs = {pretty(classes[i]): float(probs_raw[i]) for i in range(len(probs_raw))}
                colors = {"Home win": "#6ef3a3", "Draw": "#f6d365", "Away win": "#7ab8ff"}
                render_prob_bars(probs, colors)

                best = max(probs, key=probs.get)
                st.success(f"Prediction: {best} ({probs[best]*100:.1f}%)")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# =============================================================================
# Tab 3: Locker Room
# =============================================================================
with tabs[2]:
    st.markdown(
        "<div class='tile'><b>Locker Room</b> shows training artifacts found in this folder (confusion matrices, learning curves, etc.).</div>",
        unsafe_allow_html=True,
    )

    patterns = [
        "confusion_matrix*.png",
        "training_history*.png",
        "*history*.png",
        "*roc*.png",
        "*pr*.png",
    ]
    imgs = []
    for pat in patterns:
        imgs.extend(glob.glob(pat))
    imgs = sorted(set(imgs), key=lambda x: os.path.getmtime(x), reverse=True)

    if not imgs:
        st.info("No artifact images found (PNG). Put your saved plots (.png) in this same folder to display them here.")
    else:
        cols = st.columns(3)
        for i, img in enumerate(imgs):
            with cols[i % 3]:
                st.markdown("<div class='tile'>", unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.caption(f"{img} | {file_meta(img)}")
                st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='tile'>", unsafe_allow_html=True)
    st.write("**Loaded file summary**")
    st.write(
        {
            "model": mp,
            "scaler": sp,
            "label_encoder": lp,
            "team_encoder": tp,
            "matches_csv": cp,
            "badges_dir": os.path.abspath(BADGES_DIR),
        }
    )
    st.write("**Model parameters**:", f"{model.count_params():,}")
    st.markdown("</div>", unsafe_allow_html=True)