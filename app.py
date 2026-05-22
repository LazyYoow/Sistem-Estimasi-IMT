import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cv2
import io
import os
import math

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="NutriScan BMI · AI Body Analysis",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;500;700;900&family=Share+Tech+Mono&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background: #020810 !important;
    color: #c8d8e8 !important;
}

/* ── Main background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 20%, #001a2e 0%, #020810 50%, #000d1a 100%) !important;
    background-attachment: fixed !important;
}

/* ── Scanline overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,255,200,0.01) 2px,
        rgba(0,255,200,0.01) 4px
    );
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #010b18 0%, #011525 100%) !important;
    border-right: 1px solid rgba(0,255,200,0.15) !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}

/* ── Headers & Text ── */
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 0.08em !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #00ffc8 !important;
    border: 2px solid #00ffc8 !important;
    box-shadow: 0 0 12px #00ffc8 !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 0.12em !important;
    font-size: 0.75rem !important;
    background: linear-gradient(135deg, #003d2b 0%, #00261a 100%) !important;
    border: 1px solid #00ffc8 !important;
    color: #00ffc8 !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 12px rgba(0,255,200,0.2), inset 0 0 12px rgba(0,255,200,0.05) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #005a40 0%, #003d2b 100%) !important;
    box-shadow: 0 0 24px rgba(0,255,200,0.5), inset 0 0 20px rgba(0,255,200,0.1) !important;
    transform: translateY(-1px) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00ffc8 0%, #00c896 100%) !important;
    color: #001a10 !important;
    font-weight: 700 !important;
    box-shadow: 0 0 20px rgba(0,255,200,0.6) !important;
}

/* ── Selectbox / Number Input ── */
.stSelectbox [data-baseweb="select"] > div,
.stNumberInput input {
    background: rgba(0,20,35,0.9) !important;
    border: 1px solid rgba(0,255,200,0.25) !important;
    color: #c8d8e8 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: rgba(0,10,20,0.8) !important;
    border: 1px solid rgba(0,255,200,0.2) !important;
    border-radius: 4px !important;
    padding: 12px !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important;
    color: #00ffc8 !important;
    font-size: 1.6rem !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(0,255,200,0.15) !important;
    margin: 1.5rem 0 !important;
}

/* ── Info / Success / Warning boxes ── */
.stAlert {
    background: rgba(0,10,25,0.85) !important;
    border: 1px solid rgba(0,255,200,0.3) !important;
    border-radius: 4px !important;
}

/* ── DataFrame / Tables ── */
.stDataFrame { border: 1px solid rgba(0,255,200,0.15) !important; border-radius: 4px; }

/* ── Caption ── */
.stCaption { color: rgba(150,180,200,0.6) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.7rem !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #00ffc8, #00c8ff) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,255,200,0.3) !important;
    border-radius: 4px !important;
    background: rgba(0,10,20,0.5) !important;
}

/* ── Label styling ── */
.stSlider label, .stSelectbox label, .stNumberInput label {
    color: rgba(0,255,200,0.7) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Custom Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #010b18; }
::-webkit-scrollbar-thumb { background: rgba(0,255,200,0.4); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ──────────────────────────────────────────────
LABEL_MAP = {
    0: 'Extremely Weak',
    1: 'Weak',
    2: 'Normal',
    3: 'Overweight',
    4: 'Obesity',
    5: 'Extreme Obesity'
}

LABEL_CONFIG = {
    'Extremely Weak': {'color': '#ff4d6d', 'glow': '#ff006e', 'icon': '⚠️', 'bar': 5},
    'Weak':           {'color': '#ff9a3c', 'glow': '#ff7700', 'icon': '📉', 'bar': 25},
    'Normal':         {'color': '#00ffc8', 'glow': '#00ffc8', 'icon': '✅', 'bar': 50},
    'Overweight':     {'color': '#ffe566', 'glow': '#ffd700', 'icon': '📊', 'bar': 68},
    'Obesity':        {'color': '#ff6b35', 'glow': '#ff4500', 'icon': '⚠️', 'bar': 83},
    'Extreme Obesity':{'color': '#ff1744', 'glow': '#ff0000', 'icon': '🚨', 'bar': 96},
}

BMI_RANGES = [
    (0,   16,   'Extremely Weak',  '#ff4d6d'),
    (16,  18.5, 'Weak',            '#ff9a3c'),
    (18.5,25,   'Normal',          '#00ffc8'),
    (25,  30,   'Overweight',      '#ffe566'),
    (30,  40,   'Obesity',         '#ff6b35'),
    (40,  60,   'Extreme Obesity', '#ff1744'),
]

FEATURE_COLS = [
    'Gender_num','Height','Weight','Height_m','BMI','BMI_squared',
    'Weight_Height_ratio','Weight_squared','Height_squared',
    'Height_category','Weight_category',
    'pixel_area','aspect_ratio','body_density','contour_perimeter','compactness'
]

# ─── Model Loading ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Train & cache model on first run (no pkl required)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import cv2

    df = pd.read_csv('bmi.csv')

    def gen_silhouette(h, w, g, sz=128):
        img = np.zeros((sz, sz), dtype=np.uint8)
        h_n = (h - 140) / 60
        w_n = (w - 50) / 110
        bh = int(0.5 * sz + h_n * 0.35 * sz)
        bw = int(0.15 * sz + w_n * 0.3 * sz)
        sw = int(bw * (1.3 if g == 'Male' else 1.1))
        cx, cy = sz // 2, sz // 2
        cv2.ellipse(img, (cx, cy), (bw, bh // 2), 0, 0, 360, 255, -1)
        cv2.ellipse(img, (cx, cy - bh // 2 + 10), (sw, 10), 0, 0, 360, 255, -1)
        hr = max(8, int(bw * 0.5))
        cv2.circle(img, (cx, cy - bh // 2 - hr + 5), hr, 255, -1)
        return img

    def ext_feats(img):
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        pa = np.sum(binary > 0)
        coords = np.argwhere(binary > 0)
        if len(coords) > 0:
            y0, x0 = coords.min(axis=0); y1, x1 = coords.max(axis=0)
            bh2 = y1 - y0 + 1; bw2 = x1 - x0 + 1
        else:
            bh2 = bw2 = 1
        ar = bh2 / (bw2 + 1e-5)
        bd = pa / ((bh2 * bw2) + 1e-5)
        conts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        peri = cv2.arcLength(conts[0], True) if conts else 0
        comp = (4 * math.pi * pa) / (peri ** 2 + 1e-5)
        return [pa, round(ar, 4), round(bd, 4), round(peri, 4), round(comp, 4)]

    feats = []
    for _, row in df.iterrows():
        img = gen_silhouette(row['Height'], row['Weight'], row['Gender'])
        feats.append(ext_feats(img))
    df_img = pd.DataFrame(feats, columns=['pixel_area','aspect_ratio','body_density','contour_perimeter','compactness'])

    df['Height_m']            = df['Height'] / 100
    df['BMI']                 = df['Weight'] / (df['Height_m'] ** 2)
    df['BMI_squared']         = df['BMI'] ** 2
    df['Weight_Height_ratio'] = df['Weight'] / df['Height']
    df['Weight_squared']      = df['Weight'] ** 2
    df['Height_squared']      = df['Height'] ** 2
    df['Gender_num']          = (df['Gender'] == 'Male').astype(int)
    df['Height_category']     = pd.cut(df['Height'], bins=[0,155,170,185,300], labels=[0,1,2,3]).astype(int)
    df['Weight_category']     = pd.cut(df['Weight'], bins=[0,60,80,100,300], labels=[0,1,2,3]).astype(int)

    X = pd.concat([df[['Gender_num','Height','Weight','Height_m','BMI','BMI_squared',
                        'Weight_Height_ratio','Weight_squared','Height_squared',
                        'Height_category','Weight_category']], df_img], axis=1)
    y = df['Index']

    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    mdl = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    mdl.fit(Xtr, ytr)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(yte, mdl.predict(Xte))
    return mdl, sc, acc

# ─── Feature Engineering ────────────────────────────────────
def gen_silhouette_cv(h, w, g, sz=128):
    img = np.zeros((sz, sz), dtype=np.uint8)
    h_n = (h - 140) / 60
    w_n = (w - 50) / 110
    bh = int(0.5 * sz + h_n * 0.35 * sz)
    bw = int(0.15 * sz + w_n * 0.3 * sz)
    sw = int(bw * (1.3 if g == 'Male' else 1.1))
    cx, cy = sz // 2, sz // 2
    cv2.ellipse(img, (cx, cy), (bw, bh // 2), 0, 0, 360, 255, -1)
    cv2.ellipse(img, (cx, cy - bh // 2 + 10), (sw, 10), 0, 0, 360, 255, -1)
    hr = max(8, int(bw * 0.5))
    cv2.circle(img, (cx, cy - bh // 2 - hr + 5), hr, 255, -1)
    return img

def extract_image_features(img):
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    pa = np.sum(binary > 0)
    coords = np.argwhere(binary > 0)
    if len(coords) > 0:
        y0, x0 = coords.min(axis=0); y1, x1 = coords.max(axis=0)
        bh2 = y1 - y0 + 1; bw2 = x1 - x0 + 1
    else:
        bh2 = bw2 = 1
    ar = bh2 / (bw2 + 1e-5)
    bd = pa / ((bh2 * bw2) + 1e-5)
    conts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    peri = cv2.arcLength(conts[0], True) if conts else 0
    comp = (4 * math.pi * pa) / (peri ** 2 + 1e-5)
    return [pa, round(ar,4), round(bd,4), round(peri,4), round(comp,4)]

def build_feature_vector(gender, height, weight):
    h_m  = height / 100
    bmi  = weight / (h_m ** 2)
    img  = gen_silhouette_cv(height, weight, gender)
    img_feats = extract_image_features(img)
    gn   = 1 if gender == 'Male' else 0
    hcat = 0 if height<=155 else 1 if height<=170 else 2 if height<=185 else 3
    wcat = 0 if weight<=60  else 1 if weight<=80  else 2 if weight<=100  else 3
    return [gn, height, weight, h_m, bmi, bmi**2,
            weight/height, weight**2, height**2, hcat, wcat] + img_feats

def get_bmi_category(bmi):
    for lo, hi, label, _ in BMI_RANGES:
        if lo <= bmi < hi:
            return label
    return 'Extreme Obesity'

# ─── SVG Silhouette Generator ───────────────────────────────
def generate_svg_silhouette(height_cm, weight_kg, gender, bmi_label, width=200, height_px=380):
    cfg = LABEL_CONFIG.get(bmi_label, LABEL_CONFIG['Normal'])
    color = cfg['color']
    glow  = cfg['glow']

    # Normalize proportions
    h_ratio = (height_cm - 140) / (210 - 140)  # 0=short, 1=tall
    w_ratio = (weight_kg - 30)  / (170 - 30)   # 0=thin, 1=heavy

    # Figure dimensions
    fig_h   = int(height_px * (0.55 + h_ratio * 0.35))
    cx      = width // 2
    base_y  = height_px - 20

    # Body proportions
    shoulder_w = int(28 + w_ratio * 32)
    hip_w      = int(24 + w_ratio * 38) if gender == 'Female' else int(26 + w_ratio * 28)
    waist_w    = int(20 + w_ratio * 22)
    torso_h    = int(fig_h * 0.38)
    head_r     = int(14 + w_ratio * 4)
    neck_h     = 10
    leg_h      = int(fig_h * 0.42)
    leg_w      = int(10 + w_ratio * 12)
    arm_w      = int(8 + w_ratio * 8)
    arm_h      = int(torso_h * 0.85)

    top_y    = base_y - fig_h
    head_cy  = top_y + head_r
    neck_top = head_cy + head_r
    torso_top = neck_top + neck_h
    torso_bot = torso_top + torso_h
    leg_bot  = torso_bot + leg_h

    # Build SVG path - body as a smooth outline
    # Head
    head_svg = f'<ellipse cx="{cx}" cy="{head_cy}" rx="{head_r}" ry="{int(head_r*1.15)}" fill="{color}" opacity="0.9"/>'

    # Neck
    neck_svg = f'<rect x="{cx-6}" y="{neck_top}" width="12" height="{neck_h}" rx="3" fill="{color}" opacity="0.85"/>'

    # Torso (bezier-shaped with shoulders, waist, hips)
    torso_mid_y = torso_top + torso_h // 2
    # Shoulder line
    sh_y   = torso_top + 8
    # Waist y
    wa_y   = torso_top + int(torso_h * 0.55)
    # Hip y
    hi_y   = torso_bot

    # Smooth torso path
    torso_svg = f'''
    <path d="
        M {cx - 8} {torso_top}
        C {cx - shoulder_w} {torso_top}, {cx - shoulder_w} {sh_y}, {cx - shoulder_w} {sh_y}
        C {cx - shoulder_w} {wa_y - 10}, {cx - waist_w} {wa_y}, {cx - hip_w} {hi_y}
        L {cx + hip_w} {hi_y}
        C {cx + waist_w} {wa_y}, {cx + shoulder_w} {wa_y - 10}, {cx + shoulder_w} {sh_y}
        C {cx + shoulder_w} {torso_top}, {cx + 8} {torso_top}, {cx + 8} {torso_top}
        Z
    " fill="{color}" opacity="0.88"/>
    '''

    # Arms
    arm_top  = sh_y
    arm_bot  = sh_y + arm_h
    arm_l_cx = cx - shoulder_w - arm_w // 2
    arm_r_cx = cx + shoulder_w + arm_w // 2
    arms_svg = f'''
    <ellipse cx="{arm_l_cx}" cy="{(arm_top+arm_bot)//2}" rx="{arm_w//2}" ry="{(arm_bot-arm_top)//2}" fill="{color}" opacity="0.8"/>
    <ellipse cx="{arm_r_cx}" cy="{(arm_top+arm_bot)//2}" rx="{arm_w//2}" ry="{(arm_bot-arm_top)//2}" fill="{color}" opacity="0.8"/>
    '''

    # Legs
    leg_gap  = int(leg_w * 0.6)
    # Left leg
    lleg_cx  = cx - leg_gap - leg_w // 2
    rleg_cx  = cx + leg_gap + leg_w // 2
    # Slight taper for legs
    legs_svg = f'''
    <path d="M {lleg_cx - leg_w} {torso_bot} C {lleg_cx - leg_w} {torso_bot+leg_h//3}, {lleg_cx - int(leg_w*0.7)} {leg_bot-20}, {lleg_cx - int(leg_w*0.6)} {leg_bot}
             L {lleg_cx + int(leg_w*0.6)} {leg_bot} C {lleg_cx + int(leg_w*0.7)} {leg_bot-20}, {lleg_cx + leg_w} {torso_bot+leg_h//3}, {lleg_cx + leg_w} {torso_bot} Z"
          fill="{color}" opacity="0.85"/>
    <path d="M {rleg_cx - leg_w} {torso_bot} C {rleg_cx - leg_w} {torso_bot+leg_h//3}, {rleg_cx - int(leg_w*0.7)} {leg_bot-20}, {rleg_cx - int(leg_w*0.6)} {leg_bot}
             L {rleg_cx + int(leg_w*0.6)} {leg_bot} C {rleg_cx + int(leg_w*0.7)} {leg_bot-20}, {rleg_cx + leg_w} {torso_bot+leg_h//3}, {rleg_cx + leg_w} {torso_bot} Z"
          fill="{color}" opacity="0.85"/>
    '''

    # Height ruler lines
    ruler_x  = width - 22
    ruler_svg = ''
    for h_mark in range(140, 211, 10):
        frac = (h_mark - 140) / 70
        ry = int(base_y - (leg_h + torso_h + head_r * 2 + neck_h) * frac)
        tick_len = 8 if h_mark % 20 == 0 else 4
        ruler_svg += f'<line x1="{ruler_x}" y1="{ry}" x2="{ruler_x + tick_len}" y2="{ry}" stroke="rgba(0,255,200,0.35)" stroke-width="1"/>'
        if h_mark % 20 == 0:
            ruler_svg += f'<text x="{ruler_x - 2}" y="{ry + 4}" text-anchor="end" font-family="Share Tech Mono" font-size="8" fill="rgba(0,255,200,0.5)">{h_mark}</text>'

    # Ground line
    ground_svg = f'<line x1="10" y1="{base_y}" x2="{width-10}" y2="{base_y}" stroke="rgba(0,255,200,0.3)" stroke-width="1" stroke-dasharray="3,4"/>'

    # Height label above figure
    label_y = top_y - 12
    ht_label = f'<text x="{cx}" y="{label_y}" text-anchor="middle" font-family="Share Tech Mono" font-size="9" fill="rgba(0,255,200,0.7)">{height_cm} cm</text>'

    # Glow filter
    glow_filter = f'''
    <defs>
        <filter id="bodyGlow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="6" result="blur"/>
            <feComposite in="SourceGraphic" in2="blur" operator="over"/>
        </filter>
        <filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feComposite in="SourceGraphic" in2="blur" operator="over"/>
        </filter>
    </defs>
    '''

    svg = f'''
    <svg viewBox="0 0 {width} {height_px}" xmlns="http://www.w3.org/2000/svg" 
         style="background:transparent; overflow:visible;">
        {glow_filter}
        <!-- Glow layer -->
        <g filter="url(#bodyGlow)" opacity="0.4">
            {head_svg}
            {neck_svg}
            {torso_svg}
            {arms_svg}
            {legs_svg}
        </g>
        <!-- Main silhouette -->
        <g filter="url(#softGlow)">
            {head_svg}
            {neck_svg}
            {torso_svg}
            {arms_svg}
            {legs_svg}
        </g>
        {ground_svg}
        {ruler_svg}
        {ht_label}
    </svg>
    '''
    return svg

# ─── BMI Scale Bar ──────────────────────────────────────────
def bmi_scale_svg(bmi_val, width=340):
    bmi_min, bmi_max = 10, 50
    pct = min(max((bmi_val - bmi_min) / (bmi_max - bmi_min), 0), 1)
    indicator_x = int(20 + pct * (width - 40))

    segments = [
        (10, 16,   '#ff4d6d', 'Ext.Weak'),
        (16, 18.5, '#ff9a3c', 'Weak'),
        (18.5,25,  '#00ffc8', 'Normal'),
        (25, 30,   '#ffe566', 'Overweight'),
        (30, 40,   '#ff6b35', 'Obesity'),
        (40, 50,   '#ff1744', 'Ext.Obesity'),
    ]

    bar_y = 36
    bar_h = 14
    seg_svgs = ''
    for lo, hi, col, lbl in segments:
        lo_pct = (lo - bmi_min) / (bmi_max - bmi_min)
        hi_pct = (hi - bmi_min) / (bmi_max - bmi_min)
        x1 = int(20 + lo_pct * (width - 40))
        x2 = int(20 + hi_pct * (width - 40))
        seg_svgs += f'<rect x="{x1}" y="{bar_y}" width="{x2-x1}" height="{bar_h}" rx="2" fill="{col}" opacity="0.85"/>'

    # Label ticks
    tick_svgs = ''
    for bmi_tick in [16, 18.5, 25, 30, 40]:
        tp = (bmi_tick - bmi_min) / (bmi_max - bmi_min)
        tx = int(20 + tp * (width - 40))
        tick_svgs += f'<line x1="{tx}" y1="{bar_y}" x2="{tx}" y2="{bar_y+bar_h}" stroke="rgba(0,0,0,0.5)" stroke-width="1.5"/>'
        tick_svgs += f'<text x="{tx}" y="{bar_y+bar_h+14}" text-anchor="middle" font-family="Share Tech Mono" font-size="8" fill="rgba(150,180,200,0.7)">{bmi_tick}</text>'

    # Indicator
    ind_svg = f'''
    <polygon points="{indicator_x},{bar_y-4} {indicator_x-6},{bar_y-14} {indicator_x+6},{bar_y-14}"
             fill="white" opacity="0.9"/>
    <text x="{indicator_x}" y="{bar_y-17}" text-anchor="middle" 
          font-family="Share Tech Mono" font-size="10" fill="white" font-weight="bold">{bmi_val:.1f}</text>
    '''

    svg = f'''
    <svg viewBox="0 0 {width} 65" xmlns="http://www.w3.org/2000/svg" style="background:transparent; overflow:visible;">
        <text x="20" y="18" font-family="Share Tech Mono" font-size="9" fill="rgba(0,255,200,0.5)" letter-spacing="2">BMI SCALE</text>
        {seg_svgs}
        {tick_svgs}
        {ind_svg}
    </svg>
    '''
    return svg

# ─── Probability Bars SVG ───────────────────────────────────
def proba_bars_svg(proba_arr, pred_idx, width=320):
    bar_w_max = width - 160
    row_h     = 34
    total_h   = row_h * 6 + 20

    rows = ''
    for i, (prob, (_, cfg)) in enumerate(zip(proba_arr, LABEL_CONFIG.items())):
        bar_w = int(prob * bar_w_max)
        y     = 10 + i * row_h
        col   = cfg['color']
        is_pred = (i == pred_idx)
        opacity = '1' if is_pred else '0.45'
        label_text = LABEL_MAP[i]
        # truncate for space
        label_short = label_text.replace('Extremely', 'Ext.').replace('Obesity','Obes.')

        rows += f'''
        <rect x="0" y="{y+5}" width="120" height="{row_h-8}" rx="2" 
              fill="{'rgba(0,255,200,0.08)' if is_pred else 'transparent'}"/>
        <text x="4" y="{y+row_h//2+5}" font-family="Rajdhani" font-size="11" 
              fill="{col if is_pred else 'rgba(150,180,200,0.6)'}" font-weight="{'700' if is_pred else '400'}">{label_short}</text>
        <rect x="124" y="{y+8}" width="{max(bar_w,2)}" height="{row_h-14}" rx="2" 
              fill="{col}" opacity="{opacity}"/>
        {"<rect x='124' y='" + str(y+8) + "' width='" + str(max(bar_w,2)) + "' height='" + str(row_h-14) + "' rx='2' fill='" + col + "' opacity='0.15' filter='url(#pg)'/>" if is_pred else ''}
        <text x="{128 + bar_w + 4}" y="{y+row_h//2+5}" font-family="Share Tech Mono" font-size="9" 
              fill="{col if is_pred else 'rgba(120,150,170,0.7)'}">{prob*100:.1f}%</text>
        '''

    return f'''
    <svg viewBox="0 0 {width} {total_h}" xmlns="http://www.w3.org/2000/svg" style="background:transparent;">
        <defs>
            <filter id="pg"><feGaussianBlur stdDeviation="4"/></filter>
        </defs>
        {rows}
    </svg>
    '''

# ─── Main App ────────────────────────────────────────────────
def main():
    model, scaler, train_acc = load_artifacts()

    # ── Header ──
    st.markdown("""
    <div style="padding: 1.5rem 0 0.5rem; border-bottom: 1px solid rgba(0,255,200,0.15); margin-bottom: 1.5rem;">
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:rgba(0,255,200,0.5); 
                    letter-spacing:0.25em; text-transform:uppercase; margin-bottom:0.3rem;">
            ⚡ NUTRISCAN · AI BODY ANALYSIS SYSTEM v2.0
        </div>
        <h1 style="font-family:'Orbitron',sans-serif; font-size:1.8rem; font-weight:900; margin:0;
                   background: linear-gradient(90deg, #00ffc8 0%, #00c8ff 50%, #c8a0ff 100%);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:0.05em;">
            ESTIMASI INDEKS MASSA TUBUH
        </h1>
        <div style="font-family:'Rajdhani',sans-serif; color:rgba(150,180,200,0.6); font-size:0.95rem; margin-top:0.3rem;">
            Image Processing · Machine Learning · Random Forest Classifier
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Orbitron',sans-serif; font-size:0.7rem; color:#00ffc8; 
                    letter-spacing:0.2em; margin-bottom:1rem; padding-bottom:0.5rem;
                    border-bottom:1px solid rgba(0,255,200,0.15);">
            ◈ INPUT PARAMETER
        </div>
        """, unsafe_allow_html=True)

        gender = st.selectbox("JENIS KELAMIN", ["Male", "Female"], format_func=lambda x: f"♂ {x}" if x=="Male" else f"♀ {x}")
        height = st.slider("TINGGI BADAN (cm)", 140, 210, 170, step=1)
        weight = st.number_input("BERAT BADAN (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)

        h_m    = height / 100
        bmi_rt = weight / (h_m ** 2)
        bmi_cat= get_bmi_category(bmi_rt)
        cfg_rt = LABEL_CONFIG[bmi_cat]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(0,10,20,0.8); border:1px solid {cfg_rt['color']}44; border-radius:6px; 
                    padding:14px; text-align:center;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; 
                        color:rgba(150,180,200,0.5); letter-spacing:0.15em; margin-bottom:6px;">REALTIME BMI</div>
            <div style="font-family:'Orbitron',sans-serif; font-size:2rem; font-weight:900; 
                        color:{cfg_rt['color']}; text-shadow: 0 0 20px {cfg_rt['glow']}88;">
                {bmi_rt:.1f}
            </div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:0.85rem; 
                        color:{cfg_rt['color']}; margin-top:4px;">
                {cfg_rt['icon']} {bmi_cat}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ ANALISIS SEKARANG", use_container_width=True, type="primary")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; 
                    color:rgba(100,140,160,0.6); letter-spacing:0.1em; line-height:1.8;">
            MODEL ACCURACY<br>
            <span style="color:#00ffc8; font-size:0.9rem;">{train_acc*100:.1f}%</span><br><br>
            ALGORITHM<br>
            <span style="color:rgba(150,180,200,0.8);">Random Forest</span><br>
            <span style="color:rgba(100,140,160,0.5); font-size:0.6rem;">200 estimators · 16 features</span><br><br>
            FEATURES<br>
            <span style="color:rgba(150,180,200,0.6);">BMI · Image Processing<br>Anthropometric</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Main content ──
    if predict_btn or True:
        fv       = build_feature_vector(gender, height, weight)
        fv_sc    = scaler.transform([fv])
        pred_idx = model.predict(fv_sc)[0]
        pred_proba = model.predict_proba(fv_sc)[0]
        pred_label = LABEL_MAP[pred_idx]
        cfg      = LABEL_CONFIG[pred_label]

        # Top status bar
        status_color = cfg['color']
        status_glow  = cfg['glow']
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {status_color}11 0%, transparent 60%);
                    border: 1px solid {status_color}44; border-radius:8px; padding:16px 24px;
                    display:flex; align-items:center; gap:16px; margin-bottom:1.5rem;
                    box-shadow: 0 0 30px {status_glow}22, inset 0 0 30px {status_color}08;">
            <div style="font-size:2rem;">{cfg['icon']}</div>
            <div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; 
                            color:rgba(150,180,200,0.5); letter-spacing:0.2em;">HASIL KLASIFIKASI</div>
                <div style="font-family:'Orbitron',sans-serif; font-size:1.4rem; font-weight:700;
                            color:{status_color}; text-shadow: 0 0 20px {status_glow}88; letter-spacing:0.05em;">
                    {pred_label.upper()}
                </div>
            </div>
            <div style="margin-left:auto; text-align:right;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; 
                            color:rgba(150,180,200,0.5);">PROBABILITAS</div>
                <div style="font-family:'Orbitron',sans-serif; font-size:1.6rem; font-weight:900;
                            color:{status_color};">{pred_proba[pred_idx]*100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Three columns
        col_sil, col_data, col_prob = st.columns([1, 1.2, 1.3])

        # ── Silhouette Column ──
        with col_sil:
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; 
                        color:rgba(0,255,200,0.5); letter-spacing:0.2em; margin-bottom:10px;">
                ◈ BODY SILHOUETTE
            </div>
            """, unsafe_allow_html=True)

            svg_body = generate_svg_silhouette(height, weight, gender, pred_label, width=180, height_px=360)

            st.markdown(f"""
            <div style="background:rgba(0,8,18,0.9); border:1px solid rgba(0,255,200,0.15);
                        border-radius:8px; padding:16px; text-align:center; position:relative;">
                <div style="position:absolute; top:8px; right:10px; font-family:'Share Tech Mono',monospace;
                             font-size:0.6rem; color:rgba(0,255,200,0.35); letter-spacing:0.1em;">
                    {gender.upper()}
                </div>
                {svg_body}
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; 
                            color:{cfg['color']}; margin-top:6px; letter-spacing:0.1em;">
                    {height}cm · {weight}kg
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Data Column ──
        with col_data:
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; 
                        color:rgba(0,255,200,0.5); letter-spacing:0.2em; margin-bottom:10px;">
                ◈ ANALISIS DATA
            </div>
            """, unsafe_allow_html=True)

            bmi_v = weight / (height/100)**2
            bmi_svg = bmi_scale_svg(bmi_v, width=280)

            # Metrics grid
            metrics = [
                ("BMI", f"{bmi_v:.2f}", "kg/m²"),
                ("Tinggi", f"{height}", "cm"),
                ("Berat", f"{weight:.1f}", "kg"),
                ("Gender", gender[:1], ""),
                ("Kategori Tinggi", ["Pendek","Sedang","Tinggi","Sangat Tinggi"][0 if height<=155 else 1 if height<=170 else 2 if height<=185 else 3], ""),
                ("Rasio BB/TB", f"{weight/height:.3f}", ""),
            ]

            metric_html = ''
            for label, val, unit in metrics:
                metric_html += f'''
                <div style="background:rgba(0,10,20,0.8); border:1px solid rgba(0,255,200,0.12);
                            border-radius:4px; padding:10px 12px;">
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.6rem; 
                                color:rgba(0,255,200,0.4); letter-spacing:0.1em; text-transform:uppercase;">{label}</div>
                    <div style="font-family:'Orbitron',sans-serif; font-size:1.1rem; 
                                color:#c8d8e8; margin-top:2px;">{val} <span style="font-size:0.6rem; color:rgba(150,180,200,0.4);">{unit}</span></div>
                </div>
                '''

            st.markdown(f"""
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:14px;">
                {metric_html}
            </div>
            <div style="background:rgba(0,8,18,0.9); border:1px solid rgba(0,255,200,0.12);
                        border-radius:6px; padding:12px; margin-bottom:10px;">
                {bmi_svg}
            </div>
            """, unsafe_allow_html=True)

            # Tips
            tips = {
                'Extremely Weak': 'Perlu konsultasi dokter segera. Tambah asupan kalori bergizi.',
                'Weak':           'Tingkatkan asupan protein & kalori. Konsultasi ahli gizi.',
                'Normal':         'Pertahankan pola hidup sehat. Olahraga rutin & gizi seimbang.',
                'Overweight':     'Kurangi kalori, tingkatkan aktivitas fisik. Hindari makanan olahan.',
                'Obesity':        'Program penurunan berat terstruktur. Konsultasi dokter & ahli gizi.',
                'Extreme Obesity':'Intervensi medis diperlukan. Segera hubungi dokter spesialis.',
            }
            tip_text = tips.get(pred_label, '')
            st.markdown(f"""
            <div style="background:rgba(0,10,20,0.8); border-left:3px solid {cfg['color']};
                        border-radius:0 4px 4px 0; padding:10px 14px;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.6rem; 
                            color:rgba(0,255,200,0.4); letter-spacing:0.1em; margin-bottom:5px;">REKOMENDASI</div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:0.9rem; 
                            color:rgba(200,220,230,0.85); line-height:1.5;">{tip_text}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability Column ──
        with col_prob:
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; 
                        color:rgba(0,255,200,0.5); letter-spacing:0.2em; margin-bottom:10px;">
                ◈ DISTRIBUSI PROBABILITAS
            </div>
            """, unsafe_allow_html=True)

            prob_svg = proba_bars_svg(pred_proba, pred_idx, width=300)

            st.markdown(f"""
            <div style="background:rgba(0,8,18,0.9); border:1px solid rgba(0,255,200,0.15);
                        border-radius:8px; padding:16px;">
                {prob_svg}
            </div>
            """, unsafe_allow_html=True)

            # Image features
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; 
                        color:rgba(0,255,200,0.5); letter-spacing:0.2em; margin-bottom:10px;">
                ◈ IMAGE FEATURES
            </div>
            """, unsafe_allow_html=True)

            img_cv = gen_silhouette_cv(height, weight, gender)
            img_feats = extract_image_features(img_cv)
            feat_names = ['Pixel Area', 'Aspect Ratio', 'Body Density', 'Perimeter', 'Compactness']

            feat_html = ''
            for fname, fval in zip(feat_names, img_feats):
                fval_str = f'{fval:.4f}' if isinstance(fval, float) else str(fval)
                feat_html += f'''
                <div style="display:flex; justify-content:space-between; padding:6px 0;
                            border-bottom:1px solid rgba(0,255,200,0.07);">
                    <span style="font-family:'Rajdhani',sans-serif; font-size:0.8rem; 
                                 color:rgba(150,180,200,0.7);">{fname}</span>
                    <span style="font-family:'Share Tech Mono',monospace; font-size:0.75rem; 
                                 color:#00ffc8;">{fval_str}</span>
                </div>
                '''

            st.markdown(f"""
            <div style="background:rgba(0,8,18,0.9); border:1px solid rgba(0,255,200,0.12);
                        border-radius:6px; padding:12px;">
                {feat_html}
            </div>
            """, unsafe_allow_html=True)

    # ── Divider ──
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Batch Prediction ──
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; 
                color:rgba(0,255,200,0.5); letter-spacing:0.2em; margin-bottom:10px;">
        ◈ PREDIKSI BATCH — UPLOAD CSV
    </div>
    """, unsafe_allow_html=True)

    col_up, col_fmt = st.columns([2, 1])
    with col_fmt:
        st.markdown("""
        <div style="background:rgba(0,8,18,0.9); border:1px solid rgba(0,255,200,0.12);
                    border-radius:6px; padding:12px; font-family:'Share Tech Mono',monospace;
                    font-size:0.7rem; color:rgba(150,180,200,0.7);">
            FORMAT CSV:<br><br>
            <span style="color:#00ffc8;">Gender, Height, Weight</span><br>
            Male, 175, 70<br>
            Female, 160, 55<br>
            ...
        </div>
        """, unsafe_allow_html=True)

    with col_up:
        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_file:
            df_batch = pd.read_csv(uploaded_file)
            st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.75rem; color:rgba(0,255,200,0.7); margin-bottom:8px;">✓ {len(df_batch)} data ditemukan</div>', unsafe_allow_html=True)

            batch_feats = []
            for _, row in df_batch.iterrows():
                fv = build_feature_vector(row['Gender'], row['Height'], row['Weight'])
                batch_feats.append(fv)

            batch_scaled = scaler.transform(batch_feats)
            df_batch['BMI']           = df_batch['Weight'] / (df_batch['Height']/100)**2
            df_batch['Pred_Index']    = model.predict(batch_scaled)
            df_batch['Pred_Status']   = df_batch['Pred_Index'].map(LABEL_MAP)
            df_batch['BMI']           = df_batch['BMI'].round(2)

            st.dataframe(df_batch.head(30), use_container_width=True)

            csv_out = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ DOWNLOAD HASIL",
                csv_out, 'hasil_prediksi_bmi.csv', 'text/csv',
                use_container_width=True
            )

    # ── Footer ──
    st.markdown("""
    <div style="margin-top:2rem; padding-top:1rem; border-top:1px solid rgba(0,255,200,0.1);
                text-align:center; font-family:'Share Tech Mono',monospace; 
                font-size:0.65rem; color:rgba(100,140,160,0.5); letter-spacing:0.15em;">
        NUTRISCAN BMI · UNIVERSITAS AMIKOM YOGYAKARTA · DATA MINING PROJECT<br>
        <span style="color:rgba(0,255,200,0.3);">Random Forest · MLflow · Streamlit</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
