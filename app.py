"""
פורטל הערכת סיכון להתקף לב - Streamlit app (Hebrew, RTL)
Run:  streamlit run app.py
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.heart_model import CATEGORICAL, CLINICAL_STEP, SAMPLE_PATIENT, predict_with_explanation, train_model

st.set_page_config(page_title="הערכת סיכון להתקף לב | Heart Attack Risk",
                   page_icon="🫀", layout="wide", initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- #
# Hebrew labels & clinical knowledge
# --------------------------------------------------------------------------- #
FEATURE_HE = {
    "age": "גיל", "sex": "מין", "cp": "סוג כאב בחזה", "trtbps": "לחץ דם במנוחה", "chol": "כולסטרול",
    "fbs": "סוכר בצום מעל 120", "restecg": "אק\"ג במנוחה", "thalachh": "דופק מקסימלי במאמץ",
    "exng": "תעוקה במאמץ", "ca": "כלי דם מרכזיים חסומים",
    "cp_1": "כאב: תעוקה טיפוסית", "cp_2": "כאב: תעוקה לא טיפוסית", "cp_3": "כאב: לא תעוקתי",
    "restecg_1": "אק\"ג: חריגות ST-T", "restecg_2": "אק\"ג: עיבוי חדר שמאל",
    "ca_1": "כלי דם: 1", "ca_2": "כלי דם: 2", "ca_3": "כלי דם: 3",
}
CP_HE = {0: "0 – ללא תסמינים (אסימפטומטי)", 1: "1 – תעוקה טיפוסית (Typical angina)",
         2: "2 – תעוקה לא טיפוסית (Atypical)", 3: "3 – כאב שאינו תעוקתי"}
RESTECG_HE = {0: "0 – תקין", 1: "1 – חריגות בגלי ST-T", 2: "2 – עיבוי חדר שמאל (LVH)"}
CA_HE = {0: "0", 1: "1", 2: "2", 3: "3"}
UNIT_HE = {"age": "שנים", "trtbps": "mmHg", "chol": "mg/dl", "thalachh": "פעימות לדקה"}

# Direction expected from cardiology literature: +1 = higher value / presence raises risk.
LITERATURE = {
    "age": (+1, "גיל מבוגר הוא גורם סיכון מרכזי ומוכר למחלת לב כלילית."),
    "sex": (+1, "לגברים סיכון מוגבר להתקף לב בהשוואה לנשים בגילאים דומים."),
    "cp": (+1, "כאב בחזה, ובמיוחד תעוקה (אנגינה), הוא התסמין הקלאסי של אי-ספיקת זרימת דם ללב."),
    "trtbps": (+1, "יתר לחץ דם מעמיס על הלב וכלי הדם ומאיץ טרשת עורקים."),
    "chol": (+1, "כולסטרול גבוה תורם להיווצרות רובד טרשתי בעורקים הכליליים."),
    "fbs": (+1, "סוכר גבוה בצום (סוכרת) מכפיל את הסיכון הקרדיווסקולרי."),
    "restecg": (+1, "שינויים באק\"ג במנוחה עלולים להעיד על פגיעה או עומס על שריר הלב."),
    "thalachh": (-1, "דופק מקסימלי גבוה במאמץ מעיד בדרך כלל על כושר לבבי טוב; דופק נמוך עלול להעיד על בעיה."),
    "exng": (+1, "תעוקה המופיעה במאמץ היא סימן מחשיד לחסימה בעורקים הכליליים."),
    "ca": (+1, "מספר רב יותר של כלי דם מרכזיים חסומים (בצנתור) מעיד על מחלה כלילית נרחבת יותר."),
}

LRI, PDI = chr(0x2066), chr(0x2069)   # Unicode left-to-right isolate: keeps %, ×, +/- in place inside RTL text


def ltr(s) -> str:
    return f"{LRI}{s}{PDI}"


def pct(v: float, d: int = 0) -> str:
    return ltr(f"{v:.{d}%}")


def odds_mult(c: float) -> str:
    """Log-odds contribution -> 'odds ×2.1' / 'odds ÷1.6'."""
    return ltr(f"×{np.exp(c):.2f}") if c >= 0 else ltr(f"÷{np.exp(-c):.2f}")


def html(markup: str) -> None:
    """Render raw HTML; strips indentation/blank lines so Markdown never turns it into a code block."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines() if line.strip()),
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Styling: RTL + modern medical dashboard look + responsive
# --------------------------------------------------------------------------- #
html("""
<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --brand:#0e7c86; --brand-dark:#0a5960; --brand-soft:#e3f4f5; --accent:#e5484d; --ink:#152238; --muted:#5d6b80;
  --card:#ffffff; --bg:#f3f7fa; --line:#e3e9f0; --good:#12925e; --warn:#c27803; --bad:#d63a40;
  --shadow:0 4px 18px rgba(16,40,64,.07); --radius:16px;
}
html, body, [class*="css"], .stApp, .stMarkdown, p, li, label, input, button, h1,h2,h3,h4,h5,h6{
  font-family:'Heebo', sans-serif !important;
}
.stApp{ background:var(--bg); }
.stApp, .main, .block-container, [data-testid="stSidebar"]{ direction:rtl; text-align:right; }
.block-container{ padding-top:1.2rem; padding-bottom:1rem; max-width:1250px; }
[data-testid="stMarkdownContainer"] *{ text-align:right; }
[data-testid="stMarkdownContainer"] ul, [data-testid="stMarkdownContainer"] ol{ padding-right:1.3rem; padding-left:0; }
div[data-baseweb="select"] > div, .stNumberInput input{ direction:rtl; text-align:right; }
[data-testid="stSlider"] { direction:ltr; }            /* sliders stay LTR so min→max reads naturally */
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBar"] > *{ unicode-bidi:plaintext; white-space:nowrap; }
[data-testid="stElementContainer"]:has(#ra-ltr-locale){ display:none; }   /* locale-fix script takes no space */
[data-testid="stVegaLiteChart"], .vega-embed{ direction:ltr; }   /* canvas charts: RTL flips label alignment */
[data-testid="stSlider"] label, [data-testid="stWidgetLabel"]{ direction:rtl; text-align:right; width:100%; }
[data-testid="stWidgetLabel"] p{ font-weight:600; color:var(--ink); }
[data-testid="stRadio"] [role="radiogroup"]{ direction:rtl; gap:.3rem 1.2rem; }
.stTabs [data-baseweb="tab-list"]{ gap:.4rem; direction:rtl; background:#fff; padding:.35rem; border-radius:14px; box-shadow:var(--shadow); }
.stTabs [data-baseweb="tab"]{ height:46px; padding:0 1.2rem; border-radius:10px; font-weight:700; font-size:1rem; }
.stTabs [aria-selected="true"]{ background:var(--brand) !important; color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none; }
#MainMenu, footer{ visibility:hidden; }

/* hero */
.hero{ background:linear-gradient(120deg,#0a3d62 0%,#0e7c86 55%,#2bb3a3 100%); color:#fff; border-radius:22px;
  padding:2.2rem 2.4rem; box-shadow:0 12px 30px rgba(14,124,134,.25); position:relative; overflow:hidden; margin-bottom:1.2rem;}
.hero:after{ content:"❤️"; position:absolute; left:2rem; bottom:-1.2rem; font-size:7rem; opacity:.18; }
.hero h1{ color:#fff; font-size:2.3rem; font-weight:800; margin:0 0 .4rem 0; }
.hero p{ color:#e2f6f6; font-size:1.08rem; margin:0; max-width:760px; }
.chips{ margin-top:1rem; display:flex; gap:.5rem; flex-wrap:wrap; }
.chip{ background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.3); padding:.3rem .8rem; border-radius:999px; font-size:.88rem; }

/* cards */
.card{ background:var(--card); border-radius:var(--radius); box-shadow:var(--shadow); padding:1.3rem 1.4rem; border:1px solid var(--line); margin-bottom:1rem; }
.card h3{ margin:0 0 .6rem 0; font-size:1.2rem; color:var(--ink); font-weight:800; }
.section-title{ font-size:1.25rem; font-weight:800; color:var(--ink); margin:.4rem 0 .8rem 0; display:flex; gap:.5rem; align-items:center; }
.risk-card{ text-align:center !important; border-width:2px; }
.risk-card *{ text-align:center !important; }
.risk-high{ background:linear-gradient(135deg,#fff 0%,#fff1f1 100%); border-color:#f5c2c4; }
.risk-low{ background:linear-gradient(135deg,#fff 0%,#ecfaf3 100%); border-color:#bfe8d3; }
.risk-label{ color:var(--muted); font-weight:600; font-size:1rem; }
.risk-value{ font-size:2.1rem; font-weight:800; line-height:1.2; margin:.3rem 0; }
.risk-high .risk-value{ color:var(--bad); } .risk-low .risk-value{ color:var(--good); }
.risk-sub{ color:var(--muted); font-size:.95rem; }
.gauge{ position:relative; height:16px; border-radius:999px; margin:1.1rem .4rem .4rem;
  background:linear-gradient(to left,#d63a40 0%,#f0a23b 50%,#12925e 100%); direction:ltr; }
.gauge .mark{ position:absolute; top:-7px; width:6px; height:30px; background:var(--ink); border-radius:4px; transform:translateX(-50%); box-shadow:0 0 0 3px #fff; }
.gauge .th{ position:absolute; top:-4px; width:2px; height:24px; background:rgba(21,34,56,.45); transform:translateX(-50%); }
.gauge-scale{ display:flex; justify-content:space-between; direction:ltr; color:var(--muted); font-size:.8rem; margin:0 .4rem; }
.badge{ display:inline-block; padding:.25rem .75rem; border-radius:999px; font-weight:700; font-size:.85rem; margin-top:.6rem; }
.badge-good{ background:#e1f6ec; color:var(--good); } .badge-warn{ background:#fff3e0; color:var(--warn); } .badge-bad{ background:#fde7e8; color:var(--bad); }

.kpis{ display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin-bottom:1rem; }
.kpi{ background:#fff; border-radius:14px; box-shadow:var(--shadow); padding:1rem; border:1px solid var(--line); text-align:center !important; }
.kpi *{ text-align:center !important; }
.kpi .v{ font-size:1.6rem; font-weight:800; color:var(--ink); }
.kpi .l{ color:var(--muted); font-size:.9rem; font-weight:600; }
.kpi .s{ color:var(--muted); font-size:.78rem; }
.kpi.metric{ border-top:4px solid var(--brand); }

.explain{ background:#fff; border-right:6px solid var(--brand); border-radius:var(--radius); box-shadow:var(--shadow); padding:1.4rem 1.6rem; margin:1rem 0; }
.explain h3{ margin-top:0; color:var(--ink); font-weight:800; }
.explain p, .explain li{ line-height:1.75; color:#2a3648; font-size:1.02rem; }
.explain li{ margin-bottom:.35rem; }
.pos{ color:var(--bad); } .neg{ color:var(--good); }
.tag{ display:inline-block; font-size:.78rem; font-weight:700; padding:.05rem .55rem; border-radius:999px; margin-right:.3rem; }
.tag-ok{ background:#e1f6ec; color:var(--good); } .tag-rev{ background:#fff3e0; color:var(--warn); }
.ltr{ direction:ltr; unicode-bidi:embed; display:inline-block; }

.disclaimer{ background:#fff8e6; border:1px solid #f3d48b; border-right:6px solid #e0a400; border-radius:14px; padding:1rem 1.2rem; color:#5c4400; line-height:1.7; margin:1rem 0; }
.recall-box{ background:linear-gradient(135deg,#0a3d62 0%,#0e7c86 100%); color:#fff; border-radius:18px; padding:1.5rem 1.7rem; box-shadow:0 10px 26px rgba(10,61,98,.22); margin:1rem 0; }
.recall-box *{ color:#fff !important; }
.recall-box h3{ margin-top:0; font-weight:800; }
.recall-box p, .recall-box li{ line-height:1.75; font-size:1.02rem; }
.metric-grid{ display:grid; grid-template-columns:repeat(2,1fr); gap:1rem; }
.metric-card{ background:#fff; border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow); padding:1.2rem 1.3rem; }
.metric-card h4{ margin:0 0 .3rem 0; font-weight:800; color:var(--ink); font-size:1.1rem; }
.metric-card .q{ color:var(--brand-dark); font-weight:700; margin-bottom:.4rem; }
.metric-card p{ line-height:1.7; color:#2a3648; margin:.3rem 0; font-size:.98rem; }
.formula{ background:#0f1b2d; color:#e8f0ff; border-radius:10px; padding:.55rem .9rem; direction:ltr; text-align:center !important; font-family:Consolas,monospace !important; font-size:.95rem; margin:.5rem 0; }
.scale{ display:flex; gap:.35rem; flex-wrap:wrap; margin-top:.5rem; }
.scale span{ font-size:.8rem; padding:.15rem .6rem; border-radius:999px; font-weight:600; }

.footer{ margin-top:2.5rem; padding:1.4rem 1rem; text-align:center !important; color:#8a94a3; border-top:1px solid var(--line); font-size:.95rem; direction:rtl; }
.footer *{ text-align:center !important; }
.footer b{ color:var(--ink); }

/* responsive */
@media (max-width: 992px){
  .kpis{ grid-template-columns:repeat(2,1fr); }
  .hero h1{ font-size:1.9rem; }
}
@media (max-width: 640px){
  .block-container{ padding-left:.8rem; padding-right:.8rem; }
  .hero{ padding:1.4rem 1.2rem; border-radius:16px; }
  .hero h1{ font-size:1.5rem; } .hero p{ font-size:.95rem; } .hero:after{ font-size:4.5rem; }
  .risk-value{ font-size:1.6rem; }
  .kpi .v{ font-size:1.25rem; }
  .metric-grid{ grid-template-columns:1fr; }
  .stTabs [data-baseweb="tab"]{ padding:0 .6rem; font-size:.88rem; }
  .explain, .recall-box{ padding:1rem 1.1rem; }
}
</style>
""")

# Streamlit's sliders (React Aria) take their direction from the *browser locale*, not from CSS.
# In a Hebrew browser (he-IL) the thumb is mirrored while the filled track and min/max labels are not,
# so the thumb shows the wrong position. Pin the locale React Aria sees to en-US: sliders stay a
# consistent LTR number line (min on the left) for every visitor; the page itself stays RTL via CSS.
st.html("""
<span id="ra-ltr-locale"></span>
<script>
(() => {
  if (window.__raLtrLocale) return;            // install once per page, not on every rerun
  window.__raLtrLocale = true;
  try {
    if (navigator.language === "en-US") return;
    Object.defineProperty(navigator, "language", { configurable: true, get: () => "en-US" });
    // React Aria re-reads the locale on "languagechange", but only listens while a slider is mounted,
    // so fire it whenever sliders appear (they mount after this script runs, and again on tab switches).
    let queued = false;
    const notify = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; window.dispatchEvent(new Event("languagechange")); });
    };
    const seen = new WeakSet();                // only react to sliders we have not handled yet
    new MutationObserver(() => {
      for (const s of document.querySelectorAll('[data-testid="stSlider"]')) {
        if (!seen.has(s)) { seen.add(s); notify(); }
      }
    }).observe(document.body, { childList: true, subtree: true });
    notify();
  } catch (e) { /* non-critical: sliders still work, only the thumb would be mirrored */ }
})();
</script>
""", unsafe_allow_javascript=True)


# --------------------------------------------------------------------------- #
# Model (trained once and cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="מאמן את המודל...")
def get_model():
    return train_model()


bundle = get_model()
m = bundle.metrics
t = m["test"]
data = bundle.data
coefs = bundle.coefficients


def model_direction(group: str) -> int:
    """Sign of the learned effect of a feature (for one-hot groups: average over its categories)."""
    return 1 if coefs.loc[coefs["group"] == group, "coef_std"].mean() >= 0 else -1


def data_range(col: str, step: int = 1) -> tuple[int, int]:
    lo, hi = data[col].min(), data[col].max()
    return int(lo // step * step), int(-(-hi // step) * step)


# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
html(f"""
<div class="hero">
  <h1>פורטל הערכת סיכון להתקף לב</h1>
  <p>מערכת תומכת-החלטה אקדמית המבוססת על מודל <span class="ltr">Logistic Regression</span> שאומן על
  {m['n_rows']:,} רשומות מטופלים. הזינו את הנתונים הקליניים של המטופל וקבלו הסתברות לסיכון, סיווג,
  והסבר קליני שקוף לכל תחזית.</p>
  <div class="chips">
    <span class="chip">🎯 <span class="ltr">Recall = {t['recall']:.2f}</span></span>
    <span class="chip">📈 <span class="ltr">ROC-AUC = {t['roc_auc']:.2f}</span></span>
    <span class="chip">👥 {m['n_rows']:,} מטופלים</span>
    <span class="chip">🩺 הסבר סבירות קלינית</span>
  </div>
</div>
""")

tab_risk, tab_model = st.tabs(["🩺  כלי הערכת סיכון אינטראקטיבי", "📊  ביצועי המודל ומדדי הערכה"])

# =========================================================================== #
# TAB 1 – Interactive risk assessment
# =========================================================================== #
with tab_risk:
    form_col, result_col = st.columns([5, 7], gap="large")
    p0 = SAMPLE_PATIENT

    with form_col:
        st.markdown('<div class="section-title">📋 נתוני המטופל</div>', unsafe_allow_html=True)
        with st.container(border=True):
            age = st.slider("גיל (שנים)", *data_range("age"), p0["age"])
            sex = st.radio("מין", [1, 0], index=[1, 0].index(p0["sex"]), horizontal=True,
                           format_func={1: "♂ זכר", 0: "♀ נקבה"}.get)
            cp = st.selectbox("סוג כאב בחזה", list(CP_HE), index=p0["cp"], format_func=CP_HE.get,
                              help="תעוקה (Angina) = כאב לחץ בחזה עקב זרימת דם לא מספקת ללב")
            trtbps = st.slider("לחץ דם סיסטולי במנוחה (mmHg)", *data_range("trtbps", 5), p0["trtbps"], 1,
                               help="תקין: מתחת ל-120 · גבוה: 140 ומעלה")
            chol = st.slider("כולסטרול בדם (mg/dl)", *data_range("chol", 10), p0["chol"], 1,
                             help="רצוי: מתחת ל-200 · גבולי: 200–239 · גבוה: 240 ומעלה")
            thalachh = st.slider("דופק מקסימלי שהושג במבחן מאמץ (פעימות/דקה)", *data_range("thalachh", 5),
                                 p0["thalachh"], 1)
            st.caption(f"דופק מקסימלי צפוי לגיל {age} (220 פחות הגיל): {220 - age} · הושג "
                       f"{pct(thalachh / (220 - age))} מהצפוי")
            restecg = st.selectbox("תוצאת אק\"ג במנוחה", list(RESTECG_HE), index=p0["restecg"],
                                   format_func=RESTECG_HE.get)
            ca = st.select_slider("מספר כלי דם מרכזיים חסומים (בצנתור / פלואורוסקופיה)", options=list(CA_HE),
                                  value=p0["ca"], format_func=CA_HE.get)
            c1, c2 = st.columns(2)
            with c1:
                fbs = st.toggle("🩸 סוכר בצום גבוה", value=bool(p0["fbs"]), help="fasting blood sugar > 120 mg/dl")
            with c2:
                exng = st.toggle("🏃 תעוקה במאמץ", value=bool(p0["exng"]), help="כאב בחזה שמופיע בזמן מאמץ גופני")
            with st.expander("⚙️ הגדרות מתקדמות – סף החלטה"):
                threshold = st.slider("סף ההחלטה לסיווג \"סיכון מוגבר\"", 0.30, 0.70, 0.50, 0.05,
                                      help="הורדת הסף מגדילה את ה-Recall (פחות מטופלים בסיכון מתפספסים) "
                                           "על חשבון יותר התרעות שווא")
            st.button("❤️ חשב הערכת סיכון", type="primary", width="stretch")

    patient = {"age": age, "sex": sex, "cp": cp, "trtbps": trtbps, "chol": chol, "fbs": int(fbs),
               "restecg": restecg, "thalachh": thalachh, "exng": int(exng), "ca": ca}
    r = predict_with_explanation(bundle, patient, threshold)
    proba, high = r["proba"], r["prediction"] == 1

    with result_col:
        st.markdown('<div class="section-title">🔬 תוצאת הערכת הסיכון</div>', unsafe_allow_html=True)
        if proba >= 0.75:
            conf = '<span class="badge badge-bad">רמת ודאות גבוהה של המודל – סיכון מוגבר</span>'
        elif proba <= 0.25:
            conf = '<span class="badge badge-good">רמת ודאות גבוהה של המודל – סיכון נמוך</span>'
        else:
            conf = '<span class="badge badge-warn">אזור ביניים – מומלץ להסתמך על הערכה קלינית נוספת</span>'
        html(f"""
        <div class="card risk-card {'risk-high' if high else 'risk-low'}">
          <div class="risk-label">סיווג המודל (<span class="ltr">target</span>)</div>
          <div class="risk-value">{'⚠️ 1 – סיכון מוגבר להתקף לב' if high else '✅ 0 – סיכון נמוך להתקף לב'}</div>
          <div class="risk-sub">הסתברות לסיכון מוגבר (<span class="ltr">predict_proba</span>):
               <b style="font-size:1.25rem;color:var(--ink);">{pct(proba, 1)}</b></div>
          <div class="gauge"><div class="th" style="left:{threshold*100:.1f}%"></div>
               <div class="mark" style="left:{proba*100:.1f}%"></div></div>
          <div class="gauge-scale"><span>0%</span><span>{ltr(f'סף {threshold:.0%}')}</span><span>100%</span></div>
          {conf}
        </div>
        <div class="kpis">
          <div class="kpi"><div class="l">הסתברות המודל</div><div class="v">{pct(proba)}</div>
               <div class="s">למטופל זה</div></div>
          <div class="kpi"><div class="l">מטופל ממוצע</div><div class="v">{pct(r['baseline_proba'])}</div>
               <div class="s">נקודת הייחוס של המודל</div></div>
          <div class="kpi"><div class="l">{r['similar_count']} מטופלים דומים</div><div class="v">{pct(r['similar_rate'])}</div>
               <div class="s">שיעור <span class="ltr">target=1</span> בפועל</div></div>
          <div class="kpi"><div class="l">שיעור בכלל המאגר</div><div class="v">{pct(r['population_rate'])}</div>
               <div class="s">מתוך {m['n_rows']:,} מטופלים</div></div>
        </div>
        """)

        contrib = r["contributions"]
        cdf = pd.DataFrame({"מאפיין": [FEATURE_HE[k] for k in contrib.index], "השפעה": contrib.values,
                            "מכפיל": [f"×{np.exp(v):.2f}" for v in contrib.values]})
        cdf["כיוון"] = np.where(cdf["השפעה"] >= 0, "מעלה סיכון", "מוריד סיכון")
        chart = (alt.Chart(cdf).mark_bar(cornerRadius=5)
                 .encode(x=alt.X("השפעה:Q", title="השפעה על הסיכון ביחס למטופל ממוצע (log-odds)"),
                         y=alt.Y("מאפיין:N", sort=None, title=None, scale=alt.Scale(paddingInner=0.35),
                                 axis=alt.Axis(orient="right", labelLimit=220, labelFontSize=12)),
                         color=alt.Color("כיוון:N", scale=alt.Scale(domain=["מעלה סיכון", "מוריד סיכון"],
                                                                    range=["#d63a40", "#12925e"]), legend=None),
                         tooltip=["מאפיין", alt.Tooltip("השפעה:Q", format="+.3f"),
                                  alt.Tooltip("מכפיל:N", title="מכפיל סיכויים (odds)")])
                 .properties(height=330))
        with st.container(border=True):
            st.markdown("**📊 מה מזיז את הסיכון? – פירוק התחזית לפי מאפיינים**")
            html('<div style="font-size:.88rem;color:#5d6b80;"><span style="color:#d63a40;">■</span> מעלה סיכון'
                 ' &nbsp; <span style="color:#12925e;">■</span> מוריד סיכון</div>')
            st.altair_chart(chart, width="stretch")

    # ------------------------------------------------------------------- #
    # Medical & Clinical Reasonability Explanation (full width)
    # ------------------------------------------------------------------- #
    value_text = {
        "age": f"{age} שנים (ממוצע במאגר: {bundle.train_raw_means['age']:.0f})",
        "sex": "זכר" if sex else "נקבה",
        "cp": CP_HE[cp].split("–", 1)[1].strip(),
        "trtbps": f"{ltr(f'{trtbps} mmHg')}{' – יתר לחץ דם' if trtbps >= 140 else ''}",
        "chol": f"{ltr(f'{chol} mg/dl')}{' – גבוה' if chol >= 240 else ''}",
        "fbs": "מעל 120 (חשד לסוכרת)" if fbs else "תקין",
        "restecg": RESTECG_HE[restecg].split("–", 1)[1].strip(),
        "thalachh": f"{thalachh} פעימות/דקה ({pct(thalachh / (220 - age))} מהמקסימום הצפוי לגיל)",
        "exng": "קיימת" if exng else "לא קיימת",
        "ca": f"{ca} כלי דם",
    }

    def factor_li(feat: str, c: float) -> str:
        lit_dir, lit_txt = LITERATURE[feat]
        consistent = model_direction(feat) == lit_dir
        tag = ('<span class="tag tag-ok">✔ תואם ספרות רפואית</span>' if consistent else
               '<span class="tag tag-rev">⚠ כיוון הפוך לספרות – ממצא תלוי-נתונים</span>')
        effect = (f"<b class='pos'>מעלה</b> את הסיכון (סיכויים {odds_mult(c)})" if c >= 0 else
                  f"<b class='neg'>מוריד</b> את הסיכון (סיכויים {odds_mult(c)})")
        return (f"<li><b>{FEATURE_HE[feat]}</b> – {value_text[feat]}: {effect} {tag}<br>"
                f"<span style='color:#5d6b80;font-size:.93rem;'>{lit_txt}</span></li>")

    top = contrib[contrib.abs() >= 0.05].head(6)
    up, down = contrib[contrib > 0].head(3), contrib[contrib < 0].head(3)

    def names(s: pd.Series) -> str:
        return "، ".join(FEATURE_HE[k] for k in s.index) or "אין"

    gap = proba - r["similar_rate"]
    if abs(gap) <= 0.15:
        sanity = (f"ההסתברות שהמודל נתן ({pct(proba)}) <b>קרובה</b> לשיעור שנצפה בפועל בקרב {r['similar_count']} "
                  f"המטופלים הדומים ביותר במאגר ({pct(r['similar_rate'])}) – התחזית עקבית עם הנתונים ולכן סבירה.")
    else:
        sanity = (f"ההסתברות שהמודל נתן ({pct(proba)}) <b>שונה</b> מהשיעור שנצפה בקרב {r['similar_count']} "
                  f"המטופלים הדומים ביותר ({pct(r['similar_rate'])}). המודל משקלל את כל המאפיינים יחד, ולכן הפער "
                  f"נובע מצירוף ייחודי של ערכים – מומלץ לפרש את התוצאה בזהירות.")

    reversed_feats = [FEATURE_HE[f] for f in LITERATURE if model_direction(f) != LITERATURE[f][0]]
    html(f"""
    <div class="explain">
      <h3>🩺 הסבר סבירות קלינית</h3>
      <p>המודל מתחיל מ<b>מטופל ממוצע במאגר</b> (הסתברות {pct(r['baseline_proba'])}) ומעדכן את הסיכון לפי הסטייה
      של כל מאפיין מהממוצע. במודל לוגיסטי כל מאפיין <b>מכפיל את יחס הסיכויים (odds)</b> בגורם קבוע;
      המכפלה של כולם מתורגמת חזרה להסתברות – כאן <b>{pct(proba, 1)}</b>, ולכן הסיווג הוא
      <b>{'1 – סיכון מוגבר' if high else '0 – סיכון נמוך'}</b> (סף {pct(threshold)}).</p>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem;">
        <div><b class="pos">▲ גורמים שמעלים את הסיכון:</b> {names(up)}</div>
        <div><b class="neg">▼ גורמים שמורידים את הסיכון:</b> {names(down)}</div>
      </div>
      <p style="margin-top:.8rem;"><b>ניתוח קליני לפי מאפיינים (מהמשפיע ביותר):</b></p>
      <ul>{''.join(factor_li(f, c) for f, c in top.items())}</ul>
      <p><b>בדיקת סבירות מול מטופלים דומים:</b> {sanity}</p>
      <p><b>הערה מתודולוגית חשובה:</b> המודל לומד <b>קשרים סטטיסטיים מתוך המאגר</b>, לא פיזיולוגיה. בחלק מהמאפיינים
      ({'، '.join(reversed_feats) or 'אין'}) הכיוון שנלמד הפוך למצופה מהספרות הקרדיולוגית. זהו ממצא מוכר בגרסה הציבורית
      של מאגר זה (<span class="ltr">UCI / Kaggle Heart Disease</span>), שבה קידוד עמודת המטרה ואופן גיוס המטופלים
      (למשל, מטופלים ללא תסמינים שהופנו לבירור עקב ממצאים אחרים) יוצרים קשרים שאינם סיבתיים. לכן יש לפרש את
      ההסבר כ"מה המודל למד מהנתונים" ולא כהמלצה רפואית.</p>
    </div>
    <div class="disclaimer">⚠️ <b>הבהרה רפואית:</b> זהו מודל <span class="ltr">Machine Learning</span> אקדמי שנבנה לצורכי
    לימוד בלבד, על מדגם קטן ({m['n_rows']:,} מטופלים) ועם מגבלות ידועות בנתונים. התוצאה <b>אינה אבחנה רפואית</b>,
    אינה מחליפה בדיקה, אק"ג, בדיקות מעבדה או שיקול דעת של רופא, ואין להסתמך עליה לקבלת החלטות טיפוליות.
    בכל מקרה של כאב בחזה, קוצר נשימה או תסמין חריג – יש לפנות מיד לרופא או למוקד חירום (מד"א <span class="ltr">101</span>).</div>
    """)

# =========================================================================== #
# TAB 2 – Model performance & insights
# =========================================================================== #
GRADES = [(0.85, "מצוין", "badge-good"), (0.75, "טוב", "badge-good"), (0.65, "סביר", "badge-warn"), (0, "חלש", "badge-bad")]


def grade(v: float) -> str:
    label, cls = next((lbl, c) for th, lbl, c in GRADES if v >= th)
    return f'<span class="badge {cls}" style="margin-top:.3rem;">{label}</span>'


with tab_model:
    st.markdown('<div class="section-title">🎯 מדדי ביצוע על סט הבדיקה</div>', unsafe_allow_html=True)
    b = m["baseline_majority"]
    html(f"""
    <div class="kpis">
      <div class="kpi metric"><div class="l">Accuracy · דיוק כללי</div><div class="v">{pct(t['accuracy'], 1)}</div>
           <div class="s">אימון: {pct(m['train']['accuracy'], 1)} · נאיבי: {pct(b['accuracy'], 1)}</div>{grade(t['accuracy'])}</div>
      <div class="kpi metric"><div class="l">Precision · דיוק חיובי</div><div class="v">{pct(t['precision'], 1)}</div>
           <div class="s">אימון: {pct(m['train']['precision'], 1)}</div>{grade(t['precision'])}</div>
      <div class="kpi metric" style="border-top-color:#d63a40;"><div class="l">Recall · רגישות ⭐</div><div class="v">{pct(t['recall'], 1)}</div>
           <div class="s">אימון: {pct(m['train']['recall'], 1)} · <span class="ltr">CV: {m['cv_recall_mean']:.1%} ± {m['cv_recall_std']:.1%}</span></div>{grade(t['recall'])}</div>
      <div class="kpi metric"><div class="l">F1-Score · ציון משולב</div><div class="v">{pct(t['f1'], 1)}</div>
           <div class="s">אימון: {pct(m['train']['f1'], 1)} · <span class="ltr">CV: {m['cv_f1_mean']:.1%}</span></div>{grade(t['f1'])}</div>
    </div>
    <p style="color:#5d6b80;font-size:.92rem;">המדדים חושבו על <b>{m['n_test']} מטופלים בסט הבדיקה</b> (20% מהנתונים) שהמודל לא ראה
    באימון, בסף החלטה 0.5. <span class="ltr">ROC-AUC = {t['roc_auc']:.3f}</span> – יכולת ההפרדה הכללית של המודל בין
    מטופלים בסיכון לשאינם בסיכון (0.5 = ניחוש, 1.0 = מושלם). "נאיבי" = מודל שמנחש תמיד את הקבוצה הנפוצה.</p>
    """)

    st.markdown('<div class="section-title">📚 מה המדדים אומרים? – הסבר בשפה פשוטה</div>', unsafe_allow_html=True)
    scale = ('<div class="scale"><span style="background:#e1f6ec;color:#12925e;">85%+ מצוין</span>'
             '<span style="background:#e1f6ec;color:#12925e;">75–85% טוב</span>'
             '<span style="background:#fff3e0;color:#c27803;">65–75% סביר</span>'
             '<span style="background:#fde7e8;color:#d63a40;">מתחת ל-65% חלש</span></div>')
    html(f"""
    <div class="metric-grid">
      <div class="metric-card"><h4>✅ <span class="ltr">Accuracy</span> – דיוק כללי</h4>
        <div class="q">"מתוך כל המטופלים, בכמה המודל צדק?"</div>
        <div class="formula">(TP + TN) / All</div>
        <p>אצלנו: המודל סיווג נכון <b>{t['tp'] + t['tn']} מתוך {m['n_test']}</b> מטופלים ({pct(t['accuracy'])}).
        מדד אינטואיטיבי, אך <b>מטעה כשהקבוצות לא מאוזנות</b> – מודל שמכריז על כולם "בריאים" יכול לקבל דיוק גבוה
        ובכל זאת לפספס את כל החולים.</p>{scale}</div>
      <div class="metric-card"><h4>🎯 <span class="ltr">Precision</span> – דיוק חיובי</h4>
        <div class="q">"כשהמודל מתריע על סיכון – כמה פעמים הוא צודק?"</div>
        <div class="formula">TP / (TP + FP)</div>
        <p>אצלנו: מתוך <b>{t['tp'] + t['fp']}</b> התרעות, <b>{t['tp']}</b> היו נכונות ({pct(t['precision'])}).
        Precision נמוך = הרבה <b>"אזעקות שווא"</b>: בדיקות המשך מיותרות, חרדה ועלויות – אך לרוב ניתן לתקן אותן
        בבדיקה נוספת.</p>{scale}</div>
      <div class="metric-card" style="border:2px solid #f5c2c4;"><h4>❤️ <span class="ltr">Recall</span> (רגישות) – המדד הקריטי</h4>
        <div class="q">"מתוך כל המטופלים שבאמת בסיכון – כמה המודל זיהה?"</div>
        <div class="formula">TP / (TP + FN)</div>
        <p>אצלנו: המודל זיהה <b>{t['tp']} מתוך {t['tp'] + t['fn']}</b> המטופלים בסיכון ({pct(t['recall'])}) ופספס
        <b>{t['fn']}</b>. כל פספוס הוא מטופל שנשלח הביתה בטעות – ולכן ברפואה זהו המדד החשוב ביותר.</p>{scale}</div>
      <div class="metric-card"><h4>⚖️ <span class="ltr">F1-Score</span> – ציון משולב</h4>
        <div class="q">"האם המודל מאוזן – גם תופס חולים וגם לא מציף בהתרעות?"</div>
        <div class="formula">2 · (Precision · Recall) / (Precision + Recall)</div>
        <p>ממוצע הרמוני של Precision ו-Recall ({pct(t['f1'])} אצלנו). הוא "מעניש" חוסר איזון: אם אחד המדדים
        נמוך, גם F1 יהיה נמוך. שימושי להשוואה בין מודלים בציון אחד.</p>{scale}</div>
    </div>
    """)

    html(f"""
    <div class="recall-box">
      <h3>❤️ למה Recall הוא המדד הקריטי ביותר בהקשר רפואי?</h3>
      <p>בבדיקת סקר לסיכון להתקף לב, שתי הטעויות האפשריות <b>אינן שקולות</b>:</p>
      <ul>
        <li><b>False Negative – פספוס ({t['fn']} מטופלים בסט הבדיקה):</b> מטופל בסיכון אמיתי שסווג כ"תקין". הוא לא יופנה
        לבירור, לא יקבל טיפול מונע – והתוצאה עלולה להיות התקף לב, נזק בלתי הפיך או מוות.</li>
        <li><b>False Positive – אזעקת שווא ({t['fp']} מטופלים):</b> מטופל תקין שסווג "בסיכון". המחיר: בדיקות נוספות
        (אק"ג, מבחן מאמץ, אקו) וחרדה זמנית – לא נעים, אך הפיך ובטוח.</li>
      </ul>
      <p>לכן בבדיקות סקר רפואיות מעדיפים <b>Recall גבוה</b> גם במחיר ירידה ב-Precision. הדרך המעשית לכך היא
      <b>הורדת סף ההחלטה</b>: בטבלה למטה רואים שבסף {m['thresholds'][0]['threshold']:.1f} מספר הפספוסים יורד
      ל-{m['thresholds'][0]['fn']} (<span class="ltr">Recall {m['thresholds'][0]['recall']:.0%}</span>), תמורת יותר
      אזעקות שווא. את הסף ניתן לשנות בכלי ההערכה (טאב 1 ← הגדרות מתקדמות).</p>
    </div>
    """)

    cm_col, th_col = st.columns([6, 6], gap="large")
    with cm_col:
        st.markdown('<div class="section-title">🧮 מטריצת בלבול (Confusion Matrix)</div>', unsafe_allow_html=True)
        cm = pd.DataFrame([
            {"בפועל": "בפועל: סיכון (1)", "חזוי": "חזוי: סיכון (1)", "n": t["tp"], "תווית": f"TP – זיהוי נכון\n{t['tp']}", "סוג": "נכון"},
            {"בפועל": "בפועל: סיכון (1)", "חזוי": "חזוי: תקין (0)", "n": t["fn"], "תווית": f"FN – פספוס!\n{t['fn']}", "סוג": "פספוס"},
            {"בפועל": "בפועל: תקין (0)", "חזוי": "חזוי: סיכון (1)", "n": t["fp"], "תווית": f"FP – אזעקת שווא\n{t['fp']}", "סוג": "אזעקה"},
            {"בפועל": "בפועל: תקין (0)", "חזוי": "חזוי: תקין (0)", "n": t["tn"], "תווית": f"TN – שלילה נכונה\n{t['tn']}", "סוג": "נכון"},
        ])
        order_pred = ["חזוי: סיכון (1)", "חזוי: תקין (0)"]
        order_act = ["בפועל: סיכון (1)", "בפועל: תקין (0)"]
        base = alt.Chart(cm).encode(
            x=alt.X("חזוי:N", sort=order_pred, title=None, axis=alt.Axis(orient="top", labelAngle=0, labelFontSize=13)),
            y=alt.Y("בפועל:N", sort=order_act, title=None, axis=alt.Axis(orient="right", labelFontSize=13, labelLimit=200)))
        rect = base.mark_rect(cornerRadius=8, stroke="#fff", strokeWidth=4).encode(
            color=alt.Color("סוג:N", legend=None, scale=alt.Scale(domain=["נכון", "פספוס", "אזעקה"],
                                                                   range=["#cfeee0", "#f7c6c8", "#fde3b5"])),
            tooltip=["בפועל", "חזוי", "n"])
        txt = base.mark_text(fontSize=17, fontWeight="bold", lineBreak="\n", color="#152238").encode(text="תווית:N")
        with st.container(border=True):
            st.altair_chart((rect + txt).properties(height=300), width="stretch")
            st.caption("🟩 תחזיות נכונות · 🟥 פספוס (False Negative) – הטעות המסוכנת · 🟧 אזעקת שווא (False Positive)")

    with th_col:
        st.markdown('<div class="section-title">🎚️ השפעת סף ההחלטה</div>', unsafe_allow_html=True)
        th = pd.DataFrame(m["thresholds"])
        th_view = pd.DataFrame({
            "סף": th["threshold"].map("{:.1f}".format),
            "Recall": th["recall"].map("{:.1%}".format),
            "Precision": th["precision"].map("{:.1%}".format),
            "F1": th["f1"].map("{:.1%}".format),
            "פספוסים (FN)": th["fn"], "אזעקות שווא (FP)": th["fp"],
        })
        with st.container(border=True):
            st.dataframe(th_view, hide_index=True, width="stretch")
            st.caption("סף נמוך ← יותר מטופלים מסווגים 'בסיכון' ← Recall עולה ופספוסים יורדים, "
                       "אך Precision יורד. בבדיקת סקר רפואית מקובל לבחור סף שממזער פספוסים.")

    # ------------------------------------------------------------------- #
    # Coefficients / feature importance
    # ------------------------------------------------------------------- #
    st.markdown('<div class="section-title">🧬 חשיבות מאפיינים וניתוח מקדמים</div>', unsafe_allow_html=True)
    cf = coefs.copy()
    cf["מאפיין"] = cf["feature"].map(FEATURE_HE)
    cf["כיוון"] = np.where(cf["coef_std"] >= 0, "מעלה סיכון", "מוריד סיכון")
    cf["OR"] = cf["odds_ratio_std"].map("×{:.2f}".format)
    coef_chart = (alt.Chart(cf).mark_bar(cornerRadius=5)
                  .encode(x=alt.X("coef_std:Q", title="מקדם מתוקנן (שינוי ב-log-odds לסטיית תקן אחת)"),
                          y=alt.Y("מאפיין:N", sort=None, title=None, scale=alt.Scale(paddingInner=0.3),
                                  axis=alt.Axis(orient="right", labelLimit=220, labelFontSize=12)),
                          color=alt.Color("כיוון:N", scale=alt.Scale(domain=["מעלה סיכון", "מוריד סיכון"],
                                                                     range=["#d63a40", "#12925e"]), legend=None),
                          tooltip=["מאפיין", alt.Tooltip("coef_std:Q", title="מקדם", format="+.3f"),
                                   alt.Tooltip("OR:N", title="Odds Ratio")])
                  .properties(height=430))
    imp_col, tbl_col = st.columns([7, 5], gap="large")
    with imp_col:
        with st.container(border=True):
            html('<div style="font-size:.88rem;color:#5d6b80;"><span style="color:#d63a40;">■</span> מעלה סיכון'
                 ' &nbsp; <span style="color:#12925e;">■</span> מוריד סיכון · ממוין לפי עוצמת ההשפעה</div>')
            st.altair_chart(coef_chart, width="stretch")
    with tbl_col:
        step_txt = {"age": "לכל 10 שנים", "trtbps": "לכל 10 mmHg", "chol": "לכל 50 mg/dl", "thalachh": "לכל 10 פעימות"}
        tbl = pd.DataFrame({
            "מאפיין": cf["מאפיין"],
            "Odds Ratio": cf["odds_ratio_step"].map("×{:.2f}".format),
            "יחידה": [step_txt.get(f, "לעומת קטגוריית הבסיס" if "_" in f else "נוכחות (1 מול 0)") for f in cf["feature"]],
        })
        with st.container(border=True):
            st.markdown("**📐 יחס סיכויים (Odds Ratio) ביחידות קליניות**")
            st.dataframe(tbl, hide_index=True, width="stretch", height=360)
            st.caption("Odds Ratio גדול מ-1 ← מעלה את הסיכויים; קטן מ-1 ← מוריד. "
                       "קטגוריות בסיס: כאב – אסימפטומטי, אק\"ג – תקין, כלי דם – 0.")

    consistent = [FEATURE_HE[f] for f in LITERATURE if model_direction(f) == LITERATURE[f][0]]
    reversed_ = [FEATURE_HE[f] for f in LITERATURE if model_direction(f) != LITERATURE[f][0]]
    top3 = "، ".join(FEATURE_HE[f] for f in coefs["feature"].head(3))
    html(f"""
    <div class="explain">
      <h3>💡 תובנות מרכזיות</h3>
      <ul>
        <li><b>המאפיינים המשפיעים ביותר:</b> {top3}. המקדמים חושבו אחרי <span class="ltr">StandardScaler</span>, ולכן
        ניתן להשוות ביניהם ישירות – גודל המקדם = עוצמת ההשפעה, הסימן = הכיוון.</li>
        <li><b>תואמים את הספרות הרפואית:</b> {'، '.join(consistent) or 'אין'}.</li>
        <li><b>כיוון הפוך לספרות:</b> {'، '.join(reversed_) or 'אין'}. זוהי תזכורת שהמודל לומד <b>מתאם ולא סיבתיות</b>:
        בגרסה הציבורית של מאגר זה, קידוד עמודת המטרה ותהליך הפניית המטופלים יוצרים קשרים מפתיעים. בפרויקט
        אמיתי היה נדרש אימות מול צוות קליני ומאגר חיצוני לפני כל שימוש.</li>
        <li><b>הכללה:</b> פער מתון בין ביצועי האימון ({pct(m['train']['f1'])} F1) לבדיקה ({pct(t['f1'])} F1), ו-Recall
        יציב ב-<span class="ltr">5-fold Cross-Validation</span> ({pct(m['cv_recall_mean'])} ± {pct(m['cv_recall_std'])}) –
        המודל אינו סובל מ-Overfitting משמעותי.</li>
      </ul>
    </div>
    """)

    with st.expander("🔧 צינור עיבוד הנתונים ופרטי המודל"):
        html(f"""
        <ol>
          <li><b>טעינה ובדיקה:</b> {m['n_rows_raw']} רשומות, 11 עמודות · {m['missing_values_raw']} ערכים חסרים ·
          {m['duplicates_raw']} כפילויות. (הקוד ממלא ערכים חסרים בחציון/שכיח אם יופיעו.)</li>
          <li><b>One-Hot Encoding</b> למשתנים רב-קטגוריים <span class="ltr">cp, restecg, ca</span> עם
          <span class="ltr">drop_first</span> – הקטגוריה 0 משמשת בסיס להשוואה. המשתנים הבינאריים
          <span class="ltr">sex, fbs, exng</span> נשארו כמות שהם. סה"כ {m['n_features']} מאפיינים.</li>
          <li><b>חלוקה</b> מרובדת (stratified) <span class="ltr">80/20</span>: {m['n_train']} לאימון, {m['n_test']} לבדיקה.</li>
          <li><b>StandardScaler</b> – נרמול כל המאפיינים (ממוצע 0, סטיית תקן 1), מותאם על סט האימון בלבד.</li>
          <li><b>LogisticRegression</b> (רגולריזציית L2 ברירת מחדל) בתוך <span class="ltr">Pipeline</span> אחד.</li>
        </ol>
        <div class="formula">P(target=1) = 1 / (1 + e^-(b0 + b1·x1 + … + bn·xn))</div>
        """)

html("""
<div class="footer">
  <div><b>© כל הזכויות שמורות לאיתי בסטקר</b></div>
  <div style="margin-top:.3rem;">פרויקט אקדמי · Logistic Regression · Streamlit · אינו מהווה ייעוץ או אבחנה רפואית</div>
</div>
""")
