import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from langchain_ollama import OllamaLLM
import requests
import re
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DB_URL  = "postgresql+psycopg2://postgres:newpass123@localhost:8080/healthcare_db"
TABLE   = "patient_predictions"
API_URL = "http://127.0.0.1:8000/predict"

BG     = "#0f1829"
CYAN   = "#00d4ff"
PURPLE = "#7b61ff"
GREEN  = "#00e5a0"
RED    = "#ff4d6d"
ORANGE = "#ffb347"
GRID   = "#1e3a5f"
TEXT   = "#8899bb"

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;background:#0a0e1a;color:#e8eaf0;}
.stApp{background:#0a0e1a;}
.hero-header{background:linear-gradient(135deg,#0d1b2e,#0a2540,#0d1b2e);
    border:1px solid #1e3a5f;border-radius:16px;padding:1.8rem 2.5rem;margin-bottom:1.5rem;}
.hero-title{font-size:1.9rem;font-weight:800;
    background:linear-gradient(90deg,#00d4ff,#7b61ff);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 0.3rem 0;}
.hero-sub{color:#6b7fa3;font-size:0.8rem;font-family:'DM Mono',monospace;margin:0;}
.stat-card{background:#0f1829;border:1px solid #1e3a5f;border-radius:12px;padding:1rem 1.2rem;}
.stat-label{font-family:'DM Mono',monospace;font-size:0.65rem;color:#6b7fa3;
    text-transform:uppercase;letter-spacing:1px;margin-bottom:0.4rem;}
.stat-value{font-size:1.7rem;font-weight:700;line-height:1;}
.stat-delta{font-family:'DM Mono',monospace;font-size:0.7rem;margin-top:0.2rem;opacity:0.7;}
.section-title{font-size:0.68rem;font-family:'DM Mono',monospace;color:#6b7fa3;
    text-transform:uppercase;letter-spacing:2px;margin:1.5rem 0 1rem 0;
    padding-bottom:0.4rem;border-bottom:1px solid #1e3a5f;}
.answer-box{background:linear-gradient(135deg,#0d2137,#0f1829);
    border:1px solid #00d4ff33;border-left:3px solid #00d4ff;
    border-radius:12px;padding:1.3rem 1.5rem;margin-top:1rem;
    font-size:1rem;line-height:1.6;color:#c8d8f0;}
.answer-box strong{color:#00d4ff;}
.stTextInput>div>div>input{background:#0f1829!important;border:1px solid #1e3a5f!important;
    border-radius:10px!important;color:#e8eaf0!important;font-family:'DM Mono',monospace!important;}
.stTextInput>div>div>input:focus{border-color:#00d4ff!important;}
.stNumberInput>div>div>input{background:#0f1829!important;border:1px solid #1e3a5f!important;
    border-radius:10px!important;color:#e8eaf0!important;}
.stSelectbox>div>div{background:#0f1829!important;border:1px solid #1e3a5f!important;border-radius:10px!important;}
.stButton>button{background:linear-gradient(135deg,#00d4ff,#7b61ff)!important;
    border:none!important;border-radius:10px!important;color:#0a0e1a!important;
    font-family:'Syne',sans-serif!important;font-weight:700!important;}
[data-testid="stSidebar"]{background:#0d1220!important;border-right:1px solid #1e3a5f;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# INIT
# ─────────────────────────────────────────
@st.cache_resource
def get_llm():
    return OllamaLLM(model="phi3", temperature=0)

@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

@st.cache_data(ttl=300)
def load_all():
    return pd.read_sql(f"SELECT * FROM {TABLE}", get_engine())

llm    = get_llm()
engine = get_engine()

try:
    df_all      = load_all()
    data_loaded = True
except Exception as e:
    data_loaded = False

# ─────────────────────────────────────────
# LLM SQL AGENT
# ─────────────────────────────────────────
PROMPT = """You are a PostgreSQL expert working with a healthcare database.
Your job is to read the user's question carefully and generate the correct SQL query.

Table name: patient_predictions
Columns:
  - id              : integer
  - age             : float  (patient age in years)
  - gender          : string ('1' = Male, '0' = Female)
  - length_of_stay  : float  (hospital stay in days)
  - icu_los         : float  (ICU stay in days)
  - num_diagnoses   : integer
  - num_medications : integer
  - probability     : float  (readmission probability between 0 and 1)
  - risk_level      : string (exactly 'High', 'Low', or 'Medium')
  - created_at      : timestamp

Rules:
- Output ONLY the raw SQL query. Nothing else.
- Always start with SELECT
- Never use backticks or markdown
- Never include the database name
- Use AND for multiple conditions
- For "top N" or "show" queries use ORDER BY + LIMIT
- risk_level values are exactly: 'High', 'Low', 'Medium'
- gender values are exactly: '1' for Male, '0' for Female

Examples:
Q: How many high risk patients are there?
A: SELECT COUNT(*) FROM patient_predictions WHERE risk_level = 'High'

Q: Give me top 10 patients with age greater than 60
A: SELECT * FROM patient_predictions WHERE age > 60 ORDER BY probability DESC LIMIT 10

Q: How many patients have more than 5 diagnoses and are high risk?
A: SELECT COUNT(*) FROM patient_predictions WHERE num_diagnoses > 5 AND risk_level = 'High'

Q: How many patients have ICU stay greater than 7 days and are high risk?
A: SELECT COUNT(*) FROM patient_predictions WHERE icu_los > 7 AND risk_level = 'High'

Q: Show me patients with more than 15 medications and high risk
A: SELECT * FROM patient_predictions WHERE num_medications > 15 AND risk_level = 'High' ORDER BY probability DESC LIMIT 10

Q: What is the average age of female high risk patients?
A: SELECT AVG(age) FROM patient_predictions WHERE gender = '0' AND risk_level = 'High'

Q: How many male patients are high risk?
A: SELECT COUNT(*) FROM patient_predictions WHERE gender = '1' AND risk_level = 'High'

Q: What is the average ICU stay for high risk patients?
A: SELECT AVG(icu_los) FROM patient_predictions WHERE risk_level = 'High'

Q: How many patients have more than 10 diagnoses and probability greater than 0.8?
A: SELECT COUNT(*) FROM patient_predictions WHERE num_diagnoses > 10 AND probability > 0.8

Q: What is the average length of stay for patients older than 65?
A: SELECT AVG(length_of_stay) FROM patient_predictions WHERE age > 65

Q: Give me top 10 patients who have high num_medications
A: SELECT * FROM patient_predictions ORDER BY num_medications DESC LIMIT 10

Q: Show top 5 female patients who are high risk
A: SELECT * FROM patient_predictions WHERE gender = '0' AND risk_level = 'High' ORDER BY probability DESC LIMIT 5

Q: {question}
A:"""

def clean_sql(raw: str) -> str:
    raw = raw.replace("```sql","").replace("```","").strip()
    m   = re.search(r"(SELECT\b.+?)(?:;|\n\n|$)", raw, re.IGNORECASE|re.DOTALL)
    sql = m.group(1).strip() if m else raw.strip()
    # Fix wrong table names the LLM might generate
    for p in [r'\bpatient\s+end\b', r'\bpatients\b', r'\bpatient_data\b', r'\bhealthcare_db\.\w+']:
        sql = re.sub(p, TABLE, sql, flags=re.IGNORECASE)
    if TABLE not in sql:
        sql = re.sub(r'FROM\s+\S+', f'FROM {TABLE}', sql, flags=re.IGNORECASE)
    return sql.rstrip(";").strip()

def ask_agent(question: str) -> tuple:
    """Send question to phi3 → get SQL → run on DB → return (answer_text, dataframe, sql)"""
    # Step 1: LLM generates SQL
    raw_sql = llm.invoke(PROMPT.format(question=question))
    sql     = clean_sql(raw_sql)

    # Step 2: Run SQL on PostgreSQL
    df = pd.read_sql(sql, engine)

    # Step 3: Format a human-readable answer
    answer = format_answer(question, df)

    return answer, df, sql

def format_answer(question: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "No matching records found for your question."
    q = question.lower()
    # Single value result
    if df.shape == (1, 1):
        val = df.iloc[0, 0]
        if isinstance(val, float): val = round(val, 2)
        if any(k in q for k in ["how many","count","total","number of"]):
            if   "high risk"  in q or ("high" in q and "risk" in q): return f"There are **{val}** patients matching that criteria."
            elif "low risk"   in q: return f"There are **{val}** low-risk patients matching that criteria."
            elif "female"     in q: return f"There are **{val}** patients matching that criteria."
            elif "male"       in q: return f"There are **{val}** patients matching that criteria."
            else:                   return f"There are **{val}** patients matching that criteria."
        elif any(k in q for k in ["average","avg","mean"]):
            if   "age"        in q: return f"The average age is **{val}** years."
            elif "icu"        in q: return f"The average ICU stay is **{val}** days."
            elif "length"     in q or "stay" in q: return f"The average length of stay is **{val}** days."
            elif "prob"       in q: return f"The average readmission probability is **{val}**."
            elif "medication" in q: return f"The average number of medications is **{val}**."
            elif "diagnos"    in q: return f"The average number of diagnoses is **{val}**."
            else:                   return f"The average is **{val}**."
        elif any(k in q for k in ["max","maximum","highest","longest"]):
            if   "age"        in q: return f"The oldest patient is **{val}** years old."
            elif "icu"        in q: return f"The longest ICU stay is **{val}** days."
            elif "prob"       in q: return f"The highest readmission probability is **{val}**."
            else:                   return f"The maximum value is **{val}**."
        elif any(k in q for k in ["min","minimum","lowest","shortest"]):
            if   "age"        in q: return f"The youngest patient is **{val}** years old."
            elif "icu"        in q: return f"The shortest ICU stay is **{val}** days."
            else:                   return f"The minimum value is **{val}**."
        else:
            return f"Result: **{val}**."
    # Multi-row result — return None, will show as table
    return None

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <p class="hero-title">Healthcare AI Assistant</p>
  <p class="hero-sub">Patient Readmission Risk · MIMIC Dataset · XGBoost + FastAPI + GenAI</p>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────
if data_loaded:
    total    = len(df_all)
    high_r   = int((df_all['risk_level']=='High').sum())
    low_r    = int((df_all['risk_level']=='Low').sum())
    med_r    = int((df_all['risk_level']=='Medium').sum())
    avg_age  = round(df_all['age'].mean(),1)
    avg_stay = round(df_all['length_of_stay'].mean(),1)
    avg_icu  = round(df_all['icu_los'].mean(),1)
    avg_prob = round(df_all['probability'].mean(),3)
    pct_high = round(high_r/total*100,1)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,color,label,val,delta in [
        (c1,CYAN,  "Total Patients", f"{total:,}",  "all records"),
        (c2,RED,   "High Risk",      f"{high_r:,}", f"{pct_high}% of total"),
        (c3,GREEN, "Avg Age",        f"{avg_age}",  "years"),
        (c4,PURPLE,"Avg ICU Stay",   f"{avg_icu}",  "days"),
        (c5,ORANGE,"Avg Probability",f"{avg_prob}", "readmission risk"),
    ]:
        with col:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value" style="color:{color}">{val}</div>
                <div class="stat-delta" style="color:{color}">{delta}</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  Dashboard", "🤖  AI Assistant", "🔬  Live Prediction"])

# ══════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════
with tab1:
    if not data_loaded:
        st.error("Cannot connect to database.")
    else:
        def dark_fig(w=4,h=3):
            fig,ax = plt.subplots(figsize=(w,h))
            fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
            ax.tick_params(colors=TEXT,labelsize=8)
            ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)
            ax.grid(color=GRID,linewidth=0.5,linestyle='--',alpha=0.5)
            return fig,ax

        st.markdown('<div class="section-title">📊 Patient Overview</div>', unsafe_allow_html=True)
        r1,r2,r3 = st.columns(3)

        with r1:
            fig,ax = dark_fig()
            ax.pie([high_r,med_r,low_r],labels=['High','Medium','Low'],
                colors=[RED,ORANGE,GREEN],autopct='%1.1f%%',pctdistance=0.78,
                wedgeprops=dict(width=0.55,edgecolor=BG,linewidth=2),startangle=90)
            ax.set_title("Risk Distribution",color="#c8d8f0",fontsize=10,pad=8)
            ax.axis('equal'); ax.grid(False)
            st.pyplot(fig,use_container_width=True); plt.close()

        with r2:
            fig,ax = dark_fig()
            ax.hist(df_all['age'].dropna(),bins=20,color=CYAN,alpha=0.85,edgecolor=BG)
            ax.set_title("Age Distribution",color="#c8d8f0",fontsize=10,pad=8)
            ax.set_xlabel("Age",fontsize=8); ax.set_ylabel("Count",fontsize=8)
            st.pyplot(fig,use_container_width=True); plt.close()

        with r3:
            df_all['gl'] = df_all['gender'].astype(str).map({'1':'Male','0':'Female'}).fillna('Unknown')
            g = df_all['gl'].value_counts()
            fig,ax = dark_fig()
            bars = ax.bar(g.index,g.values,color=[PURPLE,CYAN],width=0.5,edgecolor=BG)
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2,b.get_height()+5,str(int(b.get_height())),
                    ha='center',color=TEXT,fontsize=8)
            ax.set_title("Gender Distribution",color="#c8d8f0",fontsize=10,pad=8)
            st.pyplot(fig,use_container_width=True); plt.close()

        r4,r5 = st.columns(2)
        with r4:
            fig,ax = dark_fig(5,3)
            risk_order = ['High','Medium','Low']
            colors_r   = [RED,ORANGE,GREEN]
            means = [df_all[df_all['risk_level']==r]['icu_los'].mean() for r in risk_order]
            bars = ax.bar(risk_order,means,color=colors_r,width=0.5,edgecolor=BG)
            for b,v in zip(bars,means):
                ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f"{v:.2f}",
                    ha='center',color=TEXT,fontsize=8)
            ax.set_title("Avg ICU Stay by Risk",color="#c8d8f0",fontsize=10,pad=8)
            ax.set_ylabel("Days",fontsize=8)
            st.pyplot(fig,use_container_width=True); plt.close()

        with r5:
            fig,ax = dark_fig(5,3)
            cmap = {'High':RED,'Medium':ORANGE,'Low':GREEN}
            samp = df_all.sample(min(400,len(df_all)),random_state=42)
            for level,color in cmap.items():
                sub = samp[samp['risk_level']==level]
                ax.scatter(sub['age'],sub['probability'],c=color,alpha=0.45,s=12,label=level)
            ax.set_title("Age vs Probability",color="#c8d8f0",fontsize=10,pad=8)
            ax.set_xlabel("Age",fontsize=8); ax.set_ylabel("Probability",fontsize=8)
            ax.legend(fontsize=8,framealpha=0,labelcolor=[RED,ORANGE,GREEN])
            st.pyplot(fig,use_container_width=True); plt.close()

        st.markdown('<div class="section-title">🕒 Recent Predictions</div>', unsafe_allow_html=True)
        recent = df_all.sort_values('created_at',ascending=False).head(10).copy()
        recent['gender'] = recent['gender'].astype(str).map({'1':'Male','0':'Female'})
        recent['probability'] = recent['probability'].round(4)
        recent['age'] = recent['age'].round(1)
        st.dataframe(recent[['id','age','gender','length_of_stay','icu_los',
            'num_diagnoses','num_medications','probability','risk_level']],
            use_container_width=True,hide_index=True)

# ══════════════════════════════════════════
# TAB 2 — AI ASSISTANT (Pure LLM)
# ══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🤖 Ask the AI Agent</div>', unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono;font-size:0.8rem;color:#6b7fa3'>The AI reads your question, understands it, generates SQL, and returns the answer.</p>", unsafe_allow_html=True)

    question = st.text_input("", placeholder="e.g. How many patients have ICU stay > 7 days and are high risk?",
        label_visibility="collapsed", key="ai_q")

    if st.button("Ask Agent →", type="primary"):
        if not question.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("🤖 Agent is thinking... (10–15 seconds)"):
                try:
                    answer, df_res, sql = ask_agent(question)

                    if answer:
                        answer_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', answer)
                        st.markdown(f'<div class="answer-box">{answer_html}</div>', unsafe_allow_html=True)
                    else:
                        disp = df_res.copy()
                        if 'gender' in disp.columns:
                            disp['gender'] = disp['gender'].astype(str).map(
                                {'1':'Male','0':'Female','1.0':'Male','0.0':'Female'}
                            ).fillna(disp['gender'])
                        if 'probability' in disp.columns:
                            disp['probability'] = disp['probability'].round(4)
                        if 'age' in disp.columns:
                            disp['age'] = disp['age'].round(1)
                        st.dataframe(disp, use_container_width=True, hide_index=True)

                    with st.expander("🔍 Generated SQL"):
                        st.code(sql, language="sql")

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.caption("Try rephrasing your question.")

    # Sample questions
    st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono;font-size:0.68rem;color:#6b7fa3;text-transform:uppercase;letter-spacing:1px'>Sample questions to try</p>", unsafe_allow_html=True)
    chips = [
        "How many patients have more than 5 diagnoses and are high risk?",
        "What is the average ICU stay for high risk patients?",
        "Give me top 10 patients with age greater than 60",
        "How many male patients are high risk?",
        "Show patients with more than 15 medications and high risk",
        "What is the average age of female high risk patients?",
    ]
    cols = st.columns(3)
    for i, chip in enumerate(chips):
        with cols[i % 3]:
            st.markdown(f"""<div style='background:#0f1829;border:1px solid #1e3a5f;
                border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.5rem;
                font-family:DM Mono,monospace;font-size:0.72rem;color:#8899bb;
                line-height:1.4'>💬 {chip}</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 3 — LIVE PREDICTION
# ══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🔬 Live Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono;font-size:0.8rem;color:#6b7fa3'>Enter patient details to get real-time readmission risk from XGBoost via FastAPI.</p>", unsafe_allow_html=True)

    with st.form("pred_form"):
        pc1,pc2 = st.columns(2)
        with pc1:
            p_age    = st.number_input("Age",                   min_value=1,   max_value=120, value=65)
            p_icu    = st.number_input("ICU Stay (days)",        min_value=0.0, max_value=30.0,value=3.0,step=0.5)
            p_diag   = st.number_input("Number of Diagnoses",    min_value=1,   max_value=20,  value=5)
        with pc2:
            p_gender = st.selectbox("Gender", ["Male (1)","Female (0)"])
            p_los    = st.number_input("Length of Stay (days)",  min_value=0.0, max_value=60.0,value=7.0,step=0.5)
            p_meds   = st.number_input("Number of Medications",  min_value=1,   max_value=50,  value=10)
        submitted = st.form_submit_button("Predict Risk →", type="primary")

    if submitted:
        gender_val = 1 if "Male" in p_gender else 0
        payload = {"age":p_age,"gender":gender_val,"length_of_stay":p_los,
                   "icu_los":p_icu,"num_diagnoses":p_diag,"num_medications":p_meds}
        with st.spinner("Calling FastAPI..."):
            try:
                resp = requests.post(API_URL, params=payload, timeout=10)
                if resp.status_code == 200:
                    result = resp.json()
                    risk   = result.get("risk_level","Unknown")
                    prob   = round(result.get("readmission_probability",0),4)
                    color  = {"High":RED,"Low":GREEN,"Medium":ORANGE}.get(risk,"#888")
                    border = {"High":"#ff4d6d33","Low":"#00e5a033","Medium":"#ffb34733"}.get(risk,"#88888833")
                    _,mid,_ = st.columns([1,2,1])
                    with mid:
                        st.markdown(f"""<div style='background:#0f1829;border:1px solid {border};
                            border-left:4px solid {color};border-radius:12px;
                            padding:1.5rem;text-align:center;margin-top:1rem'>
                            <p style='font-family:DM Mono;font-size:0.7rem;color:#6b7fa3;
                            text-transform:uppercase;letter-spacing:2px;margin:0 0 0.5rem 0'>Prediction Result</p>
                            <p style='font-size:2rem;font-weight:800;color:{color};margin:0.3rem 0'>{risk} Risk</p>
                            <p style='font-family:DM Mono;font-size:1rem;color:#8899bb;margin:0'>
                            Readmission Probability: {prob}</p>
                        </div>""", unsafe_allow_html=True)
                    with st.expander("📋 Request / Response"):
                        st.json({"request":payload,"response":result})
                else:
                    st.error(f"API error {resp.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("FastAPI not running. Start it: `uvicorn main:app --reload` in /api folder.")
            except Exception as e:
                st.error(f"Error: {e}")

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("<p style='font-family:Syne;font-weight:800;font-size:1.1rem;color:#00d4ff;margin:1rem 0 0.2rem 0'>🏥 Healthcare AI</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono;font-size:0.7rem;color:#6b7fa3;margin:0 0 1rem 0'>Readmission Risk System</p>", unsafe_allow_html=True)

    if data_loaded:
        st.markdown("---")
        st.metric("Total Patients",  f"{total:,}")
        st.metric("High Risk",       f"{high_r:,}", f"{pct_high}%")
        st.metric("Low Risk",        f"{low_r:,}")
        st.metric("Avg Age",         f"{avg_age} yrs")
        st.metric("Avg Stay",        f"{avg_stay} days")
        st.metric("Avg ICU",         f"{avg_icu} days")
        st.metric("Avg Probability", f"{avg_prob}")

    st.markdown("---")
    st.markdown("""
    <p style='font-family:DM Mono;font-size:0.68rem;color:#6b7fa3;text-transform:uppercase;letter-spacing:1px'>Services</p>
    <p style='font-family:DM Mono;font-size:0.75rem;margin:0.3rem 0'><span style='color:#00e5a0'>●</span> PostgreSQL :8080</p>
    <p style='font-family:DM Mono;font-size:0.75rem;margin:0.3rem 0'><span style='color:#00e5a0'>●</span> FastAPI :8000</p>
    <p style='font-family:DM Mono;font-size:0.75rem;margin:0.3rem 0'><span style='color:#00e5a0'>●</span> Ollama phi3</p>
    <p style='font-family:DM Mono;font-size:0.75rem;margin:0.3rem 0'><span style='color:#00e5a0'>●</span> Streamlit :8501</p>
    <p style='font-family:DM Mono;font-size:0.7rem;color:#6b7fa3;margin-top:0.5rem'>🤖 Pure LLM Agent</p>
    """, unsafe_allow_html=True)