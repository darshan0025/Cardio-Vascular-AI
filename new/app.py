import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ============================================================
#                      PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Cardio Risk Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
#                      CUSTOM CSS STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Modern Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Animations */
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2); }
        50% { transform: scale(1.02); box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.3); }
        100% { transform: scale(1); box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2); }
    }

    /* Card Styling */
    .stCard {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(255, 255, 255, 0.5);
        animation: slideUp 0.6s ease-out;
    }
    
    /* Input Container Styling */
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
        background: rgba(255, 255, 255, 0.6);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 10px;
        animation: fadeIn 0.8s ease-out;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        animation: fadeIn 0.5s ease;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: rgba(255,255,255,0.8);
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #fff;
        color: #4f46e5;
        border-top: 3px solid #4f46e5;
        transform: translateY(-2px);
    }

    /* Custom Headers */
    .header-title {
        background: linear-gradient(120deg, #2563eb, #1e40af);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        text-align: center;
        padding: 20px 0;
        animation: slideUp 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .section-header {
        color: #1e293b;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 15px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 5px;
    }

    /* Risk Level Badges */
    .risk-badge {
        padding: 25px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        animation: pulse 2s infinite ease-in-out;
    }
    .risk-low { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .risk-moderate { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
    .risk-high { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%); }

    /* Button */
    div.stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.2rem;
        width: 100%;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
        transition: all 0.3s ease;
        animation: fadeIn 1s ease-out;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.4);
    }
</style>

""", unsafe_allow_html=True)


# ============================================================
#                      LOAD MODEL
# ============================================================
@st.cache_resource
def load_prediction_model():
    return joblib.load("cardio_pipeline.pkl")

try:
    model = load_prediction_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# ============================================================
#                      HEADER
# ============================================================
st.markdown("<h1 class='header-title'>🫀 CardioGuard AI</h1>", unsafe_allow_html=True)


# ============================================================
#                      MAIN TABS
# ============================================================
# Creating the requested 3 tabs layout
main_tab, info_tab, disclaim_tab = st.tabs(["🏥 Risk Assessment", "ℹ️ System Info", "⚠️ Disclaimer"])


# ============================================================
#                      TAB 1: RISK ASSESSMENT
# ============================================================
with main_tab:
    # ------------------ INPUT SECTION ------------------
    # Using a container to group inputs visually
    with st.container():
        
        # 3 Column Layout for Inputs (Single Page View)
        col_basic, col_vitals, col_life = st.columns(3, gap="large")
        
        # --- Column 1: Basic Data ---
        with col_basic:
            st.markdown("<div class='section-header'>👤 Basic Demographics</div>", unsafe_allow_html=True)
            age_years = st.number_input("Age (years)", 18, 100, 45)
            gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
            height = st.number_input("Height (cm)", 120, 220, 165)
            weight = st.number_input("Weight (kg)", 30, 200, 70)
            
            # Live BMI Preview
            current_bmi = weight / ((height / 100) ** 2)
            st.caption(f"Calculated BMI: **{current_bmi:.1f}**")

        # --- Column 2: Vitals ---
        with col_vitals:
            st.markdown("<div class='section-header'>🫀 Clinical Vitals</div>", unsafe_allow_html=True)
            ap_hi = st.number_input("Systolic BP (mmHg)", 80, 220, 120)
            ap_lo = st.number_input("Diastolic BP (mmHg)", 40, 160, 80)
            cholesterol = st.selectbox("Cholesterol", [1, 2, 3], 
                                     format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])
            gluc = st.selectbox("Glucose", [1, 2, 3], 
                              format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])

        # --- Column 3: Lifestyle ---
        with col_life:
            st.markdown("<div class='section-header'>🏃 Lifestyle Factors</div>", unsafe_allow_html=True)
            smoke = st.selectbox("Smoker", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            alco = st.selectbox("Alcohol Consumption", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            active = st.selectbox("Physical Activity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            st.markdown("<br>", unsafe_allow_html=True) # Spacer
            predict_btn = st.button("🔍 Run Assessment", use_container_width=True)

    # ------------------ PREDICTION LOGIC ------------------
    if predict_btn:
        st.markdown("---")
        
        # Feature Engineering
        bmi = weight / ((height / 100) ** 2)
        bp_diff = ap_hi - ap_lo
        
        if bmi < 18.5: bmi_cat = 0
        elif bmi < 25: bmi_cat = 1
        elif bmi < 30: bmi_cat = 2
        else: bmi_cat = 3
        
        input_data = pd.DataFrame([{
            'age_years': age_years, 'height': height, 'weight': weight,
            'ap_hi': ap_hi, 'ap_lo': ap_lo, 'bmi': bmi, 'bp_diff': bp_diff,
            'gender': gender, 'cholesterol': cholesterol, 'gluc': gluc,
            'smoke': smoke, 'active': active, 'alco': alco, 'bmi_cat': bmi_cat
        }])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        confidence = max(model.predict_proba(input_data)[0]) * 100

        # Styles
        if probability < 0.35:
            risk_label = "Low Risk"
            risk_css = "risk-low"
            emoji = "🛡️"
            msg = "Great job! Maintain your healthy lifestyle."
        elif probability < 0.65:
            risk_label = "Moderate Risk"
            risk_css = "risk-moderate"
            emoji = "⚠️"
            msg = "Warning: Consider lifestyle improvements."
        else:
            risk_label = "High Risk"
            risk_css = "risk-high"
            emoji = "🚨"
            msg = "Action Required: Seek medical advice."

        # Result Display with Animation
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            st.markdown(f"""
                <div class="risk-badge {risk_css}">
                    <h1 style="color:white; margin:0; font-size: 3rem;">{emoji}</h1>
                    <h2 style="color:white; margin:10px 0;">{risk_label}</h2>
                    <h3 style="color:rgba(255,255,255,0.9);">{probability*100:.1f}%</h3>
                    <p style="color:rgba(255,255,255,0.8);">{msg}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown("### Analysis")
            # Radial Chart using Plotly
            categories = ['Age', 'BMI', 'BP', 'Cholesterol', 'Glucose']
            values = [
                min(age_years/80, 1),
                min(bmi/40, 1),
                min(ap_hi/180, 1),
                cholesterol/3,
                gluc/3
            ]

            fig = px.line_polar(r=values, theta=categories, line_close=True, range_r=[0,1])
            fig.update_traces(fill='toself', line_color='#4f46e5')
            fig.update_layout(
                margin=dict(t=20, b=20, l=40, r=40),
                height=300,
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("BMI Status", f"{bmi:.1f}", "Normal" if 18.5 <= bmi < 25 else "Attention", delta_color="inverse")
        m2.metric("BP Status", f"{ap_hi}/{ap_lo}", "Elevated" if ap_hi > 120 else "Normal", delta_color="inverse")
        m3.metric("Pulse Pressure", f"{bp_diff}", "Wide" if bp_diff > 60 else "Normal")
        m4.metric("AI Confidence", f"{confidence:.0f}%")


# ============================================================
#                      TAB 2: SYSTEM INFO
# ============================================================
with info_tab:
    st.markdown("### 🤖 System Architecture")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.info("""
        **Model Configuration:**
        - **Algorithm**: Random Forest Classifier (Optimized)
        - **Training Data**: Cardiovascular Disease Dataset (70k records)
        - **Optimization Goal**: Recall (Sensitivity) - To minimize false negatives.
        
        This system uses a machine learning pipeine that preprocesses user inputs (BMI calculation, categorical encoding) before passing them to the prediction model.
        """)
        
        st.code("""
# Pipeline Structure
1. Preprocessing:
   - BMI Calculation
   - Blood Pressure Differential
   - Categorical Encoding
2. Model:
   - Random Forest
   - Max Depth: Optimized
        """, language="python")
        
    with c2:
        st.image("https://img.icons8.com/clouds/500/server.png", caption="ML Backend")


# ============================================================
#                      TAB 3: DISCLAIMER
# ============================================================
with disclaim_tab:
    st.markdown("### ⚠️ Important Medical Disclaimer")
    
    st.error("""
    **FOR EDUCATIONAL PURPOSES ONLY**
    
    This application is a demonstration of machine learning capabilities in healthcare. It is **NOT** a certified medical device and should **NOT** be used for self-diagnosis or clinical decision-making.
    """)
    
    st.warning("""
    **Limitations:**
    - The model presumes data quality similar to the training set.
    - It does not account for genetic factors, family history, or pre-existing conditions not listed.
    - False positives and false negatives are possible.
    """)
    
    st.markdown("""
    **Always consult with a qualified healthcare provider for:**
    - Cardiovascular risk assessment
    - Interpretation of blood pressure or lab results
    - Lifestyle modification plans
    """)

