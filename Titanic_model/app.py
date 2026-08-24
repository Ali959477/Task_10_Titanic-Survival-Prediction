import os
import joblib
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model.joblib"
    )

    if not os.path.exists(model_path):
        st.error("❌ model.joblib not found!")
        st.info(
            "Make sure model.joblib is inside the same folder as app.py."
        )
        st.stop()

    try:
        return joblib.load(model_path)

    except Exception as e:
        st.error("❌ Unable to load model.joblib")
        st.code(str(e))
        st.stop()


model = load_model()

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 1rem;
}

.hero {
    text-align: center;
    padding: 10px 0 25px 0;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 18px;
    color: #777;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 15px;
}

.info-card {
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.08);
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.08);
    text-align: center;
}

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 700;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 14px;
    padding: 14px;
    background: rgba(128,128,128,0.06);
}

.footer {
    text-align: center;
    color: #777;
    font-size: 14px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🚢 Titanic Survival Predictor
</div>

<div class="hero-subtitle">
Machine Learning • Scikit-learn • Streamlit
</div>

</div>
""", unsafe_allow_html=True)

st.info(
    "👋 Enter the passenger details below and click "
    "**Predict Survival**."
)

# =========================================================
# PASSENGER INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Passenger Information</div>',
    unsafe_allow_html=True
)

# =========================================================
# FIRST ROW
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3],
        index=2,
        help="1 = First Class, 2 = Second Class, 3 = Third Class"
    )

with col2:

    sex = st.selectbox(
        "Sex",
        ["female", "male"]
    )

with col3:

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )

with col4:

    fare = st.number_input(
        "Ticket Fare",
        min_value=0.0,
        value=32.0,
        step=1.0
    )

# =========================================================
# SECOND ROW
# =========================================================

col5, col6, col7 = st.columns(3)

with col5:

    sibsp = st.number_input(
        "Siblings / Spouses Aboard",
        min_value=0,
        max_value=8,
        value=0,
        step=1
    )

with col6:

    parch = st.number_input(
        "Parents / Children Aboard",
        min_value=0,
        max_value=6,
        value=0,
        step=1
    )

with col7:

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"],
        format_func=lambda x: {
            "S": "Southampton",
            "C": "Cherbourg",
            "Q": "Queenstown"
        }[x]
    )

st.divider()

# =========================================================
# SUMMARY + PREDICTION
# =========================================================

left, right = st.columns(
    [1, 1],
    gap="large"
)

# =========================================================
# PASSENGER SUMMARY
# =========================================================

with left:

    st.markdown(
        '<div class="section-title">📋 Passenger Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-card">',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "Class",
            pclass
        )

    with b:
        st.metric(
            "Age",
            f"{age:.0f}"
        )

    with c:
        st.metric(
            "Ticket Fare",
            f"${fare:.2f}"
        )

    st.write("")

    d, e, f = st.columns(3)

    with d:
        st.metric(
            "Sex",
            sex.title()
        )

    with e:
        st.metric(
            "Siblings",
            sibsp
        )

    with f:
        st.metric(
            "Parents/Children",
            parch
        )

    st.write("")

    family_members = sibsp + parch

    if family_members == 0:
        family_status = "Travelling Alone"
    else:
        family_status = f"{family_members} family member(s)"

    st.info(
        f"👨‍👩‍👧 **Family Status:** {family_status}"
    )

    port_names = {
        "S": "Southampton",
        "C": "Cherbourg",
        "Q": "Queenstown"
    }

    st.info(
        f"⚓ **Embarkation:** {port_names[embarked]}"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =========================================================
# PREDICTION
# =========================================================

with right:

    st.markdown(
        '<div class="section-title">🎯 Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="result-card">',
        unsafe_allow_html=True
    )

    st.write(
        "The trained machine-learning model will analyze "
        "the passenger information."
    )

    st.write("")

    predict = st.button(
        "🔮 Predict Survival",
        use_container_width=True,
        type="primary"
    )

    if predict:

        # =================================================
        # CREATE INPUT DATA
        # =================================================

        input_data = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked
        }])

        # =================================================
        # MODEL PREDICTION
        # =================================================

        try:

            prediction = int(
                model.predict(input_data)[0]
            )

            # =================================================
            # SURVIVAL PROBABILITY
            # =================================================

            probability = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                probability = float(
                    probabilities[1]
                )

            st.write("")

            # =================================================
            # PREDICTION RESULT
            # =================================================

            if prediction == 1:

                st.success(
                    "🟢 Passenger likely survived"
                )

                st.markdown(
                    """
                    <h3 style="text-align:center;">
                    Survival Prediction: YES
                    </h3>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.error(
                    "🔴 Passenger likely did not survive"
                )

                st.markdown(
                    """
                    <h3 style="text-align:center;">
                    Survival Prediction: NO
                    </h3>
                    """,
                    unsafe_allow_html=True
                )

            # =================================================
            # PROBABILITY DISPLAY
            # =================================================

            if probability is not None:

                st.write("")

                st.metric(
                    "Estimated Survival Probability",
                    f"{probability:.1%}"
                )

                st.progress(
                    probability
                )

                if probability >= 0.75:

                    st.success(
                        "📈 High estimated survival probability"
                    )

                elif probability >= 0.50:

                    st.info(
                        "📊 Moderate estimated survival probability"
                    )

                else:

                    st.warning(
                        "📉 Low estimated survival probability"
                    )

            # =================================================
            # INPUT DETAILS AFTER PREDICTION
            # =================================================

            st.write("")

            st.markdown(
                "### 🔎 Prediction Details"
            )

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:

                st.metric(
                    "Passenger Class",
                    pclass
                )

            with result_col2:

                st.metric(
                    "Age",
                    f"{age:.0f}"
                )

            with result_col3:

                st.metric(
                    "Sex",
                    sex.title()
                )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.warning(
                "The model features do not match the "
                "application input features."
            )

            st.code(str(e))

    else:

        st.info(
            "👆 Click **Predict Survival** to see the result."
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown("""
<div class="footer">

🚢 Titanic Survival Predictor
&nbsp; • &nbsp;
Python
&nbsp; • &nbsp;
Scikit-learn
&nbsp; • &nbsp;
Streamlit
&nbsp; • &nbsp;
Joblib

</div>
""", unsafe_allow_html=True)