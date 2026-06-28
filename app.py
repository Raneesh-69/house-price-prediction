import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.ensemble import RandomForestRegressor

from utils import (
    create_single_input,
    prepare_prediction_input,
    format_price,
    get_best_model,
    get_feature_names
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOAD CSS
# ==========================================

def load_css():
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# ==========================================
# LOAD MODEL
# ==========================================

model = pickle.load(open("model.pkl", "rb"))
metrics = pickle.load(open("model_metrics.pkl", "rb"))
y_true, y_pred = pickle.load(open("predictions.pkl", "rb"))

data = pd.read_csv("data.csv")

best_model = get_best_model(metrics)

# ==========================================
# SIDEBAR
# ===
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Prediction",
        "📊 Dashboard",
        "📈 Analytics",
        "🤖 Models",
        "📂 Batch Prediction",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="profile-card">

<h3 class="profile-name">
👨‍💻 Pitamber Raneesh Joga
</h3>

<p class="profile-role">
AI & Machine Learning Student
</p>

</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.success("⭐ Portfolio Project")

# ==========================================
# HEADER
# ==========================================



col1, col2 = st.columns([1, 8])

with col1:
    st.image("assets/logo.png", width=100)

with col2:
    st.title("House Price Prediction")
    st.caption("Predict House Prices using Machine Learning")
    st.markdown("---")

# =====================================================
# PREDICTION PAGE
# =====================================================

if page == "🏠 Prediction":

    left, right = st.columns([1.4, 1])

    # ======================================
    # LEFT PANEL
    # ======================================

    with left:

        with st.container(border=True):

            st.subheader("🏡 Enter House Details")

            area = st.number_input(
                "Area (sq ft)",
                min_value=500,
                max_value=20000,
                value=3000,
                step=100
            )

            c1, c2 = st.columns(2)

            with c1:

                bedrooms = st.number_input(
                    "Bedrooms",
                    1,
                    10,
                    3
                )

                bathrooms = st.number_input(
                    "Bathrooms",
                    1,
                    10,
                    2
                )

                stories = st.number_input(
                    "Stories",
                    1,
                    5,
                    2
                )

                parking = st.number_input(
                    "Parking",
                    0,
                    10,
                    2
                )

            with c2:

                mainroad = st.selectbox(
                    "Main Road",
                    ["Yes", "No"]
                )

                guestroom = st.selectbox(
                    "Guest Room",
                    ["Yes", "No"]
                )

                basement = st.selectbox(
                    "Basement",
                    ["Yes", "No"]
                )

                hotwater = st.selectbox(
                    "Hot Water Heating",
                    ["Yes", "No"]
                )

                air = st.selectbox(
                    "Air Conditioning",
                    ["Yes", "No"]
                )

                prefarea = st.selectbox(
                    "Preferred Area",
                    ["Yes", "No"]
                )

                furnish = st.selectbox(
                    "Furnishing",
                    [
                        "furnished",
                        "semi-furnished",
                        "unfurnished"
                    ]
                )

            predict = st.button(
                "🔍 Predict House Price",
                use_container_width=True
            )

    # ======================================
    # RIGHT PANEL
    # ======================================

    with right:

        with st.container(border=True):

            st.markdown("""
            <h2 style="color:#111827;">
            📊 Prediction Result
            </h2>
            """, unsafe_allow_html=True)

            if predict:

                input_df = create_single_input(

                    area,

                    bedrooms,

                    bathrooms,

                    stories,

                    1 if mainroad == "Yes" else 0,

                    1 if guestroom == "Yes" else 0,

                    1 if basement == "Yes" else 0,

                    1 if hotwater == "Yes" else 0,

                    1 if air == "Yes" else 0,

                    parking,

                    1 if prefarea == "Yes" else 0,

                    furnish

                )

                prediction = model.predict(input_df)[0]

                st.success("Prediction Successful")

                st.markdown("## 💰 Estimated Price")

                st.markdown(
                    f"# {format_price(prediction)}"
                )

                st.divider()

                m1, m2 = st.columns(2)

                with m1:

                    st.metric(
                        "Best Model",
                        best_model
                    )

                    st.metric(
                        "R² Score",
                        f"{metrics[best_model]['R2']:.3f}"
                    )

                with m2:

                    st.metric(
                        "RMSE",
                        f"{metrics[best_model]['RMSE']:.0f}"
                    )

                    st.metric(
                        "MAE",
                        f"{metrics[best_model]['MAE']:.0f}"
                    )

            else:

                st.info(
                    "👈 Enter the house details and click **Predict House Price**."
                )

    st.markdown("")

    st.markdown("### 🌟 Top Features Used")

    f1, f2, f3, f4 = st.columns(4)

    f1.info("📐 Area")

    f2.info("🛏 Bedrooms")

    f3.info("🚿 Bathrooms")

    f4.info("🚗 Parking")

# =====================================================
# DASHBOARD
# =====================================================

elif page == "📊 Dashboard":

    st.subheader("📊 Model Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "🏆 Best Model",
            best_model
        )

    with c2:
        st.metric(
            "R² Score",
            f"{metrics[best_model]['R2']:.3f}"
        )

    with c3:
        st.metric(
            "RMSE",
            f"{metrics[best_model]['RMSE']:.0f}"
        )

    with c4:
        st.metric(
            "MAE",
            f"{metrics[best_model]['MAE']:.0f}"
        )

    st.divider()

    left, right = st.columns(2)

    # -------------------------------
    # Actual vs Predicted
    # -------------------------------

    with left:

        st.markdown("### 📈 Actual vs Predicted")

        fig, ax = plt.subplots(figsize=(6,5))

        ax.scatter(
            y_true,
            y_pred,
            alpha=0.7,
            color="royalblue"
        )

        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title("Actual vs Predicted")

        st.pyplot(fig)

    # -------------------------------
    # Feature Importance
    # -------------------------------

    with right:

        if isinstance(model, RandomForestRegressor):

            st.markdown("### 🌳 Feature Importance")

            importance = model.feature_importances_

            names = get_feature_names()

            fig, ax = plt.subplots(figsize=(6,5))

            ax.barh(
                names,
                importance,
                color="steelblue"
            )

            ax.set_xlabel("Importance")

            st.pyplot(fig)

        else:

            st.info(
                "Feature importance is available only for Random Forest."
            )

# =====================================================
# ANALYTICS
# =====================================================

elif page == "📈 Analytics":

    st.subheader("📈 Exploratory Data Analysis")

    st.write(
        "Explore relationships between different house features."
    )

    # --------------------------
    # Dataset Preview
    # --------------------------

    with st.expander("📂 Dataset Preview"):

        st.dataframe(data.head())

    st.divider()

    # --------------------------
    # Price Distribution
    # --------------------------

    left, right = st.columns(2)

    with left:

        st.markdown("### 💰 Price Distribution")

        fig, ax = plt.subplots(figsize=(6,4))

        ax.hist(
            data["price"],
            bins=25,
            color="royalblue"
        )

        ax.set_xlabel("Price")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

    with right:

        st.markdown("### 📐 Area vs Price")

        fig, ax = plt.subplots(figsize=(6,4))

        ax.scatter(
            data["area"],
            data["price"],
            color="green",
            alpha=0.7
        )

        ax.set_xlabel("Area")
        ax.set_ylabel("Price")

        st.pyplot(fig)

    st.divider()

    # --------------------------
    # Bedrooms & Bathrooms
    # --------------------------

    left, right = st.columns(2)

    with left:

        st.markdown("### 🛏 Bedrooms vs Price")

        fig, ax = plt.subplots(figsize=(6,4))

        ax.scatter(
            data["bedrooms"],
            data["price"],
            color="orange"
        )

        ax.set_xlabel("Bedrooms")
        ax.set_ylabel("Price")

        st.pyplot(fig)

    with right:

        st.markdown("### 🚿 Bathrooms vs Price")

        fig, ax = plt.subplots(figsize=(6,4))

        ax.scatter(
            data["bathrooms"],
            data["price"],
            color="purple"
        )

        ax.set_xlabel("Bathrooms")
        ax.set_ylabel("Price")

        st.pyplot(fig)

    st.divider()

    # --------------------------
    # Correlation Heatmap
    # --------------------------

    st.markdown("### 🔥 Correlation Heatmap")

    temp = data.copy()

    yes_no = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea"
    ]

    for col in yes_no:

        temp[col] = temp[col].map({
            "yes":1,
            "no":0
        })

    temp = pd.get_dummies(
        temp,
        columns=["furnishingstatus"],
        drop_first=True
    )

    corr = temp.corr()

    fig, ax = plt.subplots(figsize=(10,8))

    image = ax.imshow(
        corr,
        cmap="Blues"
    )

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(
        corr.columns,
        rotation=90,
        fontsize=8
    )

    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(
        corr.columns,
        fontsize=8
    )

    plt.colorbar(image)

    st.pyplot(fig)    

# =====================================================
# MODELS
# =====================================================

elif page == "🤖 Models":

    st.subheader("🤖 Model Comparison")

    comparison = pd.DataFrame(metrics).T

    comparison = comparison.round(4)

    st.dataframe(
        comparison,
        use_container_width=True
    )

    best = get_best_model(metrics)

    st.success(f"🏆 Best Model : {best}")

    st.bar_chart(comparison["R2"])


# =====================================================
# BATCH PREDICTION
# =====================================================

elif page == "📂 Batch Prediction":

    st.subheader("📂 Batch Prediction")

    st.write(
        "Upload a CSV file to predict house prices for multiple houses."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.markdown("### Uploaded Data")

        st.dataframe(df.head())

        try:

            X = prepare_prediction_input(df)

            prediction = model.predict(X)

            result = df.copy()

            result["Predicted Price"] = prediction

            st.markdown("### Prediction Result")

            st.dataframe(result)

            csv = result.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Prediction CSV",
                data=csv,
                file_name="predicted_house_prices.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(e)


# =====================================================
# ABOUT
# =====================================================

elif page == "ℹ About":

    st.subheader("ℹ About Project")

    st.markdown("""

## 🏠 House Price Prediction

This project predicts house prices using **Machine Learning**.

### 🚀 Features

- House Price Prediction
- Dashboard
- Model Comparison
- Feature Importance
- EDA
- Batch Prediction
- Download Prediction CSV

---

### 🤖 Models Used

- Linear Regression

- Ridge Regression

- Random Forest

---

### 📚 Technologies

- Python

- Streamlit

- Scikit-Learn

- Pandas

- NumPy

- Matplotlib

---

#### 👨‍💻 Developed By

**Raneesh**

AI & Machine Learning Student

""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown("""
<style>

.footer{
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);

    color: #6B7280;

    font-size: 15px;

    z-index: 999;
}

.footer b{
    color:#2563EB;
}

</style>

<div class="footer">
    Made with ❤️ by <b> Raneesh</b>
</div>
""", unsafe_allow_html=True)