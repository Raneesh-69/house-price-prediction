import pandas as pd

# ==========================================
# Yes / No Columns
# ==========================================

YES_NO_COLUMNS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

# ==========================================
# Feature Order
# (Must match training data)
# ==========================================

FEATURE_COLUMNS = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "parking",
    "prefarea",
    "furnishingstatus_semi-furnished",
    "furnishingstatus_unfurnished"
]

# ==========================================
# Data Preprocessing
# ==========================================

def preprocess_data(df):

    df = df.copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Remove missing values
    df.dropna(inplace=True)

    # Convert yes/no columns
    for col in YES_NO_COLUMNS:

        if col in df.columns:

            df[col] = df[col].map({
                "yes": 1,
                "no": 0
            })

    # One Hot Encoding
    if "furnishingstatus" in df.columns:

        df = pd.get_dummies(
            df,
            columns=["furnishingstatus"],
            drop_first=True
        )

    return df

# ==========================================
# Prepare Uploaded CSV
# ==========================================

def prepare_prediction_input(df):

    df = preprocess_data(df)

    # Add missing columns
    for col in FEATURE_COLUMNS:

        if col not in df.columns:
            df[col] = 0

    # Arrange columns in correct order
    df = df[FEATURE_COLUMNS]

    return df

# ==========================================
# Create Single Prediction Input
# ==========================================

def create_single_input(

    area,
    bedrooms,
    bathrooms,
    stories,
    mainroad,
    guestroom,
    basement,
    hotwaterheating,
    airconditioning,
    parking,
    prefarea,
    furnishingstatus

):

    semi = 0
    unfurnished = 0

    if furnishingstatus == "semi-furnished":
        semi = 1

    elif furnishingstatus == "unfurnished":
        unfurnished = 1

    input_df = pd.DataFrame({

        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "mainroad": [mainroad],
        "guestroom": [guestroom],
        "basement": [basement],
        "hotwaterheating": [hotwaterheating],
        "airconditioning": [airconditioning],
        "parking": [parking],
        "prefarea": [prefarea],
        "furnishingstatus_semi-furnished": [semi],
        "furnishingstatus_unfurnished": [unfurnished]

    })

    return input_df

# ==========================================
# Indian Currency Formatter
# ==========================================

def format_price(price):

    return f"₹ {price:,.2f}"

# ==========================================
# Dashboard Metrics
# ==========================================

def get_best_model(metrics):

    best_model = max(
        metrics,
        key=lambda x: metrics[x]["R2"]
    )

    return best_model

# ==========================================
# Feature Importance
# ==========================================

def get_feature_names():

    return [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Stories",
        "Main Road",
        "Guest Room",
        "Basement",
        "Hot Water",
        "Air Conditioning",
        "Parking",
        "Preferred Area",
        "Semi Furnished",
        "Unfurnished"
    ]