import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#
#   Main page -- Synthetic Credit Risk Modeling
#


st.set_page_config(page_title="Synthetic Credit Risk Modeling", layout="wide")
st.title("Synthetic Credit Risk Modeling (Option 1)")

st.markdown("""
This app demonstrates synthetic data generation, exploratory data analysis,
model training, and evaluation using an SVM classifier.
""")


# 
#   Synthetic Data Generation
# 


@st.cache_data
def generate_data(n_per_class=2000):
    np.random.seed(42)

    params = {
        "LowRisk":  [(80000,12000),(45,8),(720,30),(0.18,0.05),(9000,2000)],
        "MediumRisk":[(55000,10000),(40,10),(650,40),(0.32,0.08),(13000,3000)],
        "HighRisk": [(30000,9000),(35,12),(580,50),(0.60,0.10),(18000,4000)]
    }

    rows = []
    for cls, vals in params.items():
        for _ in range(n_per_class):
            rows.append([
                np.random.normal(*vals[0]),
                np.random.normal(*vals[1]),
                np.random.normal(*vals[2]),
                np.random.normal(*vals[3]),
                np.random.normal(*vals[4]),
                cls
            ])

    df = pd.DataFrame(rows, columns=[
        "income","age","credit_score","debt_ratio","loan_amount","risk_class"
    ])
    return df

df = generate_data()


# 
#   Sidebar menu
# 


st.sidebar.header("Controls")
show_data = st.sidebar.checkbox("Show raw data")
show_eda = st.sidebar.checkbox("Show EDA", True)
train_model = st.sidebar.checkbox("Train SVM model", True)

if show_data:
    st.subheader("Raw Synthetic Dataset")
    st.dataframe(df.head(50))


# 
#   EDA Section
#


if show_eda:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Income Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df, x="income", hue="risk_class", kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("### Debt Ratio by Risk Class")
        fig, ax = plt.subplots()
        sns.boxplot(df, x="risk_class", y="debt_ratio", ax=ax)
        st.pyplot(fig)

    st.markdown("### Feature Correlation Matrix")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.drop(columns="risk_class").corr(),
                annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# 
#   Model Training & Evaluation
# 


if train_model:
    st.subheader("SVM Model Training & Evaluation")

    features = ["income","age","credit_score","debt_ratio","loan_amount"]
    X = df[features]
    y = df["risk_class"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    svm = SVC(kernel="rbf")
    svm.fit(X_train_s, y_train)
    y_pred = svm.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)

    st.markdown(f"### Accuracy: **{acc:.3f}**")

    st.markdown("### Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    st.pyplot(fig)

    st.markdown("### Mean Feature Values per Predicted Class")
    df_eval = X_test.copy()
    df_eval["predicted"] = y_pred
    st.dataframe(df_eval.groupby("predicted").mean())
