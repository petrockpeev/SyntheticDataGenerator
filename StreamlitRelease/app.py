import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#   Streamlit session configuration
if "df" not in st.session_state:
    st.session_state.df = None

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False


#
#   Main page -- Synthetic Credit Risk Modeling
#


st.set_page_config(page_title="Synthetic Credit Risk Modeling", layout="wide")
st.title("Synthetic Credit Risk Modeling")

st.markdown("""
This app demonstrates synthetic data generation, exploratory data analysis,
model training, and evaluation using an SVM classifier.
""")


# 
#   Sidebar menu
# 


st.sidebar.header("Dataset Configuration")

samples_per_class = st.sidebar.slider(
    "Samples per risk class",
    500, 5000, 2000, step=500
)

test_size = st.sidebar.slider(
    "Test set size",
    0.1, 0.5, 0.3, step=0.05
)

generate_button = st.sidebar.button("Generate & Train Model")


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
#   Main Tabs
#


tab1, tab2, tab3, tab4 = st.tabs([
    "Data Generation",
    "Exploratory Data Analysis",
    "Modeling",
    "Evaluation"
])


# 
#   Tab 1 – Data Generation
#


with tab1:
    st.header("Synthetic Dataset Overview")
    st.markdown(
        """
        The dataset represents financial applicants grouped into three credit risk
        categories. Each class was generated using predefined statistical parameters
        (mean and standard deviation) to simulate realistic financial behavior.
        """
    )

    st.write("**Dataset shape:**", df.shape)
    st.dataframe(df.head(30))


#
#   Tab 2 – Exploratory Data Analysis
#


with tab2:
    st.header("Exploratory Data Analysis (EDA)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Income Distribution by Risk Class")
        fig, ax = plt.subplots()
        sns.histplot(df, x="income", hue="risk_class", kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Debt Ratio by Risk Class")
        fig, ax = plt.subplots()
        sns.boxplot(df, x="risk_class", y="debt_ratio", ax=ax)
        st.pyplot(fig)

    st.subheader("Feature Correlation Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df.drop(columns="risk_class").corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)


# 
#   Tab 3 – Modeling
# 

with tab3:
    st.header("SVM Model Training")

    features = ["income", "age", "credit_score", "debt_ratio", "loan_amount"]
    X = df[features]
    y = df["risk_class"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=test_size,
        random_state=42,
        stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    svm = SVC(kernel="rbf")
    svm.fit(X_train_s, y_train)

    st.success("SVM model trained successfully.")


#
#   Tab 4 – Evaluation
#


with tab4:
    st.header("Model Evaluation and Analysis")

    y_pred = svm.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    st.metric("Accuracy", f"{acc:.3f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Comparison with Known Synthetic Properties")
    st.markdown("""
    Table legend: 0 = HighRisk, 1 = LowRisk, 2 = MediumRisk
    """)

    df_eval = X_test.copy()
    df_eval["predicted"] = y_pred
    st.dataframe(df_eval.groupby("predicted").mean())
