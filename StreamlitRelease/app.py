import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#  helper function to get a random balanced sample from each class
def get_random_balanced_sample(df, n_per_class=5, random_state=42):
    return (
        df.groupby("risk_class", group_keys=False)
          .apply(lambda x: x.sample(n=min(n_per_class, len(x)), random_state=random_state))
          .reset_index(drop=True)
    )

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

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
This project demonstrates synthetic data generation, exploratory data analysis,
model training, and evaluation using an SVM model.
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

if generate_button:
    st.session_state.df = generate_data(samples_per_class)
    st.session_state.model_trained = False

df = st.session_state.df

if generate_button:
    df = generate_data(samples_per_class)
    st.session_state.df = df

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

    st.session_state.svm = svm
    st.session_state.scaler = scaler
    st.session_state.X_test = X_test
    st.session_state.X_test_s = X_test_s
    st.session_state.y_test = y_test
    st.session_state.le = le
    st.session_state.model_trained = True


#
#   Main Tabs
#


tab1, tab2 = st.tabs([
    "Data Generation & Exploratory Data Analysis",
    "Modeling & Evaluation"
])


# 
#   Tab 1 – Data Generation & Exploratory Data Analysis
#


if df is None:
    st.info("Click **Generate & Train Model** in the sidebar to begin.")
    st.stop()


with tab1:
    st.header("Synthetic Dataset Overview")
    st.markdown(
        """
        The dataset represents financial applicants grouped into three credit risk
        categories. Each class was generated using custom parameters
        (mean and standard deviation) to simulate realistic financial behavior.
        Features were standardized using **StandardScaler**
            (mean = 0, standard deviation = 1).
        """
    )

    st.write("**Dataset shape:**", df.shape)

    raw_sample = get_random_balanced_sample(df, n_per_class=5)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Raw Data Sample (Randomized)")
        st.dataframe(raw_sample)

        csv_data = convert_df_to_csv(df)
        st.download_button(
            label="Download Raw Dataset",
            data=csv_data,
            file_name="synthetic_credit_risk_dataset.csv",
            mime="text/csv"
        )

    with col2:
        st.subheader("Scaled Data Sample")

        if "scaler" not in st.session_state:
            st.warning("Train the model to view scaled data.")
        else:
            scaled_values = st.session_state.scaler.transform(
                raw_sample[["income", "age", "credit_score", "debt_ratio", "loan_amount"]]
            )

            scaled_df = pd.DataFrame(
                scaled_values,
                columns=["income", "age", "credit_score", "debt_ratio", "loan_amount"]
            )

            scaled_df["risk_class"] = raw_sample["risk_class"].values

            st.dataframe(scaled_df)

        if "scaler" in st.session_state:
            scaled = st.session_state.scaler.transform(
                df[["income","age","credit_score","debt_ratio","loan_amount"]]
            )

            scaled_df = pd.DataFrame(
                scaled,
                columns=["income","age","credit_score","debt_ratio","loan_amount"]
            )
            scaled_df["risk_class"] = df["risk_class"].values

            st.download_button(
                label="Download Scaled Dataset",
                data=convert_df_to_csv(scaled_df),
                file_name="scaled_credit_risk_dataset.csv",
                mime="text/csv"
            )


#
#   Exploratory Data Analysis
#

    st.header("Exploratory Data Analysis (EDA)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Income Distribution by Risk Class")
        st.markdown("""
            Low-risk applicants tend to have higher incomes, while high-risk
            applicants generally have lower incomes.
            """)
        fig, ax = plt.subplots()
        sns.histplot(df, x="income", hue="risk_class", kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Debt Ratio by Risk Class")
        st.markdown("""
            Debt ratio indicates the proportion of an applicant's income that
            goes towards debt payments. Higher debt ratios are often associated
            with higher risk classes.
            """)
        fig, ax = plt.subplots()
        sns.boxplot(df, x="risk_class", y="debt_ratio", ax=ax)
        st.pyplot(fig)

    st.subheader("Feature Correlation Matrix")
    st.markdown("""The correlation matrix shows the relationships between different features.""")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df.drop(columns="risk_class").corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Feature Visualization")

    plot_type = st.radio(
        "Select plot type",
        ["2D Plot", "3D Plot"],
        horizontal=True
    )

    features = ["income", "age", "credit_score", "debt_ratio", "loan_amount"]

    x_feat = st.selectbox("Select X-axis feature", features, index=0)
    y_feat = st.selectbox("Select Y-axis feature", features, index=1)

    if plot_type == "2D Plot":
        fig = px.scatter(
            df,
            x=x_feat,
            y=y_feat,
            color="risk_class",
            opacity=0.7,
            title=f"2D Feature Visualization: {x_feat} vs {y_feat}",
            labels={
                x_feat: x_feat,
                y_feat: y_feat,
                "risk_class": "Risk Class"
            }
        )

        fig.update_layout(
            height=500,
            legend_title_text="Risk Class"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        z_feat = st.selectbox("Select Z-axis feature", features, index=2)

        fig = px.scatter_3d(
            df,
            x=x_feat,
            y=y_feat,
            z=z_feat,
            color="risk_class",
            opacity=0.7,
            title=f"3D Feature Visualization: {x_feat}, {y_feat}, {z_feat}",
            labels={
                x_feat: x_feat,
                y_feat: y_feat,
                z_feat: z_feat,
                "risk_class": "Risk Class"
            }
        )

        fig.update_layout(
            height=600,
            legend_title_text="Risk Class"
        )

        st.plotly_chart(fig, use_container_width=True)


# 
#   Tab 2 – Modeling & Evaluation
# 


if df is None:
    st.info("Click **Generate & Train Model** in the sidebar to begin.")
    st.stop()


with tab2:
    st.header("SVM Model Training")

    if not st.session_state.model_trained:
        st.info("Generate the dataset first.")
        st.stop()

    st.success("SVM model trained successfully.")
    st.write("Train/Test Split: ", f"{int((1-test_size)*100)}% / {int(test_size*100)}%")



    st.header("Model Evaluation and Analysis")

    if not st.session_state.model_trained:
        st.info("Train the model to view evaluation results.")
        st.stop()

    svm = st.session_state.svm
    X_test_s = st.session_state.X_test_s
    y_test = st.session_state.y_test
    le = st.session_state.le
    X_test = st.session_state.X_test

    y_pred = svm.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    st.metric("Accuracy", f"{acc:.3f}")

    st.subheader("Classification Report")
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)

    st.subheader("Confusion Matrix")
    st.markdown("Confusion matrices show the model's prediction performance across different classes.")
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
    st.markdown(" The closer the values of the predicted class means are to the \
                true class means, indicates that the model is performing better.")

    df_eval = X_test.copy()
    df_eval["true_class"] = le.inverse_transform(y_test)
    df_eval["predicted_class"] = le.inverse_transform(y_pred)

    true_means = (
    df_eval
    .groupby("true_class")[["income","age","credit_score","debt_ratio","loan_amount"]]
    .mean()
    )

    pred_means = (
        df_eval
        .groupby("predicted_class")[["income","age","credit_score","debt_ratio","loan_amount"]]
        .mean()
    )

    st.subheader("True Class Mean Features")
    st.dataframe(true_means)

    st.subheader("Predicted Class Mean Features")
    st.dataframe(pred_means)

    comparison = (pred_means - true_means) / true_means
    st.subheader("Relative Error (Predicted vs True Means)")
    st.markdown("""
        **Interpretation:**

        - < 10% = best performance
        - 10–25% = acceptable
        - \> 25% = model confusion
        """)
    st.dataframe(comparison.style.format("{:.2%}"))
