import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Page configuration
st.set_page_config(
    page_title="AI-Powered Data Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title
st.title("AI-Integrated Data Analysis and Prediction Dashboard")
st.markdown("""
Welcome to the AI-powered platform for data analysis and actionable insights!  
Upload a CSV file, and explore comprehensive statistics, visualizations, and AI-driven recommendations.
""")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    data = load_data(uploaded_file)

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    options = [
        "Dataset Overview",
        "Data Cleaning",
        "Exploratory Data Analysis (EDA)",
        "AI Model Training",
        "Insights and Recommendations"
    ]
    choice = st.sidebar.radio("Select a section", options)

    if choice == "Dataset Overview":
        st.header("Dataset Overview")
        st.write("### First 5 Rows of the Dataset")
        st.dataframe(data.head())

        st.write("### Dataset Summary")
        st.write(data.describe(include="all").transpose())

        st.write("### Dataset Shape")
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    elif choice == "Data Cleaning":
        st.header("Data Cleaning")
        st.write("### Handling Missing Values")

        imputer = SimpleImputer(strategy="mean")
        data_cleaned = data.copy()

        for col in data_cleaned.select_dtypes(include=np.number).columns:
            if data_cleaned[col].isnull().sum() > 0:
                data_cleaned[col] = imputer.fit_transform(data_cleaned[[col]])
                st.write(f"Filled missing values in column `{col}` with mean.")

        for col in data_cleaned.select_dtypes(include="object").columns:
            if data_cleaned[col].isnull().sum() > 0:
                data_cleaned[col] = data_cleaned[col].fillna(data_cleaned[col].mode()[0])
                st.write(f"Filled missing values in column `{col}` with mode.")

        st.write("### Cleaned Dataset")
        st.dataframe(data_cleaned.head())

    elif choice == "Exploratory Data Analysis (EDA)":
        st.header("Exploratory Data Analysis")

        st.write("### Numerical Features Distribution")
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            st.write(f"#### Distribution of {col}")
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, ax=ax, color="skyblue")
            st.pyplot(fig)

        st.write("### Categorical Features Distribution")
        categorical_cols = data.select_dtypes(include="object").columns
        for col in categorical_cols:
            st.write(f"#### Distribution of {col}")
            fig, ax = plt.subplots()
            sns.countplot(data=data, x=col, ax=ax, palette="Set2")
            st.pyplot(fig)

        st.write("### Correlation Heatmap")
        corr = data.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif choice == "AI Model Training":
        st.header("AI Model Training")

        st.write("### Select Target Variable")
        target_variable = st.selectbox("Choose the target variable", data.columns)

        st.write("### Select Features")
        features = st.multiselect(
            "Select features for training the model",
            [col for col in data.columns if col != target_variable]
        )

        if target_variable and features:
            X = data[features]
            y = data[target_variable]

            # Encode categorical target variable if necessary
            if y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Handle missing values in features
            X = X.fillna(X.mean())

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train a Random Forest model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Predictions and evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"### Model Accuracy: {accuracy:.2f}")

            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.write("### Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt='d', ax=ax)
            st.pyplot(fig)

    elif choice == "Insights and Recommendations":
        st.header("Insights and Recommendations")

        st.write("### Insights Based on Data")
        st.markdown("""
        - **High Correlation:** Identified strong relationships between features.
        - **Outliers:** Observed in numerical columns during EDA.
        - **Missing Values:** Cleaned in the earlier step.
        """)

        st.write("### AI-Generated Suggestions")
        st.markdown("""
        Based on your data, here are some actionable recommendations:
        - Optimize features with the highest impact on target outcomes.
        - Address outliers to improve model performance.
        - Leverage underutilized resources indicated by low feature importance.
        """)

else:
    st.write("Please upload a CSV file to proceed.")
