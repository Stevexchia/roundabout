import streamlit as st
import pandas as pd
from src.preprocess import preprocess_dataframe
from src.policies import MultiPolicyClassifier
from pathlib import Path
import tempfile

st.title("ROUNDABAOUT - Google Location Review Checker")

st.write(
    """
    **Evaluate the quality and relevancy of Google location reviews using ML.**
    
    This tool detects and flags:
    - Spam or advertisements
    - Irrelevant content (off-topic or unrelated to the location/rating category)
    - Rants or complaints from users who likely have not visited the place

    Upload a CSV of reviews (raw or preprocessed). If needed, the system will automatically clean your data before analysis.
    """
)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'text_clean' not in df.columns:
        st.warning("Data has not been cleaned! Running preprocessing...")
        df = preprocess_dataframe(df)
        if 'text_clean' not in df.columns:
            st.error("Preprocessing failed to produce a sanitised data.")

    if 'text_clean' in df.columns:
        st.write("Preview of your data:", df.head())

        if st.button("Run Policy Evaluation"):
            with st.spinner("Loading models and running inference..."):
                multi_classifier = MultiPolicyClassifier()
                models_dir = Path("outputs/models")
                multi_classifier.load_all_models(models_dir=models_dir)
                results_df = multi_classifier.predict_all(df)
                if 'review_id' in df.columns:
                    results_df['review_id'] = df['review_id']

                if 'review_id' in df.columns and 'review_id' in results_df.columns:
                    output_df = pd.merge(df, results_df, on='review_id', how='left', suffixes=('', '_pred'))
                else:
                    output_df = pd.concat([df, results_df], axis=1)
                st.success("Inference complete!")
                st.write(output_df.head())

                # Download link
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                output_df.to_csv(tmp_file.name, index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=open(tmp_file.name, "rb"),
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )