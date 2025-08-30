import streamlit as st
import pandas as pd
from src.policies import MultiPolicyClassifier
from pathlib import Path
import tempfile

st.title("Roundabout Batch Inference")

st.write("Upload a CSV file with a 'text_clean' column to get policy predictions.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'text_clean' not in df.columns:
        st.error("CSV must contain a 'text_clean' column.")
    else:
        st.write("Loaded data preview:", df.head())

        if st.button("Run Inference"):
            with st.spinner("Loading models and running inference..."):
                multi_classifier = MultiPolicyClassifier()
                models_dir = Path("outputs/models")
                multi_classifier.load_all_models(models_dir=models_dir)
                results_df = multi_classifier.predict_all(df['text_clean'].tolist())
                if 'review_id' in df.columns:
                    results_df['review_id'] = df['review_id']
                # Combine with original data if you want
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