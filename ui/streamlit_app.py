import streamlit as st
import pandas as pd
from src.preprocess import preprocess_dataframe
from src.policies import MultiPolicyClassifier
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

def display_policy_summary(results_df: pd.DataFrame):
    """Display a brief summary of the policy analysis results."""
    total_reviews = len(results_df)
    
    st.subheader("Policy Analysis Summary")
    
    # Overall relevance
    if 'is_relevant' in results_df.columns:
        relevant_count = results_df['is_relevant'].sum()
        relevant_pct = (relevant_count / total_reviews) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Reviews Analyzed", total_reviews)
        with col2:
            st.metric("Relevant Reviews", f"{relevant_count} ({relevant_pct:.1f}%)")
    
    # Policy violations breakdown
    st.write("**Policy Violations Detected:**")
    
    policy_cols = {
        'is_advertisement': 'Advertisement/Spam',
        'is_irrelevant': 'Irrelevant Content', 
        'is_rant_without_visit': 'Rants (No Visit)'
    }
    
    violation_data = []
    for col, label in policy_cols.items():
        if col in results_df.columns:
            count = results_df[col].sum()
            pct = (count / total_reviews) * 100
            violation_data.append({
                'Policy': label,
                'Violations': count,
                'Percentage': f"{pct:.1f}%"
            })
    
    if violation_data:
        violation_df = pd.DataFrame(violation_data)
        st.dataframe(violation_df, width="stretch", hide_index=True)
    
    # Visualizations
    st.subheader("Visual Analysis")
    
    # 1. Overall pie chart of relevant vs non-relevant
    if 'is_relevant' in results_df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart of relevance
        relevance_counts = results_df['is_relevant'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4']
        labels = ['Not Relevant', 'Relevant']
        ax1.pie(relevance_counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Overall Review Relevance')
        
        # Bar chart of policy violations
        policy_violations = []
        policy_names = []
        for col, label in policy_cols.items():
            if col in results_df.columns:
                policy_violations.append(results_df[col].sum())
                policy_names.append(label)  # Use label as-is since there are no emojis
        
        bars = ax2.bar(policy_names, policy_violations, color=['#ff9999', '#ffcc99', '#99ccff'])
        ax2.set_title('Policy Violations Count')
        ax2.set_ylabel('Number of Violations')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # 2. Rating category analysis if available
    if 'rating_category' in results_df.columns:
        st.write("**Analysis by Review Category:**")
        category_summary = results_df.groupby('rating_category').agg({
            'is_relevant': 'sum',
            'rating_category': 'count'
        }).rename(columns={'rating_category': 'total_count'})
        
        category_summary['relevant_percentage'] = (category_summary['is_relevant'] / category_summary['total_count'] * 100).round(1)
        category_summary = category_summary.reset_index()
        category_summary.columns = ['Category', 'Relevant Reviews', 'Total Reviews', 'Relevant %']
        
        st.dataframe(category_summary, width="stretch", hide_index=True)
        
        # Visualization for category analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Stacked bar chart showing relevant vs non-relevant by category
        category_data = results_df.groupby('rating_category')['is_relevant'].agg(['sum', 'count']).reset_index()
        category_data['not_relevant'] = category_data['count'] - category_data['sum']
        
        ax1.bar(category_data['rating_category'], category_data['sum'], 
                label='Relevant', color='#4ecdc4')
        ax1.bar(category_data['rating_category'], category_data['not_relevant'], 
                bottom=category_data['sum'], label='Not Relevant', color='#ff6b6b')
        ax1.set_title('Reviews by Category')
        ax1.set_ylabel('Number of Reviews')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Relevance percentage by category
        relevance_pct = (category_data['sum'] / category_data['count'] * 100)
        bars = ax2.bar(category_data['rating_category'], relevance_pct, color='#95e1d3')
        ax2.set_title('Relevance Rate by Category')
        ax2.set_ylabel('Relevant Reviews (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 100)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, relevance_pct):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # 3. Confidence distribution
    confidence_cols = [col for col in results_df.columns if col.endswith('_confidence')]
    if confidence_cols:
        st.write("**Model Confidence Distribution:**")
        fig, ax = plt.subplots(figsize=(10, 4))
        
        for i, col in enumerate(confidence_cols):
            policy_name = col.replace('_confidence', '').replace('_', ' ').title()
            ax.hist(results_df[col], bins=20, alpha=0.7, label=policy_name)
        
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Number of Reviews')
        ax.set_title('Distribution of Model Confidence Scores')
        ax.legend()
        st.pyplot(fig)

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

                # Merge results with original data
                if 'review_id' in df.columns and 'review_id' in results_df.columns:
                    output_df = pd.merge(df, results_df, on='review_id', how='left', suffixes=('', '_pred'))
                else:
                    output_df = pd.concat([df, results_df], axis=1)
                
                st.success("Inference complete!")
                
                # Display policy summary
                display_policy_summary(results_df)
                
                # Show sample results
                st.subheader("📋 Sample Results")
                display_cols = ['text', 'rating_category', 'is_relevant', 'is_advertisement', 'is_irrelevant', 'is_rant_without_visit']
                available_cols = [col for col in display_cols if col in results_df.columns]
                st.write(results_df[available_cols].head(10))

                # Download link
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                output_df.to_csv(tmp_file.name, index=False)
                st.download_button(
                    label="Download Full Results as CSV",
                    data=open(tmp_file.name, "rb"),
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )