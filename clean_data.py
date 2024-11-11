import pandas as pd
import streamlit as st

def fetch_student_data():
    """Fetches and combines the student performance datasets."""
    try:
        math_df = pd.read_csv('dirty/student-mat.csv', sep=';', index_col=0)
        port_df = pd.read_csv('dirty/student-por.csv', sep=';', index_col=0)
        for df in [math_df, port_df]:
            string_columns = df.select_dtypes(include=['object']).columns
            for col in string_columns:
                df[col] = df[col].str.replace('"', '').str.strip()
        math_df['subject'] = 'math'
        port_df['subject'] = 'portuguese'
        combined_df = pd.concat([math_df, port_df], axis=0)
        combined_df = combined_df.reset_index(drop=True)
        return combined_df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def clean_data(df):
    """Cleans the student data."""
    if df is None:
        return None
    
    try:
        cleaned_df = df.copy()
        grade_columns = ['G1', 'G2', 'G3']
        for col in grade_columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        numeric_columns = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
        return cleaned_df
    
    except Exception as e:
        st.error(f"Error cleaning data: {str(e)}")
        return None