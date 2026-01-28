import streamlit as st
import requests
import pandas as pd
import time
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="Automated Data Analyzer", layout="wide")

st.title("Automated Data Analyzer")
st.markdown("Upload a CSV file for AI-powered comprehensive analysis")

API_URL = os.getenv("API_URL", "http://localhost:8000")

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")
    
    df = pd.read_csv(uploaded_file)
    
    st.info(f"Dataset loaded: {len(df)} rows × {len(df.columns)} columns")
    
    with st.expander("Data Preview (first 50 rows)", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)
    
    if st.button("Analyze Dataset with AI", type="primary", use_container_width=True):
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
        
        try:
            response = requests.post(f"{API_URL}/api/upload/", files=files, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                job_id = result['job_id']
                
                st.info(f"Analysis started | Job ID: {job_id}")
                
                progress_bar = st.progress(0)
                status_container = st.empty()
                
                max_polls = 60
                poll_count = 0
                
                while poll_count < max_polls:
                    status_response = requests.get(f"{API_URL}/api/jobs/{job_id}")
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        
                        progress = status_data.get('progress', 0)
                        progress_bar.progress(progress / 100)
                        
                        current_status = status_data['status'].upper()
                        
                        if progress < 30:
                            status_text = "Loading and validating dataset..."
                        elif progress < 50:
                            status_text = "Running statistical analysis..."
                        elif progress < 60:
                            status_text = "Computing correlations..."
                        elif progress < 70:
                            status_text = "Detecting patterns and outliers..."
                        elif progress < 90:
                            status_text = "Generating AI insights..."
                        else:
                            status_text = "Finalizing analysis..."
                        
                        status_container.info(f"Status: {current_status} | {status_text} ({progress}%)")
                        
                        if current_status == 'COMPLETED':
                            progress_bar.progress(100)
                            st.success("Analysis Complete")
                            
                            st.markdown("---")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Rows", f"{status_data.get('rows', 'N/A'):,}")
                            with col2:
                                st.metric("Columns", status_data.get('columns', 'N/A'))
                            with col3:
                                duration = "N/A"
                                if status_data.get('created_at') and status_data.get('completed_at'):
                                    from datetime import datetime
                                    start = datetime.fromisoformat(status_data['created_at'].replace('Z', '+00:00'))
                                    end = datetime.fromisoformat(status_data['completed_at'].replace('Z', '+00:00'))
                                    duration = f"{(end - start).total_seconds():.1f}s"
                                st.metric("Duration", duration)
                            
                            st.markdown("---")
                            
                            if 'analysis' in status_data:
                                analysis = status_data['analysis']
                                
                                if 'llm_insights' in analysis and 'failed' not in analysis['llm_insights'].lower():
                                    st.subheader("AI-Powered Insights")
                                    st.markdown(analysis['llm_insights'])
                                    st.markdown("---")
                                
                                # NEW: Missing Data Patterns
                                if 'missing_patterns' in analysis:
                                    st.subheader("Missing Data Patterns Analysis")
                                    patterns = analysis['missing_patterns']
                                    
                                    if 'overall' in patterns:
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Total Missing Cells", f"{patterns['overall'].get('total_missing_cells', 0):,}")
                                        with col2:
                                            st.metric("Missing Percentage", f"{patterns['overall'].get('missing_percentage', 0):.2f}%")
                                        with col3:
                                            st.metric("Rows with Missing", f"{patterns['overall'].get('rows_with_missing', 0):,}")
                                    
                                    if 'by_column' in patterns and patterns['by_column']:
                                        st.write("**Missing Values by Column:**")
                                        missing_col_data = []
                                        for col, info in patterns['by_column'].items():
                                            missing_col_data.append({
                                                'Column': col,
                                                'Missing Count': info['count'],
                                                'Percentage': f"{info['percentage']:.2f}%"
                                            })
                                        st.dataframe(pd.DataFrame(missing_col_data), use_container_width=True)
                                    
                                    if 'patterns' in patterns and 'top_10_patterns' in patterns['patterns']:
                                        with st.expander("View Top Missing Patterns"):
                                            st.write(f"**Unique Patterns Found:** {patterns['patterns'].get('unique_patterns', 0)}")
                                            for pattern in patterns['patterns']['top_10_patterns'][:5]:
                                                st.write(f"Pattern {pattern['rank']}: {pattern['count']} rows ({pattern['percentage']:.2f}%)")
                                    
                                    st.markdown("---")
                                
                                # NEW: Missing Mechanism Classification
                                if 'missing_mechanisms' in analysis and analysis['missing_mechanisms']:
                                    st.subheader("Missing Data Mechanism (MCAR/MAR/MNAR)")
                                    
                                    mech_data = []
                                    for col, mech_info in analysis['missing_mechanisms'].items():
                                        mech_data.append({
                                            'Column': col,
                                            'Mechanism': mech_info.get('mechanism', 'Unknown'),
                                            'Explanation': mech_info.get('explanation', '')
                                        })
                                    
                                    if mech_data:
                                        st.dataframe(pd.DataFrame(mech_data), use_container_width=True)
                                        
                                        st.info("**MCAR** = Missing Completely At Random | **MAR** = Missing At Random | **MNAR** = Missing Not At Random")
                                    
                                    st.markdown("---")
                                
                                # NEW: Normality Tests
                                if 'normality_tests' in analysis and analysis['normality_tests']:
                                    st.subheader("Distribution Normality Tests (Shapiro-Wilk)")
                                    
                                    norm_data = []
                                    for col, test_result in analysis['normality_tests'].items():
                                        if isinstance(test_result, dict) and 'p_value' in test_result:
                                            norm_data.append({
                                                'Column': col,
                                                'P-Value': f"{test_result['p_value']:.4f}",
                                                'Distribution': test_result.get('interpretation', 'Unknown'),
                                                'Is Normal': '✓' if test_result.get('is_normal', False) else '✗'
                                            })
                                    
                                    if norm_data:
                                        st.dataframe(pd.DataFrame(norm_data), use_container_width=True)
                                        st.caption("p-value > 0.05 indicates normal distribution")
                                    
                                    st.markdown("---")
                                
                                # NEW: Intelligent Type Detection
                                if 'intelligent_types' in analysis and analysis['intelligent_types']:
                                    with st.expander("Intelligent Column Type Detection"):
                                        type_data = []
                                        for col, col_type in analysis['intelligent_types'].items():
                                            type_data.append({
                                                'Column': col,
                                                'Detected Type': col_type
                                            })
                                        
                                        if type_data:
                                            st.dataframe(pd.DataFrame(type_data), use_container_width=True)
                                
                                # NEW: Advanced Outlier Detection
                                if 'outliers_advanced' in analysis and analysis['outliers_advanced']:
                                    st.subheader("Advanced Outlier Detection")
                                    
                                    outliers_adv = analysis['outliers_advanced']
                                    
                                    if 'total_outliers' in outliers_adv:
                                        st.info(f"**Isolation Forest Detection:** {outliers_adv.get('total_outliers', 0)} outliers ({outliers_adv.get('percentage', 0):.2f}%)")
                                    else:
                                        outlier_data = []
                                        for col, outlier_info in outliers_adv.items():
                                            if isinstance(outlier_info, dict) and 'count' in outlier_info:
                                                outlier_data.append({
                                                    'Column': col,
                                                    'Outliers': outlier_info['count'],
                                                    'Percentage': f"{outlier_info['percentage']:.2f}%",
                                                    'Lower Bound': f"{outlier_info.get('lower_bound', 'N/A')}",
                                                    'Upper Bound': f"{outlier_info.get('upper_bound', 'N/A')}"
                                                })
                                        
                                        if outlier_data:
                                            st.dataframe(pd.DataFrame(outlier_data), use_container_width=True)
                                    
                                    st.markdown("---")
                                
                                # NEW: Data Quality Report
                                if 'data_quality' in analysis:
                                    with st.expander("Comprehensive Data Quality Report"):
                                        quality = analysis['data_quality']
                                        
                                        if 'recommendations' in quality and quality['recommendations']:
                                            st.write("**Column-Specific Recommendations:**")
                                            for col, recs in quality['recommendations'].items():
                                                st.write(f"**{col}:**")
                                                if 'imputation' in recs:
                                                    st.write(f"  - Imputation: {recs['imputation'].get('strategy', 'N/A')} - {recs['imputation'].get('reason', '')}")
                                                if 'outlier_treatment' in recs:
                                                    st.write(f"  - Outliers: {recs['outlier_treatment'].get('recommendation', 'N/A')} - {recs['outlier_treatment'].get('reason', '')}")
                                
                                # NEW: Feature Engineering Results
                                if 'feature_engineering' in analysis:
                                    st.subheader("Automated Feature Engineering")
                                    
                                    feat_eng = analysis['feature_engineering']
                                    
                                    if feat_eng.get('new_features_count', 0) > 0:
                                        st.success(f"✓ Generated {feat_eng['new_features_count']} new features")
                                        
                                        if 'top_20_features' in feat_eng and feat_eng['top_20_features']:
                                            st.write("**Top 20 Features by Importance:**")
                                            
                                            top_features_data = []
                                            for feat in feat_eng['top_20_features']:
                                                top_features_data.append({
                                                    'Rank': feat['rank'],
                                                    'Feature': feat['feature'],
                                                    'Importance Score': f"{feat['score']:.6f}",
                                                    'Method': feat['method']
                                                })
                                            
                                            st.dataframe(pd.DataFrame(top_features_data), use_container_width=True)
                                        
                                        if 'statistics' in feat_eng:
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Total Features", feat_eng['statistics'].get('total_features', 0))
                                            with col2:
                                                st.metric("Numeric Features", feat_eng['statistics'].get('numeric_features', 0))
                                            with col3:
                                                st.metric("Missing in Features", feat_eng['statistics'].get('missing_values', 0))
                                    else:
                                        st.info("No new features could be generated from this dataset")
                                    
                                    st.markdown("---")
                                
                                # Existing: Correlation Analysis
                                if 'correlations' in analysis and 'heatmap' in analysis['correlations']:
                                    st.subheader("Correlation Analysis")
                                    
                                    try:
                                        fig = go.Figure(json.loads(analysis['correlations']['heatmap']))
                                        st.plotly_chart(fig, use_container_width=True)
                                    except:
                                        pass
                                    
                                    if analysis['correlations'].get('high_correlations'):
                                        st.write("**High Correlations Found:**")
                                        corr_df = pd.DataFrame(analysis['correlations']['high_correlations'])
                                        st.dataframe(corr_df, use_container_width=True)
                                    
                                    st.markdown("---")
                                
                                # Existing: Missing Values
                                st.subheader("Missing Values Analysis")
                                missing = analysis.get('missing_analysis', {})
                                if missing.get('columns_with_missing'):
                                    missing_df = pd.DataFrame(missing['columns_with_missing'])
                                    st.dataframe(missing_df, use_container_width=True)
                                else:
                                    st.success("No missing values detected")
                                
                                # Existing: Numeric Statistics
                                st.subheader("Numeric Statistics")
                                numeric_stats = analysis.get('numeric_stats', {})
                                if numeric_stats and 'error' not in numeric_stats and 'message' not in numeric_stats:
                                    stats_df = pd.DataFrame(numeric_stats).T
                                    st.dataframe(stats_df, use_container_width=True)
                                elif 'message' in numeric_stats:
                                    st.info(numeric_stats['message'])
                                
                                # Existing: Outliers
                                st.subheader("Outliers Detected")
                                outliers = analysis.get('outliers', {})
                                if outliers and 'error' not in outliers:
                                    outliers_df = pd.DataFrame(outliers).T
                                    st.dataframe(outliers_df, use_container_width=True)
                                else:
                                    st.info("No significant outliers detected")
                                
                                # Existing: Recommendations
                                if 'recommendations' in analysis and analysis['recommendations']:
                                    st.subheader("Data Quality Recommendations")
                                    for rec in analysis['recommendations']:
                                        severity = rec.get('severity', 'info')
                                        if severity == 'high':
                                            st.error(f"**{rec['column']}**: {rec['issue']} - {rec['action']}")
                                        elif severity == 'medium':
                                            st.warning(f"**{rec['column']}**: {rec['issue']} - {rec['action']}")
                                        else:
                                            st.info(f"**{rec['column']}**: {rec['issue']} - {rec['action']}")
                                
                                with st.expander("View Complete Analysis JSON"):
                                    st.json(status_data.get('analysis', {}))
                            else:
                                st.info("Analysis completed but detailed results not available")
                                with st.expander("View Job Details"):
                                    st.json(status_data)
                            
                            break
                        
                        elif current_status == 'FAILED':
                            st.error(f"Analysis failed: {status_data.get('error_message', 'Unknown error')}")
                            break
                    
                    else:
                        st.error(f"Failed to get status: {status_response.status_code}")
                        break
                    
                    poll_count += 1
                    time.sleep(2)
                
                if poll_count >= max_polls:
                    st.warning("Analysis is taking longer than expected")
            
            else:
                st.error(f"Upload failed: {response.status_code}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.caption("Built with FastAPI + Celery + PostgreSQL + Redis + Streamlit | Powered by Groq AI")