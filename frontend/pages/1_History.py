import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Analysis History")
st.write("View all your past dataset analyses")

try:
    response = requests.get(f"{API_URL}/api/jobs?limit=100")
    
    if response.status_code == 200:
        data = response.json()
        
        if isinstance(data, list):
            jobs = data
        elif isinstance(data, dict) and 'jobs' in data:
            jobs = data['jobs']
        else:
            jobs = []
        
        if not jobs:
            st.info("No analyses found yet. Upload a CSV to get started!")
        else:
            for job in jobs:
                status = job.get('status', 'UNKNOWN')
                filename = job.get('filename', 'Unknown file')
                
                with st.expander(f"{filename} - {status}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Status", status)
                    with col2:
                        st.metric("Rows", job.get('rows', 'N/A'))
                    with col3:
                        st.metric("Columns", job.get('columns', 'N/A'))
                    
                    if status == 'COMPLETED':
                        job_id = job.get('id', '')
                        if job_id and st.button(f"View Results", key=job_id):
                            st.session_state.selected_job_id = job_id
                            st.switch_page("pages/2_Results.py")
    else:
        st.error(f"Failed to fetch history: {response.status_code}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write(f"API URL: {API_URL}")

st.markdown("---")
st.caption("Built with FastAPI + Celery + PostgreSQL + Redis + Streamlit | Powered by Groq AI")
