import streamlit as st
from src.analyzer import *
import pandas as pd

uploaded_JD= st.file_uploader("Upload the Job Description",type=['pdf','docx','txt'])

uploaded_resumes = st.file_uploader("Upload Resumes", accept_multiple_files=True,type=['pdf','docx'])

if uploaded_JD and uploaded_resumes:
    if st.button("Analyze"):
        result_list = []
        with st.spinner(text="Analyzing..."):
            jd_file,uploaded_JD_type = file_create(uploaded_JD,"job_description")
            jd_content = extract_text(jd_file,uploaded_JD_type)
            jd_format = format_content(jd_content)
            for resume in uploaded_resumes:
                resume_file,resume_type = file_create(resume,"resume")
                resume_content = extract_text(resume_file,resume_type)
                resume_format = format_content(resume_content)
                applicant_eligibility = elibility_check(jd_format,resume_format)
                response = analyze_jd_resume(jd_format,resume_format,applicant_eligibility)
                json_data = response
                result_list.extend(json_data)
        final_result = convert_to_dataframe(result_list)
        st.write(final_result)