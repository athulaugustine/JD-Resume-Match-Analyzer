from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import Docx2txtLoader, PyPDFium2Loader , TextLoader
from langchain_community.chat_models import ChatOllama
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import pandas as pd

class Applicant(BaseModel):
    """Information about a job applicant based on the job description and applicant resume."""

    name: Optional[str] = Field(default=None, description="Name of the job applicant.")
    email: Optional[str] = Field(default=None, description="Email address of the job applicant.")
    mobile: Optional[str] = Field(default=None, description="Mobile number of the job applicant.")
    skills_matching: Optional[str] = Field(
        default=None,
        description="required skills from the Job Description which the applicant possesses ."
    )
    skills_missing: Optional[str] = Field(
        default=None,
        description="required skills from the Job Description which the applicant lacks."
    )
    eligible_for_role: Optional[str] = Field(
        default=None,
        description="Applicant's eligibility for the role based on the APPLICANT ELIGIBILITY."
    )
    education_background: Optional[str] = Field(
        default=None,
        description="Educational background of the job applicant."
    )
    total_work_experience: Optional[int] = Field(
        default=None,
        description="Total years of work experience of the job applicant."
    )
    relevant_work_experience: Optional[int] = Field(
        default=None,
        description="Years of relevant work experience of the job applicant."
    )
    skills: Optional[List[str]] = Field(
        default=None,
        description="List of key skills possessed by the job applicant."
    )
    certifications: Optional[List[str]] = Field(
        default=None,
        description="List of relevant certifications held by the job applicant."
    )
    languages: Optional[List[str]] = Field(
        default=None,
        description="Languages known by the job applicant."
    )
    additional_information: Optional[str] = Field(
        default=None,
        description="Any additional relevant information about the job applicant."
    )

class Data(BaseModel):
    """
    Extracted data about job applicants.
    """

    applicants: List[Applicant]


def extract_text(file_is, file_type):
    if file_type == 'docx':
        loader = Docx2txtLoader(file_is)
    elif file_type == 'pdf':
        loader = PyPDFium2Loader(file_is)
    elif file_type == 'text':
        loader = TextLoader(file_is)
    else:
        return None
    
    data = loader.load()
    return data




def format_content(JD_Resume,gpt_key):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo",api_key=gpt_key)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(JD_Resume)
    prompt_template = """
    Extract and convert the following resume or job description into a structured and valid JSON format summary, focusing on details, skills, experience, technologies, qualifications, certifications, and any other relevant information. Extract applicant details like mobile number and email if it is a resume.

    JD_OR_RESUME:
    "{text}"

    Please provide the summary strictly in valid JSON format, clearly indicating whether it is a job description (if requirements are present) or a resume (if applicant details are present).

    JSON FORMAT SUMMARY:
    """
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final structured valid JSON format summary.\n"
        "We have provided an existing summary in JSON format up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary in valid JSON format "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary in valid JSON format. "
        "If the context isn't useful, return the original summary in valid JSON format."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )

    response = chain.invoke({"input_documents": split_docs}, return_only_outputs=True)
    return response["output_text"]


def elibility_check(job_description, resume,gpt_key):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo",api_key=gpt_key)
    prompt_template = """Analyze the following job description and resume to determine the eligibility of the applicant. Provide a comprehensive assessment covering the following aspects:

    1. Key Requirements from Job Description:
    - List of required skills, qualifications, and experiences.
    - Specific responsibilities and duties mentioned.
    - Preferred attributes or experiences.

    2. Applicant's Qualifications:
    - List of skills, qualifications, and experiences from the resume.
    - Relevant job titles and durations of employment.
    - Educational background and certifications.

    3. Matching Analysis:
    - Direct matches between job requirements and applicant's qualifications.
    - Examples where the applicant meets or exceeds the job requirements.
    - Instances where the applicant has additional skills or experiences beneficial but not listed in the job description.

    4. Gaps Analysis:
    - Specific areas where the applicant does not meet the job requirements.
    - Missing qualifications or experiences.
    - Potential impact of these gaps on the applicant's overall eligibility.
    - BTech(Bachelor of Technology) and B.E.(Bachelor of Engineering) are both same engineering degrees.

    5. Overall Eligibility Assessment:
    - Summary of the applicant's suitability for the position.
    - Strengths and potential weaknesses.
    - Final recommendation on the applicant's eligibility.

    Job Description:
    {job_description}

    Applicant's Resume:
    {applicant_resume}

    Final Decision:
    """




    prompt = PromptTemplate(
        input_variables=["job_description","applicant_resume"], template=prompt_template
    )
    chain = prompt | llm
    response = chain.invoke({"job_description":job_description,"applicant_resume":resume})
    print(response)
    return response


def analyze_jd_resume(job_description, resume,applicant_eligibility,gpt_key):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo",api_key=gpt_key)
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "As an advanced hiring manager algorithm, extract all the required information"
        "ignore educational qualification in matching_skills and missing_skills"
        "skills_matching and skills_missing should have only the skills from JOB DESCRIPTION which the applicant possess or lack"
    ),
    (
        "human",
        "JOB DESCRIPTION: {job_description}\n\n"
        "RESUME: {resume}\n\n"
        "APPLICANT ELIGIBILITY: {applicant_eligibility}"
    ),
])

    runnable = prompt | llm.with_structured_output(schema=Data)
    response = runnable.invoke({"job_description": job_description, "resume": resume,"applicant_eligibility":applicant_eligibility})
    
    return response.applicants


def file_create(uploaded_JD,file_name):
    uploaded_JD_type = uploaded_JD.name.split('.')[-1]
    if uploaded_JD_type=='pdf':
                    file_is = f"{file_name}.pdf"
                    with open(file_is,'wb') as file:
                        file.write(uploaded_JD.getvalue())
    elif uploaded_JD_type=='docx' or uploaded_JD_type=='doc':
                    file_is = f"{file_name}.docx"
                    with open(file_is,'wb') as file:
                        file.write(uploaded_JD.getvalue())
    elif uploaded_JD_type=='txt':
                    file_is = f"{file_name}.txt"
                    with open(file_is,'wb') as file:
                        file.write(uploaded_JD.getvalue())
    return file_is,uploaded_JD_type                    


def convert_to_dataframe(applicants_data):
    data_dict = {
        'Name': [a.name for a in applicants_data],
        'Email': [a.email for a in applicants_data],
        'Mobile': [a.mobile for a in applicants_data],
        'Skills Matching': [a.skills_matching for a in applicants_data],
        'Skills Not Matching': [a.skills_missing for a in applicants_data],
        'Eligible for role': [a.eligible_for_role for a in applicants_data],
        'Education Background': [a.education_background for a in applicants_data],
        'Total Years of Experience': [a.total_work_experience for a in applicants_data],
        'Relevant Years of Experience': [a.relevant_work_experience for a in applicants_data],
        'Skills': [a.skills for a in applicants_data],
        'Certifications': [a.certifications for a in applicants_data],
        'Languages': [a.languages for a in applicants_data],
        'Additional Information': [a.additional_information for a in applicants_data]
    }

    df = pd.DataFrame(data_dict)
    return df
