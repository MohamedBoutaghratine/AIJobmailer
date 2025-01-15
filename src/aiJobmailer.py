import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load embeddings from HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Define LLM
llm = ChatGroq(
    temperature=0.3,
    groq_api_key='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',
    model_name='llama-3.1-70b-versatile'
)

# Process Resume PDF
def process_resume(uploaded_file):
    with open("resume.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyMuPDFLoader("resume.pdf")
    documents = loader.load()
    return documents

# Create Vector Store from Job Post URL and Resume
def get_combined_vectorstore(jobpost_url, resume_docs):
    # Process job post
    jobpost_loader = WebBaseLoader(jobpost_url)
    jobpost_docs = jobpost_loader.load()

    # Combine job post and resume documents
    all_docs = jobpost_docs + resume_docs

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    document_chunks = text_splitter.split_documents(all_docs)

    # Create vector store
    vector_store = Chroma.from_documents(document_chunks, embedding_model)
    return vector_store

# Define RAG Chain
def get_email_generation_chain(retriever):
    prompt = ChatPromptTemplate.from_template(
        "Given the following context:\n\n{context}\n\nGenerate a professional email highlighting only the most matched candidate's relevant skills and projects from his resume and CV with the job post requirements and be clear and direct. Focus on matching job requirements with the projects and the skills of the candidate and try to extract the name of candidate his education and his professional experience (jobs,internships) if it matched with the job post if there are a lot of things matched the job post requirements just give the most matched and important infomration from candidate resume to enhance the probabilty to be choosed, and the email object should contain the role of the job post and be clear and short and ensuring the tone is professional."
    )
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, stuff_documents_chain)

# Generate Email Response
def generate_email(vector_store):
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 5})
    email_chain = get_email_generation_chain(retriever)

    # Corrected Input Format
    query = "Generate a professional job application email mathcing the candidate skills,projects,experiences with the requirements of the job post,if the just some of projects of the candidate matches the jobpost requirements you have to generate just this projects else generate the most matched projects or skills with the job post requirements also extract the name of candidate and his education from his resume to give a good presentation about him and no need to put all informations about him just the important ones to be a personnalized email."
    response = email_chain.invoke({"input": query})
    return response['answer']


# Streamlit App Configuration
st.set_page_config(page_title="AI JobMailer", page_icon="📧")
st.title("AI JobMailer")

# Sidebar Inputs
with st.sidebar:
    st.header("Inputs")
    jobpost_url = st.text_input('Job Post URL')
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Generate Email Button
if st.button("Generate Email"):
    if not jobpost_url or not uploaded_file:
        st.warning("Please provide both Job Post URL and Resume PDF.")
    else:
        resume_docs = process_resume(uploaded_file)
        vector_store = get_combined_vectorstore(jobpost_url, resume_docs)
        email_response = generate_email(vector_store)
        st.write(email_response)
