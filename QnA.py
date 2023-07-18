import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.question_answering import load_qa_chain
from genai.extensions.langchain import LangChainInterface
from genai.schemas import ModelType, GenerateParams
from genai.model import Credentials
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import base64
import fitz

st.title("Retrieval Augmented Generation App")
st.caption("This app was developed by Sharath Kumar RK, Ecosystem Engineering Watsonx team")

genai_api_key = st.sidebar.text_input("GenAI API Key", type="password")
genai_api_url = st.sidebar.text_input("GenAI API URL", type="default")
chunk_size = st.sidebar.text_input("Select Chunk size", type="default")
chunk_overlap = st.sidebar.text_input("Select Chunk overlap", type="default")


uploaded_pdf = st.file_uploader("Load pdf: ", type=['pdf'])

if uploaded_pdf is not None:
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.getText()
    st.write(text) 
    doc.close()
    
text_splitter=CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
chunked_docs=splitter.split_documents(text)



with st.form("myform"):
    question = st.text_input("Type your question:", value="", placeholder="")
    submitted = st.form_submit_button("Submit")
    if submitted and genai_api_key.startswith('pak-'):
        with st.spinner('Working on it...'):
            if not genai_api_key:
                st.info("Please add your GenAI API key & GenAI API URL to continue.")
            elif submitted:      
                qa.run(question)
