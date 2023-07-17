import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.question_answering import load_qa_chain
from genai.extensions.langchain import LangChainInterface
from genai.schemas import ModelType, GenerateParams
from genai.model import Credentials
import time
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import pdfminer
from pdfminer.high_level import extract_pages

st.title("Retrieval Augmented Generation App")
st.header("This app was developed by Sharath Kumar RK, Ecosystem Engineering Watsonx team")

genai_api_key = st.sidebar.text_input("GenAI API Key", type="password")
genai_api_url = st.sidebar.text_input("GenAI API URL", type="default")
chunk_size = st.sidebar.text_input("Select Chunk size", type="default")
chunk_overlap = st.sidebar.text_input("Select Chunk overlap", type="default")


st.write(pdfminer.__version__)  

uploaded_file = st.file_uploader("Choose a file", "pdf")
if uploaded_file is not None:
    for page_layout in extract_pages(uploaded_file):
        for element in page_layout:
            st.write(element)


splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
loader = UnstructuredPDFLoader()
loaded_docs = loader.load()
chunked_docs = splitter.split_documents(loaded_docs)




embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs={"device": "cpu"}
        )

docsearch = Chroma.from_documents(chunked_docs, embeddings)


def rag(question):
    # Create creds object
    creds = Credentials(api_key=genai_api_key, api_endpoint=genai_api_url)
    # Define parameters
    params = GenerateParams(decoding_method="sample", temperature=0.7, max_new_tokens=400, min_new_tokens=10, repetition_penalty=2)
    # Instantiate LLM model
    llm=LangChainInterface(model=ModelType.FLAN_T5_11B, params=params, credentials=creds)
    # Create retriever object
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=docsearch.as_retriever())
    # Run LLM model
    response = qa.run(question)
    # Print results
    return st.info(response)


with st.form("myform"):
    question = st.text_input("Type your question:", "")
    submitted = st.form_submit_button("Submit")
    if not genai_api_key:
        st.info("Please add your GenAI API key & GenAI API URL to continue.")
    elif submitted:
        rag(question)
