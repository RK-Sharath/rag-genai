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
import base64

st.title("Retrieval Augmented Generation App")
st.header("This app was developed by Sharath Kumar RK, Ecosystem Engineering Watsonx team")

genai_api_key = st.sidebar.text_input("GenAI API Key", type="password")
genai_api_url = st.sidebar.text_input("GenAI API URL", type="default")
chunk_size = st.sidebar.text_input("Select Chunk size", type="default")
chunk_overlap = st.sidebar.text_input("Select Chunk overlap", type="default")

file = st.file_uploader("Upload a PDF file from your computer", type="pdf")
if file is not None:
    progress_bar = st.progress(0, text="Uploading file...")
    st.session_state.engine.save_file(file)
    progress_bar.progress(100, text=f"File {file.name} is successfully saved.")
    progress_bar.empty()
    st.experimental_rerun()

def filename(self):
    return self._filename

def load_data(self):
    loader = UnstructuredPDFLoader(self._filename)
    self._data = loader.load()
        
def chunked_docs(self):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )
    self._texts = text_splitter.split_documents(self._data)
    if self._texts is None or len(self._texts) == 0:
        raise Exception("The document does not contain any text.")


def embeddings(self):
    embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs={"device": "cpu"}
        )
    return embeddings

def creds():
    creds = Credentials(api_key=genai_api_key, api_endpoint=genai_api_url)
    return creds

def params(self):   
    params= GenerateParams(decoding_method="sample", temperature=0.7, max_new_tokens=400, min_new_tokens=10, repetition_penalty=2)
    return params

def llm(self):
    llm=LangChainInterface(model=ModelType.FLAN_T5_11B, params=params, credentials=creds)
    return llm

def docsearch(self):
    docsearch = Chroma.from_documents(chunked_docs, embeddings)
    return docsearch

def qa(self):
    qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=docsearch.as_retriever())
    response = qa.run()
    return st.info(response)


with st.form("myform"):
    question = st.text_input("Type your question:", value="", placeholder="")
    submitted = st.form_submit_button("Submit")
    if not genai_api_key:
        st.info("Please add your GenAI API key & GenAI API URL to continue.")
    elif submitted:
        qa(question)
