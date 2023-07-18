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
from PyPDF2 import PdfReader

st.title("Retrieval Augmented Generation App")
st.caption("This app was developed by Sharath Kumar RK, Ecosystem Engineering Watsonx team")

genai_api_key = st.sidebar.text_input("GenAI API Key", type="password")
genai_api_url = st.sidebar.text_input("GenAI API URL", type="default")
chunk_size = st.sidebar.text_input("Select Chunk size", type="default")
chunk_overlap = st.sidebar.text_input("Select Chunk overlap", type="default")

uploaded_file = st.file_uploader("Choose PDF file", type="pdf", accept_multiple_files=False)
if uploaded_file:
    raw_text = ''
    pdf_reader = PdfReader(uploaded_file)
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
            
                
    def gen_content(question):
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(raw_text)
        docs = [Document(page_content=t) for t in texts]
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large",model_kwargs={"device": "cpu"})
        embeddings=embeddings
        docsearch = Chroma.from_documents(docs, embeddings)
        creds = Credentials(api_key=genai_api_key, api_endpoint=genai_api_url)
        params= GenerateParams(decoding_method="sample", temperature=0.7, max_new_tokens=400, min_new_tokens=10, repetition_penalty=2)
        llm=LangChainInterface(model=ModelType.FLAN_T5_11B, params=params, credentials=creds)
        qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=docsearch.as_retriever())
        response = qa.run(question)
        return st.info(response)
                
with st.form("myform"):
    question = st.text_input("Ask a question:", "")
    submitted = st.form_submit_button("Submit")
    if submitted and genai_api_key.startswith('pak-'):
        with st.spinner('Working on it...'):
            if not genai_api_key:
                st.info("Please add your GenAI API KEY & GenAI API URL to continue.")
            elif submitted:
                gen_content(question)
