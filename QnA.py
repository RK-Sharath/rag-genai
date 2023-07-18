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

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
if uploaded_files:
    raw_text = ''
    # Loop through each uploaded file
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
                text_splitter = CharacterTextSplitter(
                    separator="\n", # line break
                    chunk_size = chunk_size,
                    chunk_overlap = chunk_overlap,  
                    length_function=len,
                )
                
                texts = text_splitter.split_documents(raw_text)
                embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large",model_kwargs={"device": "cpu"})
                embeddings = embeddings
                docsearch = Chroma.from_documents(texts, embeddings)
                creds = Credentials(api_key=genai_api_key, api_endpoint=genai_api_url)
                params= GenerateParams(decoding_method="sample", temperature=0.7, max_new_tokens=400, min_new_tokens=10, repetition_penalty=2)
                llm=LangChainInterface(model=ModelType.FLAN_T5_11B, params=params, credentials=creds)
                qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=docsearch.as_retriever())
                
query = st.text_input("Ask a question or give an instruction")
submitted = st.form_submit_button("Submit")
if query:
    answer = qa.run(query)
    st.write(answer)
