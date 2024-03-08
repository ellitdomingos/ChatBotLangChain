import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader, PdfWriter
import os
import base64
import os
openai_api_key = os.getenv('OPENAI_API_KEY')
from htmlTemplates import expander_css, css, bot_template, user_template
from tempfile import NamedTemporaryFile

def process_file(doc):
    model_name = os.getenv('HUGGINGFACE_MODEL', 'default-model')
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)  
    
    pdf_search = Chroma.from_documents(doc, embeddings)
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.3), 
        retriever=pdf_search.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True)
    return chain

# Adicionar um uploader de arquivos e um botão para processar o PDF
with st.session_state.col1:
    st.session_state.pdf_doc = st.file_uploader("Upload your PDF here and click on 'Process'")

    if st.button("Process", key='process_pdf'):
        if st.session_state.pdf_doc is not None:
            with NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
                temp.write(st.session_state.pdf_doc.getvalue())
                temp.seek(0)
                loader = PyPDFLoader(temp.name)
                pdf = loader.load()
                st.session_state.conversation = process_file(pdf)
                st.session_state.col1.markdown("Done processing. You may now ask a question.")

if __name__ == '__main__':
    main()
