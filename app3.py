import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import os

# Importando os modelos HTML e CSS personalizados
from htmlTemplates import expander_css, css, bot_template, user_template

# Configuração da página
st.set_page_config(layout="wide", page_title="Interactive PDF Reader", page_icon=":books:")
st.write(css, unsafe_allow_html=True)

# Inicialização das variáveis de estado
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Caminho para o arquivo PDF pré-definido
pdf_path = 'MANUAL_DO_ORGASMO_MULTIPLO.pdf'

# Função para processar o PDF
def process_pdf(api_key):
    loader = PyPDFLoader(pdf_path)
    doc = loader.load()
    # Configure o modelo e embeddings conforme necessário
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small", model_kwargs={'device': 'cpu'})
    pdf_search = Chroma.from_documents(doc, embeddings)
    # Inicialize o cliente OpenAI com a chave de API fornecida
    chat_model = ChatOpenAI(api_key=api_key, temperature=0.3)
    chain = ConversationalRetrievalChain.from_llm(chat_model, retriever=pdf_search.as_retriever(search_kwargs={"k": 2}), return_source_documents=True)
    return chain

# Processamento do PDF
openai_api_key = os.environ.get('OPENAI_API_KEY')
 # Obter a chave de API do ambiente
if openai_api_key is None:
    st.error("A chave de API da OpenAI não foi encontrada. Certifique-se de defini-la nas variáveis de ambiente.")
else:
    st.session_state.conversation = process_pdf(openai_api_key)

# Função para lidar com a entrada do usuário
def handle_userinput(query):
    response = st.session_state.conversation({"question": query, "chat_history": st.session_state.chat_history}, return_only_outputs=True)
    st.session_state.chat_history.append((query, response['answer']))
    st.write(response['answer'])

# Função principal
def main():
    # Criando o layout da página
    col1, col2 = st.columns([1, 1])
    with col1:
        col1.header("Faça uma pergunta a Izuela responde")
        user_question = st.text_input("Digite sua pergunta aqui:")

        if user_question:
            handle_userinput(user_question)

if __name__ == '__main__':
    main()
#streamlit run app3.py    