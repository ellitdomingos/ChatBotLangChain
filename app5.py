import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import os
import uuid

# Definição do estilo CSS para os expander
expander_css = """
<style>
/* Estilo CSS para os expander */
</style>
"""

# Estilo CSS para o chat
css = """
<style>
/* Estilo CSS para o chat */
.chat-container {
    max-height: 400px;
    overflow-y: scroll;
}
</style>
"""

# Configuração da página
st.set_page_config(layout="wide", page_title="Interactive PDF Reader", page_icon=":books:")
st.write(css, unsafe_allow_html=True)

# Inicialização das variáveis de estado
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Caminho para o arquivo PDF pré-definido
pdf_path = 'ANGOLA-Trilhos31-105.pdf'

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
    st.session_state.chat_history.append(("Usuário", query))
    st.session_state.chat_history.append(("Izuela", response['answer']))
    st.session_state.user_question = ""  # Limpar o campo de entrada de texto após enviar a pergunta

# Função principal
def main():
    # Criando a barra lateral
    st.sidebar.title("Configurações")
    user_question = st.sidebar.text_input("Digite sua pergunta aqui:", value=st.session_state.get("user_question", ""))

    if st.sidebar.button("Enviar"):
        handle_userinput(user_question)
        # Executar JavaScript para mover o foco para o histórico e limpar o campo de entrada
        st.sidebar.markdown(
            """
            <script>
                document.querySelector("#output-container-root").focus();
                document.querySelector("#input-container-root input").value = "";
            </script>
            """,
            unsafe_allow_html=True
        )

    # Criando o layout da página
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Chatbot")

    # Exibindo a conversa
    with col2:
        st.header("Conversa")
        with st.container():
            for sender, message in st.session_state.chat_history:
                if sender == "Usuário":
                    st.text_area(f"Você:", value=message, height=100, max_chars=None, key=str(uuid.uuid4()), disabled=True)
                else:
                    st.text_area(f"Izuela:", value=message, height=100, max_chars=None, key=str(uuid.uuid4()), disabled=True)

if __name__ == '__main__':
    main()
