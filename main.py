import streamlit as st
import asyncio
from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnableConfig
import os
import pdfplumber
from langchain.docstore.document import Document

# Constants
OPENING_MESSAGE = "Welcome to DataLUMA! How may I assist you today?"
EMBEDDING = "all-MiniLM-L6-v2"
RETRIEVED_CONTEXT = 1  # Context to retrieve

# Initialize ChatOllama Model
try:
    chat_model = ChatOllama(model="llama3.2:1b", temperature=0.8, num_predict=256)
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChatOllama model: {str(e)}")

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING)

# Function to load PDFs into documents
def load_pdfs_to_docs(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(folder_path, file_name)
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
                documents.append({"page_content": text, "metadata": {"source": file_name}})
    return documents

# Load and convert PDF files to documents
PDF_FOLDER = "./data"
pdf_documents = [Document(**doc) for doc in load_pdfs_to_docs(PDF_FOLDER)]

# Create FAISS VectorStore
vectorstore_path = "./faiss_index"
if not os.path.exists("./faiss_index"):
    # Membuat FAISS index baru
    vectorstore = FAISS.from_documents(pdf_documents, embedding_model)
    vectorstore.save_local("./faiss_index")
else:
    # Memuat FAISS index dengan deserialisasi yang aman
    vectorstore = FAISS.load_local("./faiss_index", embedding_model, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVED_CONTEXT})

# Combine chat history with new user input
def combine_chat_history(chat_history, user_input):
    combined_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    return f"{combined_history}\nhuman: {user_input}"

# Prepare context for the model, combining previous conversation and retrieved documents
def prepare_context(chat_history, retrieved_documents):
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    doc_text = "\n".join([doc.page_content for doc in retrieved_documents])
    return f"{history_text}\n\nDocuments:\n{doc_text}"

# Response generation based on user input and chat history
async def get_response_from_chain(user_input, chat_history):
    try:
        retrieved_documents = retriever.get_relevant_documents(user_input)
        if not retrieved_documents:
            return "Sorry, I couldn't find an answer in the documents."

        context = prepare_context(chat_history, retrieved_documents)

        # Determine the prompt based on the user input
        if "jadwal dokter" in user_input.lower():
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "Berikan jawaban berdasarkan jadwal dokter yang ada di RS Medika Utama. Dr. Aisyah melayani praktik pada hari Senin, Selasa, dan Jumat. Dr. Supratno melayani praktik pada hari Rabu dan Jumat. Dr. Priyono melayani praktik pada hari Rabu, Sabtu, dan Senin. Dr. Sujiatmojo melayani praktik pada hari Kamis, Selasa, dan Jumat."),
                ("human", "{context}")
            ])
        else:
            qa_prompt = ChatPromptTemplate.from_messages([("system", "Apa yang bisa saya bantu hari ini?"), ("human", "{context}")])

        # Prepare and process question-answer chain
        question_answer_chain = create_stuff_documents_chain(llm=chat_model, prompt=qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = ""
        async for chunk in rag_chain.astream(
            {"input": user_input, "context": context},
            config=RunnableConfig()
        ):
            if chunk.get("answer"):
                response += chunk["answer"]

        return response or "Sorry, I couldn't generate a response."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit app
async def run_chatbot():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Chatbot Assistant")
    st.write(OPENING_MESSAGE)

    user_input = st.text_input("Ask me anything:", "")
    if user_input:
        combined_input = combine_chat_history(st.session_state.chat_history, user_input)
        response = await get_response_from_chain(combined_input, st.session_state.chat_history)
        st.write(response)
        st.session_state.chat_history.append({"role": "human", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    asyncio.run(run_chatbot())
