import streamlit as st
import asyncio
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnableConfig
from prompts import (
    contextualize_q_system_prompt, rag_system_prompt, hospital_general_info_prompt,
    doctor_schedule_prompt, administration_prompt, emergency_prompt,
    bed_availability_prompt, pharmacy_prompt, patient_discharging_prompt, insurance_payment_prompt
)
import os
import pdfplumber
from langchain.docstore.document import Document


OPENING_MESSAGE = "Welcome in DataLUMA! How may I assist you today?"
EMBEDDING = "all-MiniLM-L6-v2"
RETRIEVED_CONTEXT = 4

try:
    chat_model = ChatOllama(model="llama3.2:1b ", temperature=0.8, num_predict=256)
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChatOllama model: {str(e)}")

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVED_CONTEXT})

def load_pdfs_to_docs(folder_path):
    """Load PDF files from a folder and convert them into text documents."""
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

PDF_FOLDER = "./data"

pdf_documents = [Document(**doc) for doc in load_pdfs_to_docs(PDF_FOLDER)]

vectorstore.add_documents(pdf_documents)

def combine_chat_history(chat_history, user_input):
    """Combine chat history and user input into a single string."""
    combined_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    return f"{combined_history}\nhuman: {user_input}"

def prepare_context(chat_history, retrieved_documents):
    """Prepare the context for the chat model by combining history and retrieved documents."""
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    doc_text = "\n".join([doc.page_content for doc in retrieved_documents])
    return f"{history_text}\n\nDocuments:\n{doc_text}"

async def get_response_from_chain(user_input, chat_history):
    """Generate a response from the chat model based on user input and chat history."""
    try:
        retrieved_documents = retriever.get_relevant_documents(user_input)
        if not retrieved_documents:
            return "I'm sorry, I couldn't find an answer in the provided documents."
        
        context = prepare_context(chat_history, retrieved_documents)

        # Determine the appropriate prompt based on the user input
        if "jadwal dokter" in user_input.lower():
            qa_prompt = ChatPromptTemplate.from_messages([("system", doctor_schedule_prompt), ("human", "{context}")])
        elif "informasi rumah sakit" in user_input.lower():
            qa_prompt = ChatPromptTemplate.from_messages([("system", hospital_general_info_prompt), ("human", "{context}")])
        elif "administrasi" in user_input.lower():
            qa_prompt = ChatPromptTemplate.from_messages([("system", administration_prompt), ("human", "{context}")])
        elif "darurat" in user_input.lower():
            qa_prompt = ChatPromptTemplate.from_messages([("system", emergency_prompt), ("human", "{context}")])
        elif "kamar" in user_input.lower() or "tempat tidur" in user_input.lower():
            qa_prompt = ChatPromptTemplate.from_messages([("system", bed_availability_prompt), ("human", "{context}")])
        elif "apotek" in user_input.lower() or "obat" in user_input.lower():
            qa_prompt = ChatPromptTemplate.from_messages([("system", pharmacy_prompt), ("human", "{context}")])
        elif "keluar rumah sakit" in user_input.lower():
            qa_prompt = ChatPromptTemplate.from_messages([("system", patient_discharging_prompt), ("human", "{context}")])
        elif "asuransi" in user_input.lower() or "pembayaran" in user_input.lower():
            qa_prompt = ChatPromptTemplate.from_messages([("system", insurance_payment_prompt), ("human", "{context}")])
        else:
            qa_prompt = ChatPromptTemplate.from_messages([("system", rag_system_prompt), ("human", "{context}")])

        question_answer_chain = create_stuff_documents_chain(llm=chat_model, prompt=qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = ""
        async for chunk in rag_chain.astream(
            {"input": user_input, "context": context},
            config=RunnableConfig()
        ):
            if chunk.get("answer"):
                response += chunk["answer"]

        return response or "I'm sorry, I couldn't generate a response."
    except Exception as e:
        return f"An error occurred: {str(e)}"

async def run_chatbot():
    """Run the chatbot in a Streamlit interface."""
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