import streamlit as st
import asyncio
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnableConfig
from prompts import contextualize_q_system_prompt, rag_system_prompt
import os
import pdfplumber
from langchain.docstore.document import Document


OPENING_MESSAGE = "Welcome in DataLUMA! How may I assist you today?"
EMBEDDING = "all-MiniLM-L6-v2"
RETRIEVED_CONTEXT = 4

# Initialize ChatOllama model with the correct model name
try:
    chat_model = ChatOllama(model="llama3.2:1b ", temperature=0.8, num_predict=256)
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChatOllama model: {str(e)}")

# Load embedding model and vectorstore
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
# Folder path to PDFs
PDF_FOLDER = "./data"

# Load PDF content into LangChain Document format
pdf_documents = [Document(**doc) for doc in load_pdfs_to_docs(PDF_FOLDER)]

# Add documents to Chroma
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
        # Retrieve relevant documents from the Chroma retriever (based on PDFs in 'data' folder)
        retrieved_documents = retriever.get_relevant_documents(user_input)

        # If no documents are retrieved, return a default message
        if not retrieved_documents:
            return "I'm sorry, I couldn't find an answer in the provided documents."

        # Prepare context by combining chat history and retrieved documents
        context = prepare_context(chat_history, retrieved_documents)

        # Define the prompt for the retrieval-augmented generation (RAG) chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", rag_system_prompt),  # System-level prompt
            ("human", "{context}")          # Context that includes chat history and retrieved documents
        ])

        # Create the QA chain and retrieval chain
        question_answer_chain = create_stuff_documents_chain(llm=chat_model, prompt=qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Generate the response asynchronously using the RAG chain
        response = ""
        async for chunk in rag_chain.astream(
            {"input": user_input, "context": context},
            config=RunnableConfig()
        ):
            if chunk.get("answer"):
                response += chunk["answer"]

        # If no response is generated, return a default message
        return response or "I'm sorry, I couldn't generate a response."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit chatbot UI
async def run_chatbot():
    """Run the chatbot in a Streamlit interface."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Chatbot Assistant")
    st.write(OPENING_MESSAGE)

    user_input = st.text_input("Ask me anything:", "")
    if user_input:
        # Combine chat history and user input for context
        combined_input = combine_chat_history(st.session_state.chat_history, user_input)

        # Get response from the chatbot chain
        response = await get_response_from_chain(combined_input, st.session_state.chat_history)

        # Display the response
        st.write(response)

        # Update chat history
        st.session_state.chat_history.append({"role": "human", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    asyncio.run(run_chatbot())
