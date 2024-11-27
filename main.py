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

OPENING_MESSAGE = "Welcome to DataLuma! How may I assist you today?"
LLM = 'gemma:7b' 
EMBEDDING = "all-MiniLM-L6-v2"
RETRIEVED_CONTEXT = 4

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING)
try:
    chat_model = ChatOllama(model=LLM)
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChatOllama model '{LLM}': {str(e)}")

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVED_CONTEXT})

def combine_chat_history(chat_history, user_input):

    combined_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    return f"{combined_history}\nhuman: {user_input}"

def prepare_context(chat_history, retrieved_documents):
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    doc_text = "\n".join([doc.page_content for doc in retrieved_documents])
    return f"{history_text}\n\nDocuments:\n{doc_text}"

async def get_response_from_chain(user_input, chat_history):
    try:
        retrieved_documents = retriever.get_relevant_documents(user_input)
        context = prepare_context(chat_history, retrieved_documents)
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", rag_system_prompt),
            ("human", "{context}")  # Corrected to include 'context' as required
        ])
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
