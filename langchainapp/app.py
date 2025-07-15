import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Replace with your key or load from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]  

# LangChain setup
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])
llm = ChatOpenAI()

store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history=get_history,
    input_messages_key="input",
    history_messages_key="history"
)

# ✅ Streamlit chat interface
st.set_page_config(page_title="LangChain Chat", layout="centered")
st.title("💬 Chat with AI")

if "session_id" not in st.session_state:
    st.session_state.session_id = "user-web"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Chat input textbox at bottom
user_input = st.chat_input("Say something...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": st.session_state.session_id}}
    )
    st.session_state.chat_history.append(("ai", response.content))

# ✅ Display chat history with bubbles
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
