import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory


os.environ["OPENAI_API_KEY"] = "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"

# 🤖 Prompt template for the assistant
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

# 💬 Chat model
llm = ChatOpenAI()

# 🧠 In-memory conversation store (session-based)
store: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 🧩 Chain with memory support
chain = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history=get_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 🗣️ Interactive chat loop
session_id = "user-1"
print("💬 LangChain Assistant (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        print("👋 Goodbye!")
        break
    response = chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
