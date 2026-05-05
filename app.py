# app.py

from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agent import app

load_dotenv()


def get_agent_response(question: str, history: list[dict]) -> str:
    """Invokes the agent and returns the answer."""
    messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in history
    ]
    messages.append(HumanMessage(content=question))
    result = app.invoke({"messages": messages})
    answer = result["messages"][-1].content
    if isinstance(answer, list):
        answer = " ".join(b.get("text", "") for b in answer if isinstance(b, dict))
    return answer


st.title("🌫️ Air Quality Agent — Paris IDF")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask about air quality in Paris..."):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_agent_response(question, st.session_state.messages[:-1])
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})