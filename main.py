from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from src.agent import AgentModel, build_agent

if __name__ == "__main__":
    app = build_agent(model=AgentModel.LLAMA)
    result = app.invoke({
        "messages": [HumanMessage(content="Y a-t-il eu une anomalie de qualité de l'air à Paris le 6 mars 2026 ?")]
    })
    print(result["messages"][-1].content)