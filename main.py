from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

from src.agent import app

if __name__ == "__main__":
    result = app.invoke({
        "messages": [HumanMessage(content="Y a-t-il eu une anomalie de qualité de l'air à Paris le 6 mars 2026 ?")]
    })
    print(result["messages"][-1].content)
    breakpoint()