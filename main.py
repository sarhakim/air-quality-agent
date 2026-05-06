from dotenv import load_dotenv
from langchain.messages import HumanMessage

from src.agent import app

load_dotenv()

if __name__ == "__main__":
    result = app.invoke({
        "messages": [HumanMessage(content="Y a-t-il eu une anomalie le 8 mars 2026 ?")]
    })
    print(result["messages"])