from dotenv import load_dotenv
from langchain.messages import HumanMessage

from src.agent import app

load_dotenv()

if __name__ == "__main__":
    result = app.invoke({
        "messages": [HumanMessage(content="Que s'est t-il passé le 1er mai ?")]
    })
    print(result["messages"])