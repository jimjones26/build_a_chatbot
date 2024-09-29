# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# ------------------------------------------------------------------
# Create model
# ------------------------------------------------------------------
model = ChatOllama(
    model="llama3.2:3b-instruct-q8_0",
    base_url="http://a03a-216-147-123-78.ngrok-free.app",
    temperature=0.8,
    num_ctx=16384,
)

model.invoke([HumanMessage(content="Hi! I'm Jimmy")])

model.invoke([HumanMessage(content="What's my name?")])

model.invoke(
    [
        HumanMessage(content="Hi! I'm Jimmy"),
        AIMessage(content="Hello Jimmy! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
        AIMessage(content="Your name is Jimmy! How can I assist you today?"),
        HumanMessage(content="What is the Etymology of my name?"),
    ]
)
