# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# ------------------------------------------------------------------
# Create model
# ------------------------------------------------------------------
model = ChatOllama(
    model="llama3.2:3b-instruct-q8_0",
    base_url="http://a03a-216-147-123-78.ngrok-free.app",
    temperature=0.8,
    num_ctx=16384,
)

# ------------------------------------------------------------------
# Interact with model without memory
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Add message persistence
# ------------------------------------------------------------------
# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
async def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


# Define the (singal) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Add config
config = {"configurable": {"thread_id": "abc123"}}

query = "Hello, my name is liberaci."

input_messages = [HumanMessage(query)]
output = await app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
