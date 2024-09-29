# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import add_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict

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
def call_model(state: MessagesState):
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
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    chain = prompt | model
    response = chain.invoke(state)
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc345"}}
query = "Yo! I'm Sebastian."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()


query = "What is my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

# ------------------------------------------------------------------
# Make prompt template a bit more complicated
# ------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


workflow = StateGraph(state_schema=State)


def call_model(state: State):
    chain = prompt | model
    response = chain.invoke(state)
    return {"messages": [response]}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc456"}}
query = "Hi! I'm Jimmy."
language = "Spanish"

input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()

query = "What is my name?"

input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()
