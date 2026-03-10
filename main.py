from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from chains import generate_chain, reflect_chain


load_dotenv()


class MessageGraph(TypedDict):
    """ 
    add_messages: Merges two lists of messages, updating existing messages by ID.
    How to merge message lists.
    The add_messages annotation implies the graph knows how to merge or update messages 
    across states (useful when nodes return lists).
    """ 
    messages: Annotated[list[BaseMessage], add_messages]


# label graph nodes
REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    """Node function
    Use "generate_chain = generation_prompt | llm" with generation_prompt
    """
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}  # set to a single-item list containing that result.


def reflection_node(state: MessageGraph):
    """Node function"""
    res = reflect_chain.invoke({"messages": state["messages"]})
    # Message from the user. A HumanMessage is a message that is passed in from a user to the model.
    return {"messages": [HumanMessage(content=res.content)]}  # wraps the returned content into a HumanMessage


# Conditional flow control
def should_continue(state: MessageGraph):
    if len(state["messages"]) == 2:
        return END  # terminate
    return REFLECT  # continue


# Graph building: The graph expects(esperan) node functions to return partial state dicts like {"messages": [...]} to update the state.
builder = StateGraph(state_schema=MessageGraph)  # Creates a StateGraph with MessageGraph as the state schema
builder.add_node(GENERATE, generation_node)  # Adds the GENERATE and REFLECT nodes using the corresponding node functions
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)  # Sets the entry point to GENERATE.



# Adds a conditional edge from GENERATE using should_continue, mapping the returned keys to the 
# corresponding targets (END or REFLECT).
builder.add_conditional_edges(  # arista condicional, si tiene que ir a reflect o a end
    source=GENERATE,
    path=should_continue, 
    path_map={END:END, REFLECT:REFLECT}
)

# Adds a direct edge from REFLECT back to GENERATE, creating a loop: GENERATE -> (maybe REFLECT) -> GENERATE ...
builder.add_edge(REFLECT, GENERATE)  # si no existe esta arista, no se genera la arista de reflect a generate

graph = builder.compile()  # Compiles the StateGraph into a CompiledStateGraph object. The compiled graph can be invoked, streamed, batched, and run asynchronously.
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()
graph.get_graph().draw_mermaid_png(output_file_path='flow.png')


"""
Overall behavior / execution flow

1. Start at GENERATE: generate_chain processes the current messages.
2. After GENERATE, should_continue checks message count:
    If there are exactly 2 messages → go to END (stop) and return result.
    Otherwise → go to REFLECT.
    At REFLECT: reflect_chain runs and its output is wrapped as a HumanMessage and fed back into the graph.
The explicit edge REFLECT -> GENERATE causes another generation step with the updated messages.
This loop repeats until the messages list contains 2 items, then the graph ends and the final response is printed.

"""

if __name__ == "__main__":
    inputs = {
        "messages": [
            HumanMessage(  # Message from the user. A HumanMessage is a message that is passed in from a user to the model.
                content="""Make this tweet better:"
            @LangChainAI
            — newly Tool Calling feature is seriously underrated.
            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.
            Made a video covering their newest blog post""")
        ]
    }
    response = graph.invoke(inputs)  # Invokes the compiled graph. Run the graph with a single input and config.
    print(response)
