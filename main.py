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
    How to merge message lists
    """ 
    messages: Annotated[list[BaseMessage], add_messages]


# label graph nodes
REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    """Node function"""
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}  # set to a single-item list containing that result.


def reflection_node(state: MessageGraph):
    """Node function"""
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}  # wraps the returned content into a HumanMessage


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) == 2:
        return END
    return REFLECT


builder.add_conditional_edges(  # arista condicional, si tiene que ir a reflect o a end
    source=GENERATE,
    path=should_continue, 
    path_map={END:END, REFLECT:REFLECT}
)
builder.add_edge(REFLECT, GENERATE)  # si no existe esta arista, no se genera la arista de reflect a generate

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()
graph.get_graph().draw_mermaid_png(output_file_path='flow.png')


if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)
