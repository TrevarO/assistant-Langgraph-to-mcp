import logging
from typing import TypedDict, List, Dict, Any, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from .utils import load_chat_model
from .mcp_wrapper import create_tool_from_mcp

logger = logging.getLogger(__name__)

# Define state types
class GraphState(TypedDict):
    """State definition for the agent graph."""
    messages: Sequence[BaseMessage]
    structured_response: Dict[str, Any]
    active_tools: List[str]  # Add this line

    @classmethod
    def create_initial(cls) -> 'GraphState':
        """Create initial state."""
        return cls(
            messages=[],
            structured_response={
                "response": "",
                "tool_outputs": [],
                "error": None
            },
            active_tools=[]  # Add this line
        )

    @classmethod
    def validate(cls, state: Dict) -> bool:
        """Validate state structure."""
        try:
            required_keys = {"messages", "structured_response", "active_tools"}
            
            # Only log validation issues at ERROR level
            if not isinstance(state, dict):
                logger.error("State is not a dictionary")
                return False
                
            if missing := required_keys - state.keys():
                logger.error(f"Missing required keys: {missing}")
                return False

            # Check types silently unless there's an error
            is_valid = (
                isinstance(state["messages"], (list, tuple))
                and isinstance(state["structured_response"], dict)
                and isinstance(state["active_tools"], list)
            )

            if not is_valid:
                logger.error(f"Invalid types in state: messages={type(state['messages'])}, "
                            f"structured_response={type(state['structured_response'])}, "
                            f"active_tools={type(state['active_tools'])}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False


def create_workflow(config: Dict[str, Any]):
    """Create the agent workflow."""
    # Initialize tools
    tools = []
    for name, cfg in config["mcpServers"].items():
        try:
            server_tools = create_tool_from_mcp(name, cfg)
            tools.extend(server_tools)
            logger.info(f"Created {len(server_tools)} tools from {name}")
        except Exception as e:
            logger.error(f"Error creating tools for {name}: {e}")
    
    # Load model
    model = load_chat_model(config["routing_model"])
    
    # Create base agent
    agent = create_react_agent(
        model=model,
        tools=tools
    )
    
    # Create graph
    workflow = StateGraph(GraphState)
    
    # Define agent step

    async def agent_step(state: GraphState):
        try:
            messages = state["messages"]
            latest_message = messages[-1] if messages else None
            
            if not latest_message:
                logger.debug("No latest message, returning initial state")
                return {
                    "messages": messages,
                    "structured_response": {
                        "response": "",
                        "tool_outputs": [],
                        "error": None
                    },
                    "active_tools": state.get("active_tools", [])
                }

            # Updated system message with better search and extraction guidance
            system_message = SystemMessage(content="""You are a helpful AI assistant equipped with various tools.
    For queries requiring real-time information, follow these steps:

    1. IDENTIFY THE TYPE OF DATA:
    - Weather: Search for "current weather [location]" + include temperature and conditions
    - Stocks: Search for "[symbol] stock price current" + include price and change
    - News: Search for "[topic] news last 24 hours" + include latest developments
    - General: Format query for most recent, relevant results

    2. USE THE BRAVE SEARCH TOOL:
    Question: What information is needed?
    Thought: Consider what specific data to extract
    Action: brave_web_search
    Action Input: {"query": "specific targeted search"}
    Observation: <analyze search results>
    
    3. EXTRACT & SUMMARIZE:
    - Pull out specific numbers, facts, or updates
    - Include source attribution
    - Format information clearly
    - Provide context where helpful
    
    4. IF NEEDED, DO FOLLOW-UP SEARCHES:
    - For additional context
    - For verification
    - For related information

    Always provide actual data and facts, not just URLs or references.""")

            # Debug pre-invoke state
            logger.debug(f"Processing query: {latest_message.content}")
            
            try:
                # Transform input for agent
                agent_input = {
                    "messages": [system_message] + messages[:-1] + [latest_message],
                    "structured_response": {
                        "response": "",
                        "tool_outputs": [],
                    },
                    "input": latest_message.content,
                    "chat_history": messages[:-1],
                    "metadata": {
                        "active_tools": state.get("active_tools", [])
                    }
                }

                # Debug the search process
                logger.debug(f"Agent input prepared")
                
                response = await agent.ainvoke(agent_input)
                
                # Debug search results
                if isinstance(response, dict) and "intermediate_steps" in response:
                    for step in response.get("intermediate_steps", []):
                        logger.debug(f"Search result snippet: {step[1][:200]}...")  # First 200 chars

            except Exception as agent_error:
                logger.error(f"Agent invoke error: {str(agent_error)}")
                raise

            # Extract and format response
            output_text = ""
            if isinstance(response, dict):
                if "output" in response:
                    output_text = response["output"]
                elif "messages" in response and response["messages"]:
                    output_text = response["messages"][-1].content
                else:
                    output_text = str(response.get("response", "No response generated"))
            else:
                output_text = str(response)

            # Create message and return state
            ai_message = AIMessage(content=output_text)
            updated_state = {
                "messages": messages + [ai_message],
                "structured_response": {
                    "response": output_text,
                    "tool_outputs": response.get("intermediate_steps", []) if isinstance(response, dict) else [],
                    "error": None
                },
                "active_tools": response.get("active_tools", state.get("active_tools", []))
            }

            return updated_state

        except Exception as e:
            logger.error(f"Error in agent_step: {str(e)}")
            error_msg = f"Error: {str(e)}"
            
            return {
                "messages": messages + [AIMessage(content=error_msg)],
                "structured_response": {
                    "response": error_msg,
                    "tool_outputs": [],
                    "error": str(e)
                },
                "active_tools": state.get("active_tools", [])
        }

    # Add node and compile
    workflow.add_node("agent", agent_step)
    workflow.set_entry_point("agent")  # Remove initial_state parameter
    workflow.add_edge("agent", END)    # Add this line
    return workflow.compile() 


# Initialize graph
graph = create_workflow({
    "routing_model": "openai/gpt-4-0125-preview",
    "mcpServers": {}
})