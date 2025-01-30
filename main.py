import os
import signal
import asyncio
import platform
from datetime import datetime
from typing import Dict, Any, Set
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.langgraph_mcp.assistant_graph import graph, GraphState, create_workflow
from src.langgraph_mcp.server_manager import server_manager, manage_event_loop
from src.langgraph_mcp.config import MCP_SERVER_CONFIG
from src.langgraph_mcp.logging_config import setup_logging

load_dotenv()
logger = setup_logging()

# Track active tasks
active_tasks: Set[asyncio.Task] = set()

async def start_mcp_server(name: str, config: Dict[str, Any]) -> asyncio.Task:
    try:
        cmd = [config["command"]] + config["args"]
        env = {**os.environ, **config.get("env", {})}
        process = await server_manager.create_server_process(name, cmd, env)
        
        async def run_server():
            try:
                await process.wait()
            except asyncio.CancelledError:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        process.kill()
            finally:
                if process.returncode is None:
                    process.kill()
        
        server_task = asyncio.create_task(run_server())
        active_tasks.add(server_task)
        server_task.add_done_callback(active_tasks.discard)
        
        return await server_manager.add_server(name, server_task)
    except Exception as e:
        logger.error(f"Failed to start MCP server {name}: {e}")
        raise

async def cleanup_servers():
    """Cleanup function for graceful shutdown"""
    try:
        if hasattr(server_manager, 'shutdown'):
            await server_manager.shutdown()
        
        remaining_tasks = [t for t in active_tasks if not t.done()]
        if remaining_tasks:
            for task in remaining_tasks:
                task.cancel()
            try:
                await asyncio.wait(remaining_tasks, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not complete within timeout")
    except Exception as e:
        logger.error(f"Error during server shutdown: {e}")

def handle_shutdown(loop, signal_handler=None):
    """Handle shutdown across different platforms"""
    logger.info("Initiating shutdown...")
    loop.stop()

async def main():
    loop = asyncio.get_event_loop()
    
    # Setup platform-specific signal handling
    if platform.system() != 'Windows':
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(loop))
    
    async with manage_event_loop() as loop:
        try:
            # Start MCP servers
            servers = []
            for name, config in MCP_SERVER_CONFIG["mcpServers"].items():
                try:
                    server = await start_mcp_server(name, config)
                    servers.append(server)
                    logger.info(f"Started {name} server successfully")
                except Exception as e:
                    logger.error(f"Failed to start {name} server: {e}")
            
            if not servers:
                logger.error("No servers started successfully")
                return
            
            # Initialize agent and state
            workflow = create_workflow(MCP_SERVER_CONFIG)
            
            # Initialize state with proper structure
            try:
                # Create initial state
                state = GraphState.create_initial()

                # Validate initial state
                if not GraphState.validate(state):
                    raise ValueError("Invalid initial state structure")

                # Initialize workflow
                workflow = create_workflow(MCP_SERVER_CONFIG)
                
                # Add system message to state if needed
                if not any(isinstance(msg, SystemMessage) for msg in state["messages"]):
                    system_msg = SystemMessage(content="""You are a helpful AI assistant. You can engage in conversations and use various tools to assist users with their questions and tasks.""")
                    state["messages"].insert(0, system_msg)

            except Exception as e:
                logger.error(f"State initialization error: {str(e)}", exc_info=True)
                raise
            
            print("\nAI Assistant ready. Type 'exit' to quit.\n")
            
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    if user_input.lower() in ['exit', 'quit', 'q', '']:
                        break
                    
                    start_time = datetime.now()
                    
                    try:
                        # Create user message
                        user_message = HumanMessage(content=user_input)
                        
                        # Create new state with validation
                        current_messages = state["messages"]
                        new_state = GraphState(
                            messages=current_messages + [user_message],
                            structured_response=state["structured_response"],
                            active_tools=state.get("active_tools", [])
                        )
                        
                        # Validate new state (add this)
                        if not GraphState.validate(new_state):
                            raise ValueError("Invalid state after adding user message")
                        
                        # Invoke workflow
                        response = await workflow.ainvoke(new_state)
                        
                        # Validate response (add this)
                        if not GraphState.validate(response):
                            raise ValueError("Invalid response state from workflow")
                        
                        # Update state
                        state = response
                        
                        # Display latest AI message
                        if (
                            isinstance(response, dict) 
                            and "messages" in response 
                            and response["messages"]
                            and len(response["messages"]) > 0
                        ):
                            print(f"\nAssistant: {response['messages'][-1].content}")
                            
                            # Handle structured response if present
                            structured_response = response.get("structured_response")
                            if structured_response:
                                if structured_response.get("error"):
                                    logger.error(f"Structured error: {structured_response['error']}")
                                elif structured_response.get("tool_outputs"):
                                    for output in structured_response["tool_outputs"]:
                                        logger.debug(f"Tool output: {output}")
                        
                    except Exception as e:
                        logger.error(f"Error processing request: {e}")
                        print(f"\nError: {str(e)}")
                        
                        # Create error state with validation
                        error_state = GraphState(
                            messages=current_messages + [
                                HumanMessage(content=user_input),
                                AIMessage(content=f"Error: {str(e)}")
                            ],
                            structured_response={
                                "response": f"Error: {str(e)}",
                                "tool_outputs": [],
                                "error": str(e)
                            },
                            active_tools=state.get("active_tools", [])
                        )
                        
                        # Validate error state
                        if not GraphState.validate(error_state):
                            logger.error("Failed to create valid error state")
                            continue
                            
                        state = error_state
                    
                    print(f"\nTime: {(datetime.now() - start_time).total_seconds():.2f}s")
                    
                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt")
                    break
                except Exception as e:
                    logger.error(f"Main loop error: {e}")
                    continue
                    
        finally:
            await cleanup_servers()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)