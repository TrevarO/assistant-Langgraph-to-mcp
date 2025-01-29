import asyncio
import time
import logging
import contextlib
from typing import Dict, Any, AsyncGenerator
from mcp import ClientSession, StdioServerParameters
from langchain.agents import create_react_agent
from langchain_core.tools import BaseTool
from .mcp_wrapper import MCPTool, create_tool_from_mcp
from .utils import load_chat_model

logger = logging.getLogger(__name__)

class ServerManager:
    def __init__(self):
        self.servers: Dict[str, asyncio.Task] = {}
        self.processes: Dict[str, asyncio.subprocess.Process] = {}
    
    async def create_server_process(self, name: str, cmd: list[str], env: dict) -> asyncio.subprocess.Process:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        self.processes[name] = process
        return process

    async def add_server(self, name: str, server_task: asyncio.Task) -> asyncio.Task:
        if name in self.servers:
            old_task = self.servers[name]
            if not old_task.done():
                old_task.cancel()
                try:
                    await old_task
                except asyncio.CancelledError:
                    pass

        def cleanup_callback(task):
            try:
                task.result()
            except Exception as e:
                logger.error(f"Server {name} failed with error: {e}")
            finally:
                if name in self.servers:
                    del self.servers[name]

        server_task.add_done_callback(cleanup_callback)
        self.servers[name] = server_task
        return server_task
        
    async def shutdown(self):
        try:
            # First terminate processes
            for process in self.processes.values():
                if process.returncode is None:
                    process.terminate()
            
            # Wait for processes with timeout
            try:
                await asyncio.wait([process.wait() for process in self.processes.values()], timeout=2.0)
            except asyncio.TimeoutError:
                pass

            # Cancel server tasks
            for task in self.servers.values():
                if not task.done():
                    task.cancel()
            
            # Wait for tasks with timeout
            try:
                await asyncio.wait(list(self.servers.values()), timeout=2.0)
            except asyncio.TimeoutError:
                pass

            # Allow time for transports to close
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            for process in self.processes.values():
                try:
                    if process.returncode is None:
                        process.kill()
                except:
                    pass
            self.processes.clear()
            self.servers.clear()

@contextlib.asynccontextmanager
async def manage_event_loop() -> AsyncGenerator[asyncio.AbstractEventLoop, None]:
    logger.info("Event loop setup complete")
    try:
        loop = asyncio.get_event_loop()
        yield loop
    except Exception as e:
        logger.error(f"Error in event loop: {e}")
        raise
    finally:
        logger.info("Starting event loop cleanup")
        try:
            await server_manager.shutdown()
        except Exception as e:
            logger.error(f"Error during event loop cleanup: {e}")
        logger.info("Event loop cleanup complete")

server_manager = ServerManager()
__all__ = ['server_manager', 'manage_event_loop', 'create_workflow']