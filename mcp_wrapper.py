import os
import logging
import asyncio
import time
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool
from mcp import ClientSession, StdioServerParameters, stdio_client
import nest_asyncio

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, rate_limit: int = 10, window: int = 60):
        self.rate_limit = rate_limit
        self.window = window
        self.tokens = rate_limit
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(
                self.rate_limit,
                self.tokens + time_passed * (self.rate_limit / self.window)
            )
            self.last_update = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) * (self.window / self.rate_limit)
                await asyncio.sleep(wait_time)
                self.tokens = 0
                self.last_update = time.time()
            else:
                self.tokens -= 1

class ResultCache:
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp <= self.ttl:
                return value
            del self.cache[key]
        return None
    
    def set(self, key: str, value: str):
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.items(), key=lambda x: x[1][1])
            del self.cache[oldest[0]]
        self.cache[key] = (value, time.time())

class MCPTool(BaseTool):
    name: str
    description: str
    config: Dict[str, Any]
    current_thought_number: int = 1
    total_thoughts: int = 3
    _rate_limiter: Optional[RateLimiter] = None
    _result_cache: Optional[ResultCache] = None
    
    async def run_tool(self, **kwargs):
        try:
            # Parameter validation
            if hasattr(self, 'schema') and hasattr(self.schema, 'required'):
                for param in self.schema.required:
                    if param not in kwargs:
                        if param == 'path':
                            kwargs['path'] = '.'
                        else:
                            kwargs[param] = None

            # Check cache for search queries
            if self.name == 'brave-search':
                cache_key = f"{self.name}:{kwargs.get('query', '')}"
                if self._result_cache and (cached := self._result_cache.get(cache_key)):
                    return cached

            # Rate limit handling
            if self._rate_limiter:
                await self._rate_limiter.acquire()

            # Retry logic with exponential backoff
            max_retries = 3
            retry_delay = 1
            for attempt in range(max_retries):
                try:
                    result = await super().run_tool(**kwargs)
                    
                    # Cache successful search results
                    if self.name == 'brave-search' and self._result_cache:
                        self._result_cache.set(cache_key, result)
                    
                    return result
                except Exception as e:
                    if 'Rate limit exceeded' in str(e) and attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        await asyncio.sleep(wait_time)
                        continue
                    raise
        finally:
            if hasattr(self, '_cleanup_state'):
                await self._cleanup_state()

    def _preprocess_reasoner_args(self, **kwargs) -> Dict[str, Any]:
        """Prepare arguments for the MCP reasoner"""
        if self.name == "mcp-reasoner":
            query = kwargs.get("query", "")
            args = {
                "thought": f"Analyzing: {query}" if self.current_thought_number == 1 else kwargs.get("thought", "Continuing analysis..."),
                "thoughtNumber": self.current_thought_number,
                "totalThoughts": self.total_thoughts,
                "nextThoughtNeeded": self.current_thought_number < self.total_thoughts,
                "strategyType": kwargs.get("strategy_type", "beam_search")
            }
            self.current_thought_number += 1
            return args
        return kwargs
    
    def _run(self, **kwargs) -> str:
        logger.info(f"Running {self.name} with args: {kwargs}")
        try:
            processed_args = self._preprocess_reasoner_args(**kwargs)
            logger.info(f"Processed args: {processed_args}")
            
            server_params = StdioServerParameters(
                command=self.config["command"],
                args=self.config["args"],
                env={**os.environ, **(self.config.get("env") or {})}
            )
            
            async def run_tool():
                logger.info("Starting MCP client session")
                try:
                    async with stdio_client(server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            tools = await session.list_tools()
                            logger.info(f"Available tools: {tools}")
                            result = await session.call_tool(self.name, arguments=processed_args)
                            if result.isError:
                                raise Exception(result.content)
                            return result.content
                except Exception as e:
                    logger.error(f"Error in MCP session: {e}")
                    raise

            # Set up event loop with proper error handling
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                nest_asyncio.apply()
            
            try:
                result = loop.run_until_complete(run_tool())
                # Check if we need to reset thought counter
                if self.name == "mcp-reasoner" and self.current_thought_number > self.total_thoughts:
                    self.reset_thought_counter()
                return str(result)
            finally:
                # Proper cleanup
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                
        except Exception as e:
            logger.error(f"Error running {self.name}: {e}", exc_info=True)
            # Reset on error for reasoner
            if self.name == "mcp-reasoner":
                self.reset_thought_counter()
            return f"Error executing {self.name}: {str(e)}"
    
    def reset_thought_counter(self):
        """Reset the thought counter for new reasoning chains"""
        self.current_thought_number = 1

def create_tool_from_mcp(name: str, config: Dict[str, Any]) -> list[BaseTool]:
    logger.info(f"Creating tools for server: {name}")
    server_params = StdioServerParameters(
        command=config["command"],
        args=config["args"],
        env={**os.environ, **(config.get("env") or {})}
    )

    async def fetch_tools():
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await get_server_tools(session)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        nest_asyncio.apply()
    
    try:
        tools = loop.run_until_complete(fetch_tools())
    finally:
        if not loop.is_running():
            loop.close()

    created_tools = []
    for tool_name in tools:
        tool = MCPTool(
            name=tool_name,
            description=f"Access {tool_name} via {name}",
            config=config
        )
        
        # Configure rate limiting for search tools
        if name == "brave-search":
            tool._rate_limiter = RateLimiter(
                rate_limit=int(os.getenv("BRAVE_SEARCH_RATE_LIMIT", "10")),
                window=int(os.getenv("BRAVE_SEARCH_RATE_WINDOW", "60"))
            )
            tool._result_cache = ResultCache()
            
        # Special handling for reasoner tool
        if tool_name == "mcp-reasoner":
            tool.description = """Advanced reasoning tool with multiple strategies including Beam Search and Monte Carlo Tree Search.
            Parameters:
            - query: The question or topic to analyze
            - strategy_type: 'beam_search' or 'mcts' (default: beam_search)
            - beam_width: Number of paths to explore (1-10, default: 3)
            - num_simulations: Number of MCTS simulations (1-150, default: 50)
            """
        created_tools.append(tool)
    
    return created_tools

async def get_server_tools(session) -> list[str]:
    tools = await session.list_tools()
    return [tool.name for tool in tools.tools]