import os

MCP_SERVER_CONFIG = {
    # Model configuration
    "routing_model": "openai/gpt-4-0125-preview",
    "execution_model": "openai/gpt-4-0125-preview",
    
    # Reasoner configuration
    "reasoner_config": {
        "default_strategy": "beam_search",
        "beam_width": 3,
        "num_simulations": 50,
        "max_thoughts": 3
    },
    
    "mcpServers": {
        "filesystem": {
            "command": "npm.cmd" if os.name == "nt" else "npm",
            "args": ["exec", "@modelcontextprotocol/server-filesystem", "--", "."],
            "description": "File system operations",
            "env": {}
        },
        "brave-search": {
            "command": "npm.cmd" if os.name == "nt" else "npm",
            "args": ["exec", "@modelcontextprotocol/server-brave-search", "--"],
            "description": "Web search operations",
            "env": {
                "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY"),
                "BRAVE_SEARCH_RATE_LIMIT": os.getenv("BRAVE_SEARCH_RATE_LIMIT", "10"),
                "BRAVE_SEARCH_RATE_WINDOW": os.getenv("BRAVE_SEARCH_RATE_WINDOW", "60")
            }
        },
        "mcp-reasoner": {
            "command": "node",
            "args": [os.path.join(os.path.dirname(__file__), "..", "..", "mcp-reasoner", "dist", "index.js")],
            "description": "Advanced reasoning with beam search and MCTS",
            "env": {
                "DEBUG": "mcp:*",
                "MCP_REASONER_STRATEGY": os.getenv("MCP_REASONER_STRATEGY", "beam_search"),
                "MCP_REASONER_BEAM_WIDTH": os.getenv("MCP_REASONER_BEAM_WIDTH", "3"),
                "MCP_REASONER_NUM_SIMULATIONS": os.getenv("MCP_REASONER_NUM_SIMULATIONS", "50")
            }
        }
    }
}