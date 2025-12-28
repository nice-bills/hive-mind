# External Models MCP

A Model Context Protocol (MCP) server that empowers AI agents with a "second brain". This extension enables the Gemini CLI (and other MCP clients) to seamlessly delegate coding, reasoning, and brainstorming tasks to external state-of-the-art models like Kimi K2 (Moonshot AI), GLM-4 (Zhipu AI), and Minimax.

It acts as a local bridge, allowing your primary agent to consult specialized models for specific tasks, effectively creating a "squad" of expert AI developers.

## Features

- **Multi-Provider Support**: Seamlessly connects to Groq, OpenRouter, and Hugging Face Inference API via a unified interface.
- **Smart Context Injection**: Automatically reads local project files, validates them for safety (binary detection, size limits), and formats them into XML structures that coding models understand best.
- **Role-Based Routing**: Define specific aliases (e.g., 'kimi' for coding, 'glm' for reasoning) to route tasks to the best model for the job.
- **Expert Comparison**: The `compare_experts` tool allows you to run the same prompt against multiple models simultaneously to verify solutions and get diverse perspectives.
- **Security First**: Runs entirely locally. API keys are managed via `.env` and are never exposed in the terminal command history.

## Installation

You can install and run this server directly using `uv`.

### 1. Install via `uv tool` (Recommended)

```bash
uv tool install .
```
*(Run this command from inside the project directory)*

### 2. Configure API Keys

Create a `.env` file in the directory where you will run the server (or set these as system environment variables):

```bash
OPENROUTER_API_KEY=sk-or-...
GROQ_API_KEY=gsk_...
HUGGINGFACE_API_KEY=hf_...
```

## Usage with Gemini CLI

Once installed, you can register the server in your Gemini CLI configuration using the exposed command `external-models-server`.

```bash
gemini mcp add external-brain --command "external-models-server"
```

### Development / Manual Run

If you are modifying the code, you can run it directly from the source:

```bash
uv run external_models_mcp.server
```

## Configuration

The server is pre-configured with aliases for high-performance coding models. You can modify `src/external_models_mcp/server.py` to add or change aliases.

**Default Aliases:**
- `kimi-k2` (Groq): Moonshot AI's Kimi K2 (Optimized for coding)
- `glm` (OpenRouter): Zhipu AI's GLM-4 Flash
- `minimax` (OpenRouter): Minimax abab6.5s
- `hf-glm` (Hugging Face): GLM-4-9b-Chat via Inference API

### Example Prompts

**Ask a specific expert:**
"Use `ask_expert` with model='kimi-k2' to refactor `src/main.py` for better error handling."

**Compare solutions:**
"Use `compare_experts` with experts=['kimi-k2', 'hf-glm'] to write a Python function that implements A* search algorithm."

## License

MIT