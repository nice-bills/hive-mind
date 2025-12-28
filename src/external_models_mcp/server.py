from mcp.server.fastmcp import FastMCP
import litellm
import os
import sys
from typing import List, Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Set keys
litellm.api_key = os.environ.get("OPENROUTER_API_KEY")
litellm.groq_api_key = os.environ.get("GROQ_API_KEY")
litellm.huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")

# Initialize MCP
mcp = FastMCP("External Brain")

# --- CONFIGURATION ---

MODEL_ALIASES = {
    "glm": "openrouter/zhipu/glm-4-flash",
    "kimi": "openrouter/moonshotai/kimi-k2",
    "minimax": "openrouter/minimax/minimax-01",
    "kimi-k2": "groq/moonshotai/kimi-k2-instruct-0905",
    "hf-minimax": "huggingface/MiniMaxAI/MiniMax-M2.1",
    "hf-glm": "huggingface/zai-org/GLM-4.7",
    "hf-kimi-thinking": "huggingface/moonshotai/Kimi-K2-Thinking",
    "hf-kimi": "huggingface/moonshotai/Kimi-K2-Instruct-0905",
}

# Safety Limits
MAX_FILE_SIZE_BYTES = 512 * 1024 
MAX_TOTAL_CHARS = 400000     

def _resolve_model_name(alias: str) -> str:
    clean_alias = alias.lower().strip()
    return MODEL_ALIASES.get(clean_alias, alias)

def _read_context_files(file_paths: List[str]) -> str:
    """Reads files and wraps them in XML tags for better model parsing."""
    context_parts = []
    total_chars = 0
    
    for path_str in file_paths:
        try:
            path = Path(path_str).resolve()
            
            if not path.exists():
                print(f"[Warning] File not found: {path}", file=sys.stderr)
                continue
            
            if not path.is_file():
                continue

            # Skip binary files (quick check)
            with path.open("rb") as f:
                if b"\0" in f.read(1024):
                    print(f"[Warning] Skipping binary file: {path}", file=sys.stderr)
                    context_parts.append(f"<error file='{path}'>Skipped binary file</error>")
                    continue

            # Size check
            if path.stat().st_size > MAX_FILE_SIZE_BYTES:
                context_parts.append(f"<error file='{path}'>File too large (exceeds 512KB)</error>")
                continue

            content = path.read_text(encoding="utf-8", errors="replace")
            
            # Total context limit check
            if total_chars + len(content) > MAX_TOTAL_CHARS:
                context_parts.append(f"<error file='{path}'>Context limit reached. File truncated.</error>")
                break

            # Wrap in XML
            file_xml = f"<file path='{path}'>\n{content}\n</file>"
            context_parts.append(file_xml)
            total_chars += len(file_xml)
            
        except Exception as e:
            context_parts.append(f"<error file='{path_str}'>{str(e)}</error>")
            
    return "\n\n".join(context_parts)

@mcp.tool()
def ask_expert(prompt: str, model: str = "kimi-k2", context_files: List[str] = []) -> str:
    """
    Query an external 'Coding Expert' model with project context.
    
    Args:
        prompt: The task or question.
        model: Alias (e.g. 'kimi-k2', 'hf-glm', 'minimax'). Defaults to 'kimi-k2' (Groq).
        context_files: List of absolute file paths to include as context.
    """
    resolved_model = _resolve_model_name(model)
    print(f"[External Brain] Model: {resolved_model}", file=sys.stderr)
    
    # 1. Build Context
    context_xml = _read_context_files(context_files)
    
    # 2. Structure the prompt
    # We use a system message for context to keep it separated from the user's task
    messages = []
    
    if context_xml:
        messages.append({
            "role": "system", 
            "content": f"You are an expert software engineer. Below is the relevant project context provided in XML format:\n\n{context_xml}"
        })
    
    messages.append({"role": "user", "content": prompt})

    try:
        response = litellm.completion(
            model=resolved_model,
            messages=messages,
            temperature=0.2 # Lower temperature for better coding accuracy
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error using {resolved_model}: {str(e)}"

@mcp.tool()
def compare_experts(prompt: str, context_files: List[str] = [], experts: List[str] = ["kimi-k2", "hf-glm"]) -> str:
    """
    Get and compare coding solutions from two different experts.
    """
    results = []
    for alias in experts:
        res = ask_expert(prompt, model=alias, context_files=context_files)
        results.append(f"## Expert: {alias.upper()}\n\n{res}\n")
            
    return "\n---".join(results)

def _clean_code_block(content: str) -> str:
    """Removes markdown code fences if present."""
    content = content.strip()
    if content.startswith("```"):
        # Find the first newline to skip the language identifier (e.g. ```python)
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline+1:]
        
        # Remove the trailing ```
        if content.endswith("```"):
            content = content[:-3]
    return content.strip()

@mcp.tool()
def draft_editor(file_path: str, instruction: str, model: str = "kimi-k2", context_files: List[str] = []) -> str:
    """
    Ask an expert model to rewrite a file based on instructions.
    Saves the result to {file_path}.draft for review.
    
    Args:
        file_path: The file to edit.
        instruction: What changes to make.
        model: Expert alias (default: kimi-k2).
        context_files: Additional context files (optional).
    """
    resolved_model = _resolve_model_name(model)
    target_path = Path(file_path).resolve()
    
    if not target_path.exists():
        return f"Error: Target file {target_path} does not exist."
        
    print(f"[External Brain] Editing {target_path} with {resolved_model}...", file=sys.stderr)
    
    # Read the target file content
    try:
        current_content = target_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading target file: {str(e)}"
    
    # Prepare Context (Target file + Extras)
    # We treat the target file specially in the prompt
    additional_context = _read_context_files(context_files)
    
    system_prompt = (
        "You are an elite software engineer. You are rewriting a file to meet user requirements.\n"
        "Output ONLY the new content of the file. Do not output markdown code fences. Do not output conversational text.\n"
        f"Additional Context from other files:\n{additional_context}"
    )
    
    user_prompt = (
        f"--- ORIGINAL FILE: {target_path.name} ---\n"
        f"{current_content}\n"
        f"--- END ORIGINAL FILE ---\n\n"
        f"INSTRUCTION: {instruction}\n\n"
        "Rewrite the file strictly following the instruction. Maintain style and indentation."
    )

    try:
        response = litellm.completion(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1 # Very low temp for precision
        )
        
        # Process and Save
        new_content = _clean_code_block(response.choices[0].message.content)
        
        draft_path = target_path.with_suffix(target_path.suffix + ".draft")
        draft_path.write_text(new_content, encoding="utf-8")
        
        return f"Draft saved to: {draft_path}\nReview it and apply if correct."
        
    except Exception as e:
        return f"Error generating draft: {str(e)}"

def main():
    """Entry point for the package script."""
    mcp.run()

if __name__ == "__main__":
    main()
