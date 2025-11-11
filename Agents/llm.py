import os
import json
import subprocess
from types import SimpleNamespace

# Lightweight Ollama wrapper: attempts to use the Python client if installed,
# and falls back to calling the `ollama` CLI. It exposes a simple
# `invoke(prompt)` method which returns an object with a `content` attribute
# to keep compatibility with existing code that does `LLM.invoke(...).content`.

try:
    from ollama import Ollama
    _OLLAMA_AVAILABLE = True
    _OLLAMA_CLIENT = Ollama()
except Exception:
    _OLLAMA_AVAILABLE = False
    _OLLAMA_CLIENT = None


class OllamaLLM:
    def __init__(self, model: str | None = None):
        # Default model can be overridden with the OLLAMA_MODEL env var.
        # Pick a conservative default; user can set a model they have locally.
        self.model = model or os.getenv("OLLAMA_MODEL", "mistral")

    def invoke(self, prompt: str, **kwargs):
        """Invoke the Ollama model and return an object with a `content` field.

        This method first tries the Python `ollama` client (if installed).
        If that fails it will call the `ollama` CLI via subprocess. Any
        exceptions are returned inside the `.content` for easier debugging.
        """
        # Try Python client
        if _OLLAMA_AVAILABLE and _OLLAMA_CLIENT is not None:
            try:
                # Many client versions provide `chat` or `generate` APIs; try both.
                if hasattr(_OLLAMA_CLIENT, "chat"):
                    # chat usually expects model + messages
                    resp = _OLLAMA_CLIENT.chat(self.model, messages=[{"role": "user", "content": prompt}])
                    # resp may be a dict-like or object
                    if isinstance(resp, dict):
                        content = resp.get("content") or resp.get("text") or json.dumps(resp)
                    else:
                        content = getattr(resp, "content", getattr(resp, "text", str(resp)))
                    return SimpleNamespace(content=content)

                if hasattr(_OLLAMA_CLIENT, "generate"):
                    resp = _OLLAMA_CLIENT.generate(self.model, prompt)
                    content = getattr(resp, "text", str(resp))
                    return SimpleNamespace(content=content)
            except Exception as _e:
                # fall through to CLI fallback
                pass

        # Fallback: use `ollama run <model>` via subprocess; pass prompt on stdin
        try:
            cmd = ["ollama", "run", self.model, "--no-stream"]
            proc = subprocess.run(cmd, input=prompt.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            output = proc.stdout.decode().strip()
            return SimpleNamespace(content=output)
        except Exception as e:
            return SimpleNamespace(content=f"ERROR: Ollama invocation failed: {e}")


# Export a ready-to-use LLM instance
LLM = OllamaLLM()

# Example (uncomment to test manually):
# print(LLM.invoke("Hello").content)