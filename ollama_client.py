from __future__ import annotations

import time
from typing import Any

import requests

from config import OLLAMA_BASE_URL, MODEL_NAME, EMBED_MODEL_NAME, NUM_CTX

_CONN_HELP = (
    "\n\nOllama is not reachable at {url}.\n"
    "Fix:\n"
    "  1. Start Ollama:          ollama serve\n"
    "  2. Pull the LLM model:    ollama pull {model}\n"
    "  3. Pull the embed model:  ollama pull {embed_model}\n"
    "Then re-run this script.\n"
)

class OllamaError(RuntimeError):
    pass

def _check_connection(base_url: str = OLLAMA_BASE_URL) -> None:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise OllamaError(
            _CONN_HELP.format(
                url=base_url, model=MODEL_NAME, embed_model=EMBED_MODEL_NAME
            )
        )
    except requests.exceptions.Timeout:
        raise OllamaError(f"Ollama at {base_url} timed out. Is it overloaded?")

def _strip_latest(name: str) -> str:
    suffix = ":latest"
    return name[: -len(suffix)] if name.endswith(suffix) else name

def check_models_available(
    model: str = MODEL_NAME,
    embed_model: str = EMBED_MODEL_NAME,
    base_url: str = OLLAMA_BASE_URL,
) -> None:
    _check_connection(base_url)
    r = requests.get(f"{base_url}/api/tags", timeout=10)
    raw_names = {m["name"] for m in r.json().get("models", [])}

    def _is_available(wanted: str) -> bool:
        if wanted in raw_names:
            return True
        wanted_base = _strip_latest(wanted)
        for name in raw_names:
            name_base = _strip_latest(name)
            if (
                wanted_base == name_base
                or name.startswith(wanted + ":")
                or wanted.startswith(name + ":")
            ):
                return True
        return False

    missing = []
    if not _is_available(model):
        missing.append(f"  ollama pull {model}")
    if not _is_available(embed_model):
        missing.append(f"  ollama pull {embed_model}")

    if missing:
        raise OllamaError(
            "Required model(s) not found locally.\nRun:\n" + "\n".join(missing)
        )

def generate(
    prompt: str,
    model: str = MODEL_NAME,
    num_ctx: int = NUM_CTX,
    temperature: float = 0.0,
    use_json_format: bool = False,
    num_predict: int | None = None,
    base_url: str = OLLAMA_BASE_URL,
    retries: int = 2,
    backoff: float = 3.0,
    timeout: int = 120,
) -> str:
    if num_predict is None:
        num_predict = 600 if use_json_format else 250

    options: dict[str, Any] = {
        "num_ctx": num_ctx,
        "temperature": temperature,
        "num_predict": num_predict,
        "stop": ["\n\n\n", "```", "Note:", "Note "],
    }

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if use_json_format:
        payload["format"] = "json"

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["response"]
        except requests.exceptions.ConnectionError as e:
            raise OllamaError(
                _CONN_HELP.format(
                    url=base_url, model=MODEL_NAME, embed_model=EMBED_MODEL_NAME
                )
            ) from e
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (2**attempt))

    raise OllamaError(f"Ollama generate failed after {retries + 1} attempts: {last_err}")

def embed(
    text: str,
    model: str = EMBED_MODEL_NAME,
    base_url: str = OLLAMA_BASE_URL,
    retries: int = 2,
    backoff: float = 3.0,
) -> list[float]:
    payload = {"model": model, "prompt": text}

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(f"{base_url}/api/embeddings", json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["embedding"]
        except requests.exceptions.ConnectionError as e:
            raise OllamaError(
                _CONN_HELP.format(
                    url=base_url, model=MODEL_NAME, embed_model=EMBED_MODEL_NAME
                )
            ) from e
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (2**attempt))

    raise OllamaError(f"Ollama embed failed after {retries + 1} attempts: {last_err}")
