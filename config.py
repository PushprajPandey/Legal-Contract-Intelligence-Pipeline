import os
from pathlib import Path
from dotenv import load_dotenv

_ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE)

CUAD_BASE_PATH: Path = Path(os.getenv("CUAD_BASE_PATH", r"D:\Uptitude\CUAD_v1"))

PDF_ROOT: Path = CUAD_BASE_PATH / "full_contract_pdf"
TXT_ROOT: Path = CUAD_BASE_PATH / "full_contract_txt"

SELECTED_OUTPUT_DIR: Path = Path(
    os.getenv("SELECTED_OUTPUT_DIR", str(CUAD_BASE_PATH / "selected_50"))
)

PREPROCESSED_OUTPUT_PATH: Path = Path(
    os.getenv(
        "PREPROCESSED_OUTPUT_PATH",
        str(Path(__file__).parent / "contracts_preprocessed.json"),
    )
)

RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
SAMPLE_SIZE: int = int(os.getenv("SAMPLE_SIZE", "50"))

MIN_CHARS_THRESHOLD: int = 200
TXT_MATCH_RATIO_THRESHOLD: float = 0.3

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama3.1:8b")
EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text")

NUM_CTX: int = int(os.getenv("NUM_CTX", "8192"))

CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "3000"))
CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "200"))

CHECKPOINT_EVERY: int = int(os.getenv("CHECKPOINT_EVERY", "5"))

CLAUSE_MERGE_STRATEGY: str = os.getenv("CLAUSE_MERGE_STRATEGY", "longest")

_PIPELINE_DIR = Path(__file__).parent

TASK2_CSV_PATH: Path = _PIPELINE_DIR / "task2_results.csv"
TASK2_JSON_PATH: Path = _PIPELINE_DIR / "task2_results.json"

TASK2_FULL_JSON_PATH: Path = _PIPELINE_DIR / "task2_full_results.json"

TASK2_CHECKPOINT_PATH: Path = _PIPELINE_DIR / "task2_checkpoint.json"

SEMANTIC_STORE_PATH: Path = _PIPELINE_DIR / "semantic_store.npz"
