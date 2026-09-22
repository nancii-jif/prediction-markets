"""Deploy manually only after authorizing GPU costs; never imported by agents.

The image contains this serving function, public weights and the one inference
API key. No project directory, run database, .env or Kalshi credential is mounted.
"""

import os
import subprocess

import modal

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.17.1", "transformers==4.57.6", "tokenizers==0.22.2", "Jinja2==3.1.6")
)
app = modal.App("prediction-markets-qwen")


@app.function(
    image=image, gpu="A100-40GB:1", min_containers=1, max_containers=1,
    # Deploy by file path: Modal includes just this standalone source file.
    timeout=900,
    secrets=[modal.Secret.from_name("prediction-markets-model", required_keys=["VLLM_API_KEY"])],
)
@modal.concurrent(max_inputs=8)
@modal.web_server(8000, startup_timeout=900)
def serve():
    if not os.environ.get("VLLM_API_KEY", "").strip():
        raise RuntimeError("VLLM_API_KEY must be set; refusing to start an unauthenticated endpoint")
    # vLLM reads VLLM_API_KEY from the Modal Secret, never from command arguments.
    subprocess.Popen([
        "vllm", "serve", MODEL, "--revision", REVISION,
        "--tokenizer-revision", REVISION, "--served-model-name", MODEL,
        "--host", "0.0.0.0", "--port", "8000",
        "--tensor-parallel-size", "1", "--dtype", "bfloat16",
        "--language-model-only", "--max-model-len", "20480", "--max-num-seqs", "8",
        # "auto" loads the model card's own generation config, so anything the
        # client does not send explicitly follows Qwen's recommendation rather
        # than vLLM's neutral temperature=1.0 defaults.
        "--gpu-memory-utilization", "0.9", "--generation-config", "auto",
        # All eight agents send an identical system prompt and tool schemas as
        # their prefix, so cache it rather than re-prefilling it every request.
        # Pinned explicitly; do not rely on the engine's default.
        "--enable-prefix-caching",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        "--default-chat-template-kwargs", '{"enable_thinking":false}',
    ])
