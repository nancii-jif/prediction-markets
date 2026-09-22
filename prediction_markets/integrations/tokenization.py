"""Local context estimates for OpenAI, using the participant's tokenizer interface."""

from __future__ import annotations

import json
import copy
import logging
import math

from .. import config
from .model_protocol import PermanentInferenceError

log = logging.getLogger(__name__)


class OpenAITokenizer:
    """Count messages, history and schemas without loading any model weights.

    OpenAI's server-side chat template is not public. Tokenize the complete JSON
    envelope and allow 20% plus 512 tokens for framing/schema overhead. This is
    an estimate for local history trimming, not an exact API usage count; the
    returned completion's usage is the authoritative count saved in transcripts.
    """

    def __init__(self, model: str) -> None:
        import tiktoken

        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            log.warning("unknown OpenAI tokenizer for %s; estimating with o200k_base", model)
            self.encoding = tiktoken.get_encoding("o200k_base")

    def apply_chat_template(
        self, messages, *, tools, tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    ):
        if not tokenize:
            raise ValueError("OpenAITokenizer supports counting only, not prompt rendering")
        # Encrypted reasoning is opaque state, not literal prompt text. Count
        # its API-reported tokens rather than tokenizing base64 ciphertext.
        counted_messages = copy.deepcopy(messages)
        reasoning_tokens = 0
        for message in counted_messages:
            replay = message.get("_responses")
            if replay is not None:
                count = replay.get("reasoning_tokens", 0)
                if type(count) is int and count > 0:
                    reasoning_tokens += count
                for item in replay["output"]:
                    if item.get("type") == "reasoning":
                        item.pop("encrypted_content", None)
        text = json.dumps(
            {"messages": counted_messages, "tools": tools},
            ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        )
        count = len(self.encoding.encode(text, disallowed_special=()))
        # Only len() is used by Participant; avoid allocating fake token IDs.
        return range(math.ceil((count + reasoning_tokens) * 1.2) + 512)


def load_tokenizer(settings: config.Settings):
    """Download tokenizer artifacts only, never model weights or remote code."""
    if settings.model_provider == "openai":
        return OpenAITokenizer(settings.model_name)

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = snapshot_download(
        settings.model_name, revision=settings.model_revision,
        cache_dir=str(config.ROOT / "data" / "huggingface"),
        allow_patterns=["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                        "vocab.json", "merges.txt", "special_tokens_map.json"],
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise PermanentInferenceError("the pinned tokenizer has no chat template")
    return tokenizer


