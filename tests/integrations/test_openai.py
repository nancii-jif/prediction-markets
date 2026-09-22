"""Provider configuration and local counting, without hosted requests/downloads."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import yaml

from prediction_markets import config
from prediction_markets.runtime import runner as runner
from prediction_markets.integrations import tokenization as tokenization


@pytest.mark.parametrize("effort", ["medium", "high"])
def test_openai_config_loads_and_excludes_vllm_sampling(tmp_path, effort):
    values = yaml.safe_load((config.ROOT / "configs" / "openai.yaml").read_text())
    values["openai_reasoning_effort"] = effort
    path = tmp_path / "openai.yaml"
    path.write_text(yaml.safe_dump(values))
    settings = config.load(path)
    assert settings.model_provider == "openai"
    assert settings.model_revision is None
    assert settings.model_api_key_env == "OPENAI_API_KEY"
    assert settings.openai_api == "responses"
    assert settings.openai_reasoning_effort == effort
    assert settings.sampling("agent-01") == {}
    assert replace(settings, temperature=1.0).sampling("agent-01") == {"temperature": 1.0}
    reasoning = replace(settings, temperature=None, top_p=None, openai_reasoning_effort="low")
    assert reasoning.sampling("agent-02") == {}
    assert reasoning.snapshot()["openai_reasoning_effort"] == "low"


@pytest.mark.parametrize("changes", [
    {"model_provider": "unknown"}, {"openai_reasoning_effort": "invalid"},
    {"openai_reasoning_effort": True}, {"model_revision": "main"},
    {"enable_thinking": True}, {"temperature": -1}, {"top_p": 0},
    {"openai_api": "invalid"},
])
def test_openai_settings_reject_invalid_provider_options(changes):
    settings = config.load(config.ROOT / "configs" / "openai.yaml")
    with pytest.raises(ValueError):
        replace(settings, **changes)


def test_vllm_defaults_keep_existing_configs_compatible(tmp_path):
    settings = config.load(config.CONFIG_DIR / "mvp.yaml")
    values = asdict(settings)
    values.pop("model_provider")
    values.pop("openai_reasoning_effort")
    values.pop("openai_api")
    path = tmp_path / "existing.yaml"
    path.write_text(yaml.safe_dump(values))
    assert config.load(path) == settings
    with pytest.raises(ValueError, match="only applies"):
        replace(settings, openai_reasoning_effort="low")
    with pytest.raises(ValueError, match="temperature"):
        replace(settings, temperature=None)
    with pytest.raises(ValueError, match="only applies"):
        replace(settings, openai_api="responses")


def test_openai_loader_never_loads_a_hugging_face_tokenizer(monkeypatch):
    import huggingface_hub
    settings = config.load(config.ROOT / "configs" / "openai.yaml")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: pytest.fail("HF download"))
    monkeypatch.setattr(tokenization, "OpenAITokenizer", lambda name: SimpleNamespace(model=name))
    assert runner.load_tokenizer(settings).model == settings.model_name


def test_counter_includes_tools_and_history_and_treats_special_text_as_data(monkeypatch):
    import tiktoken
    seen = []

    def encode(text, *, disallowed_special):
        assert disallowed_special == ()
        seen.append(json.loads(text))
        return list(text.encode())

    encoding = SimpleNamespace(encode=encode)
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda name: encoding)
    counter = tokenization.OpenAITokenizer("gpt-4.1")
    messages = [{"role": "user", "content": "<|endoftext|> 新闻"}]
    before = len(counter.apply_chat_template(messages, tools=[]))
    tools = [{"type": "function", "function": {"name": "search_news"}}]
    with_tools = len(counter.apply_chat_template(messages, tools=tools))
    messages.append({"role": "assistant", "content": "a longer history"})
    with_history = len(counter.apply_chat_template(messages, tools=tools))
    assert before < with_tools < with_history
    assert seen[-1] == {"messages": messages, "tools": tools}
    assert with_history > len(json.dumps(seen[-1], ensure_ascii=False, separators=(",", ":")).encode())


def test_unknown_openai_tokenizer_falls_back_with_a_visible_estimate_warning(monkeypatch, caplog):
    import tiktoken
    def unknown(name):
        raise KeyError(name)
    monkeypatch.setattr(tiktoken, "encoding_for_model", unknown)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: name)
    counter = tokenization.OpenAITokenizer("future-model")
    assert counter.encoding == "o200k_base"
    assert "estimating" in caplog.text
