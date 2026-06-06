import importlib
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import Config


def import_evaluate_jailbreak_with_optional_dependency_stubs():
    sys.modules.setdefault("litellm", types.SimpleNamespace(batch_completion=None))

    vllm_module = types.ModuleType("vllm")
    vllm_module.LLM = object
    vllm_module.SamplingParams = object
    sys.modules.setdefault("vllm", vllm_module)

    distributed_module = types.ModuleType("vllm.distributed")
    parallel_state_module = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state_module.destroy_model_parallel = lambda: None
    sys.modules.setdefault("vllm.distributed", distributed_module)
    sys.modules.setdefault("vllm.distributed.parallel_state", parallel_state_module)

    module_name = "pipeline.submodules.evaluate_jailbreak"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_evaluate_jailbreak_import_does_not_require_optional_judge_dependencies():
    module_name = "pipeline.submodules.evaluate_jailbreak"
    sys.modules.pop(module_name, None)
    sys.modules.pop("litellm", None)
    sys.modules.pop("vllm", None)
    sys.modules.pop("vllm.distributed", None)
    sys.modules.pop("vllm.distributed.parallel_state", None)

    evaluate_jailbreak = importlib.import_module(module_name)

    assert hasattr(evaluate_jailbreak, "evaluate_jailbreak")


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_config_uses_ollama_llamaguard_for_jailbreak_evaluation_by_default():
    cfg = Config(model_alias="gemma-2b-it", model_path="/tmp/gemma-2b-it")

    assert "ollama_llamaguard" in cfg.jailbreak_eval_methodologies
    assert "llamaguard2" not in cfg.jailbreak_eval_methodologies


def test_ollama_llamaguard_judge_posts_prompt_response_chat_and_parses_labels():
    evaluate_jailbreak = import_evaluate_jailbreak_with_optional_dependency_stubs()
    completions = [
        (
            "prompt one",
            "safe response with enough words to avoid the short response guard in the evaluator",
        ),
        (
            "prompt two",
            "unsafe response with enough words to avoid the short response guard in the evaluator",
        ),
    ]
    ollama_payloads = iter(
        [
            {"message": {"content": "safe"}},
            {"message": {"content": "unsafe\nS2"}},
        ]
    )
    calls = []

    class FakeOpener:
        def open(self, request, timeout):
            calls.append(
                {
                    "url": request.full_url,
                    "body": json.loads(request.data.decode("utf-8")),
                    "timeout": timeout,
                }
            )
            return FakeResponse(next(ollama_payloads))

    env = {
        **os.environ,
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_GUARD_MODEL": "llama-guard3:1b",
    }
    with patch.dict(os.environ, env, clear=True):
        with patch("urllib.request.build_opener", return_value=FakeOpener()):
            classifications = evaluate_jailbreak.ollama_llamaguard_judge_fn(
                prompts=[prompt for prompt, _ in completions],
                responses=[response for _, response in completions],
            )

    assert classifications == [0, 1]
    assert calls[0]["url"] == "http://127.0.0.1:11434/api/chat"
    assert calls[0]["body"]["model"] == "llama-guard3:1b"
    assert calls[0]["body"]["stream"] is False
    assert calls[0]["body"]["messages"] == [
        {"role": "user", "content": completions[0][0]},
        {"role": "assistant", "content": completions[0][1]},
    ]


def test_ollama_llamaguard_uses_no_proxy_opener_for_local_ollama_requests():
    evaluate_jailbreak = import_evaluate_jailbreak_with_optional_dependency_stubs()
    proxy_handler_args = []
    build_opener_args = []

    class FakeOpener:
        def open(self, request, timeout):
            return FakeResponse({"message": {"content": "safe"}})

    def fake_proxy_handler(proxies=None):
        proxy_handler_args.append(proxies)
        return "proxy-handler"

    def fake_build_opener(*handlers):
        build_opener_args.append(handlers)
        return FakeOpener()

    env = {
        **os.environ,
        "HTTP_PROXY": "http://proxy.example:8080",
        "HTTPS_PROXY": "http://proxy.example:8080",
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_GUARD_MODEL": "llama-guard3:1b",
    }
    with patch.dict(os.environ, env, clear=True):
        with patch("urllib.request.ProxyHandler", fake_proxy_handler):
            with patch("urllib.request.build_opener", fake_build_opener):
                classifications = evaluate_jailbreak.ollama_llamaguard_judge_fn(
                    prompts=["What is the capital of France?"],
                    responses=["Paris is the capital of France."],
                )

    assert classifications == [0]
    assert proxy_handler_args == [{}]
    assert build_opener_args == [("proxy-handler",)]
