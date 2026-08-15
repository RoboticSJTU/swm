import base64
import json
import os
import re
from pathlib import Path
from typing import List, Optional

import httpx
from dotenv import find_dotenv, load_dotenv
from json_repair import repair_json
from openai import OpenAI
from urllib.parse import urlsplit


load_dotenv(find_dotenv())


_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validated_base_url(value: str) -> str:
    base_url = str(value).strip().rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("model base URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "model base URL must not contain credentials, a query, or a fragment"
        )
    local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme == "http" and not local:
        raise ValueError("remote model base URL must use HTTPS")
    return base_url


def _client_route(model: str) -> tuple[str, str, bool]:
    override = os.getenv("SWM_MODEL_BASE_URL")
    if override:
        key_environment = os.getenv("SWM_MODEL_API_KEY_ENV", "SWM_MODEL_API_KEY")
        if _ENV_NAME_PATTERN.fullmatch(key_environment) is None:
            raise ValueError("SWM_MODEL_API_KEY_ENV is not a valid environment name")
        base_url = _validated_base_url(override)
        local = urlsplit(base_url).hostname in {"127.0.0.1", "localhost", "::1"}
        return base_url, key_environment, local
    if model.startswith(("gemini", "gpt", "o")):
        return "https://api.linkai.shop/v1", "lin_API_KEY", False
    if model.startswith(("kimi-k3", "qwen3.7-max", "glm-5.2")):
        return "https://apicz.boyuerichdata.com/v1", "BOYUE_API_KEY", False
    if model.startswith("Qwen3.5-397B-A17B"):
        return "https://xyx.openapi-qb-ai.sii.edu.cn/v1", "QWEN_API_KEY", False
    if model.startswith("Qwen3.5-27B"):
        return "https://xy.openapi-qb.sii.edu.cn/v1", "QWEN_API_KEY", False
    base_url = _validated_base_url(
        os.getenv("SWM_LOCAL_MODEL_BASE_URL", "http://127.0.0.1:8001/v1")
    )
    local = urlsplit(base_url).hostname in {"127.0.0.1", "localhost", "::1"}
    return base_url, "SWM_LOCAL_MODEL_API_KEY", local


def get_client(
    model: str,
    *,
    timeout_seconds: Optional[float] = None,
    max_retries: Optional[int] = None,
):
    """Return the OpenAI-compatible client for a model name."""
    base_url, key_environment, local = _client_route(model)
    api_key = os.getenv(key_environment)
    if not api_key and local:
        api_key = "0"
    if not api_key:
        raise RuntimeError(
            f"Missing API credential: set the {key_environment} environment variable"
        )
    options = {
        "http_client": httpx.Client(trust_env=False),
        "api_key": api_key,
        "base_url": base_url,
    }
    if timeout_seconds is not None:
        options["timeout"] = timeout_seconds
    if max_retries is not None:
        options["max_retries"] = max_retries

    return OpenAI(**options)


def call_gpt(
    model: str,
    prompt: str,
    image_paths: Optional[List[Path]] = None,
    temperature: Optional[float] = None,
) -> str:
    content = [{"type": "text", "text": prompt}]
    if image_paths:
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths]
        for path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(path)},
                }
            )

    if temperature is None:
        temperature = 0.7 if model.startswith("Qwen3.5") else 0.3

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
    }
    if model.startswith("Qwen3.5"):
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    client = get_client(model)
    try:
        return client.chat.completions.create(**kwargs).choices[0].message.content
    finally:
        client.close()


def call_gpt_json(
    model: str,
    prompt: str,
    image_paths: Optional[List[Path]] = None,
    temperature: Optional[float] = None,
):
    for _ in range(20):
        output = None
        try:
            output = call_gpt(model, prompt, image_paths, temperature)
            output = strip_think_output(output)
            response_json = json.loads(repair_json(output))
            if isinstance(response_json, dict):
                return response_json
        except Exception as error:
            print(f"call gpt error: {error}")
            if output is not None:
                print(output)

    raise RuntimeError("model did not return a JSON object after 20 attempts")


def strip_think_output(text: str) -> str:
    if "</think>" not in text:
        return text

    text = text.rsplit("</think>", 1)[1].strip()
    stack = 0
    start = -1
    last_json = ""

    for index, char in enumerate(text):
        if char == "{":
            if stack == 0:
                start = index
            stack += 1
        elif char == "}" and stack > 0:
            stack -= 1
            if stack == 0 and start != -1:
                last_json = text[start : index + 1]

    return last_json.strip() if last_json else text


def image_to_base64(path: Path) -> str:
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def image_to_data_url(path: Path) -> str:
    media_type = {
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(Path(path).suffix.lower(), "image/jpeg")
    return f"data:{media_type};base64,{image_to_base64(path)}"
