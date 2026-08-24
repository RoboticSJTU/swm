from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import find_dotenv, load_dotenv
from json_repair import repair_json
from openai import OpenAI

load_dotenv(find_dotenv())


def get_client(model: str):
    """Return the OpenAI-compatible client for a model name."""
    if model.startswith(("gemini", "gpt")):
        # api_key = os.getenv("das_API_KEY")
        # base_url = "https://dasuapi.com/v1"
        api_key = os.getenv("A6_API_KEY")
        base_url = "https://api.a6api.com/v1"

    elif model.startswith(("kimi-k3", "qwen3.7-max", "glm-5.2")):
        api_key = os.getenv("BOYUE_API_KEY")
        base_url = "https://apicz.boyuerichdata.com/v1"

    elif model.startswith("Qwen3.8-27B"):
        api_key = os.getenv("QWEN_API_KEY")
        base_url = "https://x.openapi-qb.sii.edu.cn/v1"

    else:
        api_key = "0"
        base_url = "http://127.0.0.1:8001/v1"

    return OpenAI(api_key=api_key, base_url=base_url)


def call_gpt(
    model: str,
    prompt: str,
    image_paths: list[Path] | None = None,
    *,
    response_format: dict | None = None,
    max_tokens: int | None = None,
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

    client = get_client(model)

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if str(client.base_url).startswith("http://127.0.0.1") or model.startswith(("gemini", "gpt")):
        kwargs["temperature"] = 0

    try:
        completion = client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        if response_format is not None and choice.finish_reason != "stop":
            raise ValueError(
                "structured model response did not finish normally: "
                f"{choice.finish_reason!r}"
            )
        return choice.message.content
    finally:
        client.close()


def call_gpt_json(
    model: str,
    prompt: str,
    image_paths: list[Path] | None = None,
    *,
    response_format: dict | None = None,
    attempts: int = 3,
    max_tokens: int | None = None,
):
    if type(attempts) is not int or not 1 <= attempts <= 5:
        raise ValueError("attempts must be an integer from 1 to 5")
    last_error: Exception | None = None
    for attempt in range(attempts):
        output = None
        try:
            call_kwargs = {}
            if max_tokens is not None:
                call_kwargs["max_tokens"] = max_tokens
            if response_format is None:
                output = call_gpt(model, prompt, image_paths, **call_kwargs)
            else:
                output = call_gpt(
                    model,
                    prompt,
                    image_paths,
                    response_format=response_format,
                    **call_kwargs,
                )
            if response_format is None:
                output = strip_think_output(output)
                json_text = repair_json(output)
            else:
                json_text = output
            response_json = json.loads(json_text)
            if not isinstance(response_json, dict):
                raise ValueError("model response is not a JSON object")
            return response_json
        except Exception as error:
            last_error = error
            print(f"call gpt error: {error}")
            if output is not None:
                print(output[:2000])
            if attempt + 1 < attempts:
                time.sleep(2**attempt)

    raise RuntimeError(
        f"model did not return a JSON object after {attempts} attempts"
    ) from last_error


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


def image_to_data_url(path: Path) -> str:
    path = Path(path)
    media_type = {
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{media_type};base64,{encoded}"
