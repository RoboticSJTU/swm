from __future__ import annotations

import base64
import json
import os
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
    return OpenAI(api_key=api_key, base_url=base_url, http_client=httpx.Client(trust_env=False))


def call_gpt(
    model: str,
    prompt: str,
    image_paths: list[Path] | None = None,
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

    if str(client.base_url).startswith("http://127.0.0.1") or model.startswith(("gemini", "gpt")):
        kwargs["temperature"] = 0

    try:
        return client.chat.completions.create(**kwargs).choices[0].message.content
    finally:
        client.close()


def call_gpt_json(
    model: str,
    prompt: str,
    image_paths: list[Path] | None = None,
):
    for _ in range(20):
        output = None
        try:
            output = call_gpt(model, prompt, image_paths)
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


def image_to_data_url(path: Path) -> str:
    path = Path(path)
    media_type = {
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{media_type};base64,{encoded}"
