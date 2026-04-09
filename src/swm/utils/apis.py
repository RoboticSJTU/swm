import base64
from pathlib import Path
from typing import List, Optional
from openai import OpenAI
import json
from json_repair import repair_json
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())

def get_client(model: str):
    if model.startswith(("gemini", "gpt")):
        return OpenAI(api_key=os.getenv("SII_API_KEY"), base_url="http://apicz.boyuerichdata.com/v1")

    if model.startswith("Qwen3.5-397B-A17B"):
        return OpenAI(api_key=os.getenv("QWEN_API_KEY"), base_url="https://xyx.openapi-qb-ai.sii.edu.cn/v1")

    if model.startswith("Qwen3.5-27B"):
        return OpenAI(api_key=os.getenv("QWEN_API_KEY"), base_url="https://xy.openapi-qb-ai.sii.edu.cn/v1")
    
    return OpenAI(api_key="0", base_url="https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-0a63ad2d-c102-4f9f-bdf4-af1a164bc0b0/user-4f2781ab-e1e4-41f6-9367-e0ea36e3562e/vscode/850c6ac3-236b-444e-b556-623a20ed4f95/15ea42f2-4f31-4bbc-8a9d-ec887302fd39/proxy/8000/v1")


def call_gpt(model: str, prompt: str, image_paths: Optional[List[Path]] = None) -> str:
    content = [{"type": "text", "text": prompt}]
    if image_paths:
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths]
        for p in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(p)}"}
            })

    client = get_client(model)

    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
    )

    if model.startswith("Qwen3.5"):
        # # # 推理模式
        # kwargs["temperature"] = 1
        
        # 非推模式
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    output = client.chat.completions.create(**kwargs).choices[0].message.content
    return output

def call_gpt_json(model: str, prompt: str, image_paths: Optional[List[Path]] = None):
    for _ in range(100):
        output = None
        try:
            output = call_gpt(model, prompt, image_paths)
            
            # 用 strip_think_output 提取 JSON 部分
            output = strip_think_output(output)
            response_json = json.loads(repair_json(output))
            if not isinstance(response_json, dict):
                continue
            return response_json
        except Exception as e:
            print(f"call gpt error: {e}")
            if output is not None:
                print(output)
    
    
def strip_think_output(text: str) -> str:
    
    if "</think>" not in text:
        return text

    text = text.rsplit("</think>", 1)[1].strip()

    stack = 0
    start = -1
    last_json = ""

    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start != -1:
                    last_json = text[start:i + 1]

    return last_json.strip() if last_json else text
 
def image_to_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


    
               
    
    
    