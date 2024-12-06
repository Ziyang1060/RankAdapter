import hashlib
import json
import os
from collections import defaultdict
from typing import Dict, Any

from .LLMClients import OpenaiClient

CHARACTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
              "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

INSTRUCTIONS = defaultdict(lambda: "", {
    "DrRank": "从候选医生池中找到一个擅长使用指定治疗方法治疗疾病的医生专家",
    "trec-covid": "Retrieve Scientific paper paragraph to answer this question",
    "webis-touche2020": "You have to retrieve an argument to this debate question",
})


class Cache:
    # TODO: maybe change to SQLite for faster cache, got idea from lm-evaluation-harness
    def __init__(self):
        self.cache_path = f".cache/llm_cache.json"
        if not os.path.exists(self.cache_path):
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump({}, f)  # type: ignore
        with open(self.cache_path, "r", encoding="utf-8") as f:
            self.kv = json.load(f)

    def get(self, key):
        return self.kv.get(key)

    def set(self, key, value):
        self.kv[key] = value
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.kv, f, ensure_ascii=False, indent=4)  # type: ignore

    def exists(self, key):
        return key in self.kv


cache = Cache()

def single_prompt_chat(
        llm_name: str,
        llm_client: OpenaiClient,
        system_prompt: str,
        user_prompt: str,
        temperature=0.8,
) -> Dict[str, Any]:
    key = f"{llm_name}-{hashlib.sha256(user_prompt.encode()).hexdigest()}-{temperature}"
    # print('Looking for cache key:', key)
    if cache.exists(key):
        v = cache.get(key)
        if v["response"] != "error":
            return v
    # else:
    #     raise ValueError("Cache not found for prompt:", user_prompt)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    args = {"model": llm_name, "messages": messages, "temperature": temperature}
    response = llm_client.chat(**args)
    cache.set(key, response)
    return response
