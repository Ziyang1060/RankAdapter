import os
import time
from openai import OpenAI


class LoadBalancer:
    def __init__(self, api_keys):
        if not api_keys:
            raise ValueError("API keys list cannot be empty")
        self.api_keys = api_keys
        self.current_index = 0

        print(f"Loaded {len(api_keys)} API keys")

    def get_next_api_key(self):
        api_key = self.api_keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        return api_key


class OpenaiClient:
    def __init__(self, load_balancer, serving_backend):
        self.load_balancer = load_balancer
        self.serving_backend = serving_backend
        self.base_url = {
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com",
            "siliconflow": "https://api.siliconflow.cn/v1",
            "local": "http://localhost:5777/v1"
        }

    def chat(self, model, messages, **kwargs):
        model_name_map = {
            "Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
        }
        if model in model_name_map:
            model = model_name_map[model]
        cnt = 0
        while True:
            try:
                cnt += 1
                api_key = self.load_balancer.get_next_api_key() if self.load_balancer else None
                client = OpenAI(
                    api_key=api_key, base_url=self.base_url[self.serving_backend]
                )
                completion = client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
                response = {
                    "response": completion.choices[0].message.content,
                    "usage": completion.usage.to_dict(),
                }
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print("reduce_length")
                    return "ERROR::reduce_length"
                time.sleep(0.1)
                if cnt == 3:
                    response = {
                        "response": "error",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                            "total_tokens": 0,
                        },
                    }
                    break
        return response


def create_llm_client(model_name: str) -> OpenaiClient:
    import dotenv
    dotenv.load_dotenv()

    if "deepseek" in model_name:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        load_balancer = LoadBalancer([api_key])
        return OpenaiClient(load_balancer=load_balancer, serving_backend="deepseek")
    elif "gpt" in model_name:
        api_key = os.getenv("OPENAI_API_KEY")
        load_balancer = LoadBalancer([api_key])
        return OpenaiClient(load_balancer=load_balancer, serving_backend="openai")
    elif "Qwen" in model_name:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        load_balancer = LoadBalancer([api_key])
        return OpenaiClient(load_balancer=load_balancer, serving_backend="local")
    else:
        raise ValueError(f"Model {model_name} is not supported via API!")


if __name__ == "__main__":
    model_name = "Qwen2-7B-Instruct"
    client = create_llm_client(model_name)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who are you?"},
    ]
    for i in range(1):
        response = client.chat(model_name, messages=messages)
        print(response)
