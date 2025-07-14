from openai import OpenAI, AzureOpenAI
from strategist.llm_pricing import get_price

import os

MODEL_TO_NAME = {
    "gpt-4o": "gpt-4o-glob1",
    "gpt-4o-mini": "gpt-4o-mini-glob1",
}


class AzureOpenAIClient:
    def __init__(self, model="gpt-4o-mini", temperature=None):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
        )
        self.model = MODEL_TO_NAME[model]
        self.temperature = temperature

    def get_completion(self, messages, max_tokens=3000):
        if self.temperature is None:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                seed=123,
            )
        else:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                seed=123,
                temperature=self.temperature,
            )

    def get_price(self, completion):
        return get_price(completion)


class OpenAIClient:
    def __init__(self, model):
        self.client = OpenAI()
        self.model = model

    def get_completion(self, messages):
        return self.client.chat.completions.create(model=self.model, messages=messages)

    def get_price(self, completion):
        return get_price(completion)
