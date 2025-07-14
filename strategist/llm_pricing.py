# from https://openai.com/api/pricing/
openai_api_rates_1M = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o-mini-2024-07-18": {"input": 0.150, "output": 0.600},
}


def get_price(completion):
    model = completion.model
    price_per_1M_input_tokens = openai_api_rates_1M[model]["input"]
    price_per_1M_output_tokens = openai_api_rates_1M[model]["output"]
    price_input = price_per_1M_input_tokens * completion.usage.prompt_tokens / 1_000_000
    price_output = (
        price_per_1M_output_tokens * completion.usage.completion_tokens / 1_000_000
    )
    return price_input + price_output
