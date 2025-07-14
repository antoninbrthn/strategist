import torch
from transformers import AutoModelForCausalLM, AutoProcessor

import torch
from transformers import AutoModelForCausalLM, AutoProcessor



class LLMText:
    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct", system_prompt=None):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._is_gpt_model = "gpt-4o" in self.model_id
        self._system_prompt = system_prompt
        self.total_price = 0.0

        if self._is_gpt_model:
            from strategist.openai_client import AzureOpenAIClient
            self._client = AzureOpenAIClient(self.model_id)
        else:
            # Load the model and processor
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map="auto")
            self._processor = AutoProcessor.from_pretrained(self.model_id)

    def generate_response(self, text="", max_new_tokens=30):
        # Create the chat template
        messages = []
        if self._system_prompt is not None:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": text})

        if self._is_gpt_model:
            completion = self._client.get_completion(messages=messages, max_tokens=max_new_tokens)
            self.total_price += self._client.get_price(completion)
            response = completion.choices[0].message.content
            return response
        else:
            # Prepare the input text
            input_text = self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            processor_inputs = {"text": input_text, "add_special_tokens": False, "return_tensors": "pt"}
            inputs = self._processor(**processor_inputs).to(self._model.device)

            # Generate the response
            output = self._model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)
            return self._processor.decode(output[0])
