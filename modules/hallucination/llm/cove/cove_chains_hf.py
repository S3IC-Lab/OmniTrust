from .cove_chains import ChainOfVerification


class ChainOfVerificationHuggingFace(ChainOfVerification):
    def __init__(
        self, model_id, top_p, temperature, task, setting, questions, model, tokenizer
    ):
        super().__init__(model_id, task, setting, questions)
        self.top_p = top_p
        self.temperature = temperature
        self.model, self.tokenizer = model, tokenizer

    def call_llm(self, prompt: str, max_tokens: int) -> str:
        device = self.model.device

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
    
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=self.top_p,
            temperature=self.temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
    
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    

        response = full_text.replace(prompt, "", 1).strip()
        return response.split("\n\n")[-1] if "\n\n" in response else response

    def process_prompt(self, prompt, command) -> str:
        return self.model_config.prompt_format.format(prompt=prompt, command=command)
