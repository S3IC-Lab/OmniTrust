from .cove_chains import ChainOfVerification
from openai import OpenAI

class ChainOfVerificationOpenAI(ChainOfVerification):
    def __init__(
        self, model_id, temperature, task, setting, questions
    ):
        super().__init__(model_id, task, setting, questions)
        self.temperature = temperature        
        # openai.api_key = os.environ.get("OPENAI_API_KEY", None)
        self.client = OpenAI(
            base_url="https://zzzzapi.com/v1",
            api_key="sk-3vW4zNs9yLa69NK5QrSIRCc0l8fpdelO7nS3gMOkSItyY5jz"
        )


    def call_llm(self, prompt: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=self.temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    def process_prompt(self, prompt, _) -> str:
        # We do not need to do any processing here!
        return prompt