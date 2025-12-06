from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


class LLaVANeXT:

    def __init__(self, version):
        self.version = version
        self.build_model()

    def build_model(self):
        model_name = f"llava-hf/{self.version}"
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation='flash_attention_2',
        ).to(0)
        self.processor = LlavaNextProcessor.from_pretrained(model_name)

    def generate(self, image, question, temp):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": question
                    },
                    {
                        "type": "image",
                    }
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(0)
        output = self.model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            temperature=temp
        )
        if '7b' in self.version:
            answer = self.processor.decode(output[0], skip_special_tokens=True).split('[/INST] ')[-1].strip()
        elif '13b' in self.version:
            answer = self.processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT: ')[-1].strip()
        return answer