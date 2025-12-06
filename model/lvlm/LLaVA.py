import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


class LLaVA:

    def __init__(self, version):
        self.version = version
        self.build_model()

    def build_model(self):
        model_name = f"llava-hf/{self.version}"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation='flash_attention_2',
        ).to(0)
        self.processor = AutoProcessor.from_pretrained(model_name)

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
                                        "type": "image"
                                    }
                                ]
                            }
                        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
        output = self.model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            temperature=temp,
        )
        final_ans = self.processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT: ')[-1].strip()
        return final_ans