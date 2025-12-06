# from fastchat.llm_judge.common import (
#     chat_completion_openai,
# )
import openai
import time
from utils.watermark.exceptions.exceptions import OpenAIModelConfigurationError
from config import GET_API

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"
API_KEY, API_URL = GET_API()

def chat_completion_openai(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None:
        # openai.api_base = api_dict["api_base"]
        # TODO Change Normal Web here.
        openai.api_base = API_URL
        openai.api_key = API_KEY
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


def chat_completion_openai_demo(model_api, conv, temperature, max_new_token):
    return chat_completion_openai(model_api, conv, temperature, max_new_token)


#-----------------------------------watermark_openai---------------------------#
class OpenAIAPI:
    """API class for OpenAI API."""
    def __init__(self, model, temperature, system_content):
        """
            Initialize OpenAI API with model, temperature, and system content.

            Parameters:
                model (str): Model name for OpenAI API.
                temperature (float): Temperature value for OpenAI API.
                system_content (str): System content for OpenAI API.
        """

        self.model = model
        self.temperature = temperature
        self.system_content = system_content
        self.client = openai.OpenAI()
        

        # List of supported models
        supported_models = ['gpt-3.5-turbo', 'gpt-4']

        # Check if the provided model is within the supported models
        if self.model not in supported_models:
            raise OpenAIModelConfigurationError(f"Unsupported model '{self.model}'. Supported models are {supported_models}.")

    def get_result_from_gpt4(self, query):
        """get result from GPT-4 model."""
        response = self.client.chat.completions.create(
            model="gpt-4-0613",
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": query},
            ]
        )
        return response
    
    def get_result_from_gpt3_5(self, query):
        """get result from GPT-3.5 model."""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": query},
            ]
        )
        return response

    def get_result(self, query):
        """get result from OpenAI API."""
        while True:
            try:
                if self.model == 'gpt-3.5-turbo':
                    result = self.get_result_from_gpt3_5(query)
                elif self.model == 'gpt-4':
                    result = self.get_result_from_gpt4(query)
                break
            except Exception as e:
                print(f"OpenAI API error: {str(e)}")
            time.sleep(10)
        return result.choices[0].message.content


if __name__ == "__main__":
    openai_util = OpenAIAPI('gpt-3.5-turbo', 0.2, "Your are a helpful assistant to rewrite the text.")
    text =  "Rewrite the following paragraph:\n "
    text += "Whoever gets him, they'll be getting a good one,\" David Montgomery said. INDIANAPOLIS \u2014 Hakeem Butler has been surrounded by some of the best wide receivers on the planet this month, so it's only natural he'd get a touch of star-struck when he got a chance to meet Randy Moss, a Hall of Famer and someone Butler idolized growing up and still considers an inspiration to this day.\n\"It's crazy to see someone you've watched on TV and seen on YouTube and then be standing right there and you're like, 'Man, he's just a regular dude,'\" Butler, a 6-foot 5, 227-pound receiver from Iowa State, said Thursday at the NFL Scouts' Luncheon, which kicks off the week of the NFL Scouts' Breakfast and the NFL Scouts' Luncheon at the JW Marriott Indianapolis, which takes place Sunday and Monday, Feb. 1 and 2, at Lucas Oil Stadium, 1 Lucas Oil Stadium Dr., Indianapolis, IN 46225, and which is open to all NFL Scouts and team personnel and others with a pass from the NFL Scouts' Breakfast and the NFL Scouts' Luncheon, and which is sponsored by the NFL Scouts' Breakfast and the NFL Scouts' Luncheon, and which is organized by the NFL Scouts' Breakfast and the NFL Scouts' Luncheon, and which is hosted by the"
    result = openai_util.get_result(text)
    print(result)
