import requests
import os
from config import GET_API, LOAD_API_CONFIG

class google_scholar():
    def __init__(self):
        # self.api_key = os.environ.get("SERP_API_KEY", None)

        model_name = 'serp'
        api_key, api_url = GET_API(model_name)
        if not api_url:
            raise ValueError(f"api_url is missing for model: {model_name}")
        if not api_key:
            raise ValueError(f"api_key is missing for model: {model_name}")

        self.api_key = api_key
    
    def run(self, query):
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self.api_key,
            "hl": "en"
        }
        
        try:
            response = requests.get("https://serpapi.com/search", params=params)
            data = response.json()
            
            if "organic_results" in data and len(data["organic_results"]) > 0:
                result = data["organic_results"][0]
                return {
                    "title": result.get("title", ""),
                    "author": [author["name"] for author in result.get("authors", [])],
                    "pub_year": result.get("publication_info", {}).get("year", "")
                }
            return {'title': "no match!", "author": "no match!", "pub_year": "no match!"}
        except Exception as e:
            return {'error': str(e)}