import requests

def multilingual_mutate(goal, mutation_language, src_lang = 'auto'):
    languages_supported = {
            'en': 'English',
            'zh-CN': 'Chinese',
            'it': 'Italian',
            'vi': 'Vietnamese',
            'ar': 'Arabic',
            'ko': 'Korean',
            'th': 'Thai',
            'bn': 'Bengali',
            'sw': 'Swahili',
            'jv': 'Javanese'
        }

    lang = None
    if mutation_language in languages_supported:
        lang = languages_supported[mutation_language] 
    else:
        raise ValueError(f"Unsupported language: {mutation_language}")
    
    # 使用谷歌翻译
    googleapis_url = 'https://translate.googleapis.com/translate_a/single'
    url = '%s?client=gtx&sl=%s&tl=%s&dt=t&q=%s' % (googleapis_url, src_lang, mutation_language, goal)
    data = requests.get(url).json()
    res = ''.join([s[0] for s in data[0]])
    return res




