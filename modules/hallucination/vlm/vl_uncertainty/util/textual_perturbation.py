import string
import random


def word_swapping(question):
    try:
        words = question.split()
        if len(words) < 2:
            return question
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)
    except:
        return question

def word_deleting(question):
    try:
        words = question.split()
        if len(words) <= 1:
            return question
        idx = random.randint(0, len(words) - 1)
        del words[idx]
        return ' '.join(words)
    except:
        return question

def word_inserting(question):
    try:
        words = question.split()
        if not words:
            return question
        word_to_insert = random.choice(words)
        idx = random.randint(0, len(words))
        words.insert(idx, word_to_insert)
        return ' '.join(words)
    except:
        return question

def word_replacing(question):
    try:
        words = question.split()
        if len(words) < 2:
            return question
        idx_to_replace = random.randint(0, len(words) - 1)
        replacement_word = random.choice([w for i, w in enumerate(words) if i != idx_to_replace])
        words[idx_to_replace] = replacement_word
        return ' '.join(words)
    except:
        return question

def text_shuffle(question):
    try:
        words = question.split()
        random.shuffle(words)
        return ' '.join(words)
    except:
        return question

def noise_injection(question, noise_level=0.1):
    try:
        chars = list(question)
        num_noisy_chars = int(noise_level * len(chars))
        for _ in range(num_noisy_chars):
            idx = random.randint(0, len(chars) - 1)
            chars[idx] = random.choice(string.ascii_letters)
        return ''.join(chars)
    except:
        return question

def word_dropout(question, dropout_rate=0.1):
    try:
        words = question.split()
        dropped_words = [word for word in words if random.random() > dropout_rate]
        return ' '.join(dropped_words)
    except:
        return question

def character_dropout(question, dropout_rate=0.1):
    try:
        chars = [char for char in question if random.random() > dropout_rate]
        return ''.join(chars)
    except:
        return question

