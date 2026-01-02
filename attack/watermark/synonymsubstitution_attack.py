import random
import torch
from attack.base_attack import AttackInterface
from typing import Any
from nltk.corpus import wordnet
from transformers import BertTokenizer, BertForMaskedLM
import nltk

# Ensure wordnet data is available
nltk.download('wordnet')


class SynonymSubstitutionAttack(AttackInterface):
    """Synonym substitution attack using WordNet."""

    def __init__(self, model: Any, dataset: Any, ratio: float = 0.1):
        """
        Initialize the SynonymSubstitutionAttack.

        Parameters:
            model (Any): The model being attacked.
            dataset (Any): The dataset containing text data.
            ratio (float): The ratio of words to replace with synonyms.
        """
        super().__init__(model, dataset)
        self.ratio = ratio

    def perform_attack(self) -> Any:
        """
        Perform synonym substitution attack.

        Attack method replaces words in the text with synonyms
        based on the provided ratio.
        """
        attacked_data = []

        # Iterating over the dataset
        for text in self.dataset:
            attacked_text = self._substitute_synonyms(text)
            attacked_data.append(attacked_text)

        return attacked_data

    def _substitute_synonyms(self, text: str) -> str:
        """
        Perform synonym substitution on the input text.

        Parameters:
            text (str): The input text.

        Returns:
            str: The text with synonyms substituted.
        """
        words = text.split()
        num_words = len(words)

        # Dictionary to cache synonyms for words
        word_synonyms = {}

        # First pass: Identify replaceable words and cache their synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if word not in word_synonyms:
                synonyms = [syn for syn in wordnet.synsets(word) if len(syn.lemmas()) > 1]
                word_synonyms[word] = synonyms
            if word_synonyms[word]:
                replaceable_indices.append(i)

        # Calculate the number of words to replace
        num_to_replace = min(int(self.ratio * num_words), len(replaceable_indices))

        # Randomly select words to replace
        if num_to_replace > 0:
            indices_to_replace = random.sample(replaceable_indices, num_to_replace)

            # Perform replacement
            for i in indices_to_replace:
                synonyms = word_synonyms[words[i]]
                chosen_syn = random.choice(synonyms)
                new_word = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
                words[i] = new_word

        # Join the words back into a single string
        replaced_text = ' '.join(words)

        return replaced_text

    def get_attack_result(self) -> Any:
        """
        Get the result of the attack.

        In this case, return the attacked data.

        :return: The attacked data
        """
        return self.attacked_data
