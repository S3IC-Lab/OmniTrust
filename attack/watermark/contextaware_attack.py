import random
import torch
from attack.base_attack import AttackInterface
from typing import Any
from nltk.corpus import wordnet
from transformers import BertTokenizer, BertForMaskedLM
import nltk

# Ensure WordNet data is downloaded
nltk.download('wordnet')


class ContextAwareSynonymSubstitutionAttack(AttackInterface):
    """
    This is a context-aware synonym replacement attack method inherited from AttackInterface.
    It simulates an attack by replacing some words in the input text with synonyms in a context-aware way.
    """

    def __init__(self, model: Any, dataset: Any, ratio: float, tokenizer: BertTokenizer, bert_model: BertForMaskedLM,
                 device: str = 'cuda'):
        """
        Initialise the attack method.

        :param model: The model to be attacked
        :param dataset: the dataset to be used for the attack
        :param ratio: ratio of words to be replaced (between 0 and 1)
        :param tokenizer: the tokeniser used for BERT
        :param bert_model: BERT model, used for masked language modelling
        :param device: device selection ('cuda' or 'cpu')
        """
        super().__init__(model, dataset)  
        self.ratio = ratio 
        self.tokenizer = tokenizer  
        self.bert_model = bert_model  
        self.device = device 

    def _get_synonyms_from_wordnet(self, word: str):
        """Get synonyms for a given word via WordNet"""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

    def perform_attack(self) -> Any:
        """
        Performs a context-aware synonym replacement attack operation.

        The logic of the attack is to randomly select the words to be replaced from the text based on the specified replacement ratio, the
        and use the BERT model to select the most appropriate synonym replacement based on the context.

        :return: data after the attack (replaced text data)
        """
        attacked_data = []

        for text in self.dataset:
            words = text.split() 
            num_words_to_replace = int(len(words) * self.ratio)  
            replaceable_indices = []

            # Find all the words that can be replaced
            for i, word in enumerate(words):
                if self._get_synonyms_from_wordnet(word):
                    replaceable_indices.append(i)

            # Randomly select the word to be replaced
            indices_to_replace = random.sample(replaceable_indices, min(num_words_to_replace, len(replaceable_indices)))

            for i in indices_to_replace:
                # Replace the word to be replaced with the [MASK] mask
                masked_sentence = words[:i] + ['[MASK]'] + words[i + 1:]
                masked_text = " ".join(masked_sentence)

                # Predicting words in [MASK] position using the BERT model
                inputs = self.tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
                mask_position = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0].item()

                with torch.no_grad():
                    outputs = self.bert_model(**inputs)

                predictions = outputs.logits[0, mask_position]
                predicted_indices = torch.argsort(predictions, descending=True)
                predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices[0:1])
                words[i] = predicted_tokens[0]

            # Re-assembling the replaced words into a sentence
            attacked_text = ' '.join(words)
            attacked_data.append(attacked_text)

        return attacked_data

    def get_attack_result(self) -> Any:
        """Returns the result of the attack (replaced text)"""
        return self.dataset
