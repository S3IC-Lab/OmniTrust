import random
from attack.base_attack import AttackInterface
from typing import Any

class WordDeletionAttack(AttackInterface):
    """
    This is a simple word deletion attack method that inherits from AttackInterface.
    It simulates an attack by deleting certain words from the input text.
    """

    def __init__(self, model: Any, dataset: Any, delete_ratio: float = 0.1):
        """
        Initializes the attack method.

        :param model: The model to be attacked
        :param dataset: The dataset used for the attack
        :param delete_ratio: The proportion of words to delete (between 0 and 1)
        """
        super().__init__(model, dataset)  # Call the parent class constructor
        self.delete_ratio = delete_ratio  # Set the deletion ratio

    def perform_attack(self) -> Any:
        """
        Performs the word deletion attack.

        The logic of the attack is to randomly delete a certain proportion of words from the text based on the specified deletion ratio.

        :return: The attacked data (text data with deleted words)
        """
        attacked_data = []

        for text in self.dataset:
            words = text.split()  # Split the text into words
            num_words_to_delete = int(len(words) * self.delete_ratio)  # Calculate the number of words to delete
            words_to_delete = set(random.sample(words, num_words_to_delete))  # Randomly select words to delete

            # Delete the selected words
            attacked_text = ' '.join(word for word in words if word not in words_to_delete)
            attacked_data.append(attacked_text)

        return attacked_data  # Return the text data after words have been deleted

    def get_attack_result(self) -> Any:
        """Returns the result after the attack"""
        return self.dataset  # Return the modified data (text after deletion)
