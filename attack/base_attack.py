"""
This module contains the abstract base class for all attack methods.

All specific attack classes should inherit from `AttackInterface` and
implement the `perform_attack` method.
"""

from abc import ABC, abstractmethod
from typing import Any

class AttackInterface(ABC):
    """
    Abstract Attack Interface, all attack methods must inherit 
    this class and implement `perform_attack` .
    """

    def __init__(self, model: Any, dataset: Any):
        """
        Initialize the attack interface.

        :param model: The model to be attacked (e.g., a neural network)
        :param dataset: The dataset used for the attack
        """
        self.model = model
        self.dataset = dataset

    @abstractmethod
    def perform_attack(self) -> Any:
        """Perform the attack method."""

    @abstractmethod
    def get_attack_result(self) -> Any:
        """Get the result of the attack
        :return: The result of the attack
        """
