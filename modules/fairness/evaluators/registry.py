# OmniTrust Fairness Module - Evaluator Registry
# Copyright (c) 2025 OmniTrust Team

"""
Registry for bias evaluators.

Provides decorators and lookup functions for evaluator registration.
"""

from typing import Dict, Type, Optional, List
import threading


class EvaluatorRegistry:
    """
    Thread-safe registry for bias evaluators.

    Usage:
        @EvaluatorRegistry.register("order")
        class OrderBiasEvaluator(CognitiveBiasEvaluator):
            ...

        # Later:
        evaluator_cls = EvaluatorRegistry.get("order")
        evaluator = evaluator_cls()
    """

    _evaluators: Dict[str, Type] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register an evaluator class.

        Args:
            name: Unique identifier for the evaluator

        Returns:
            Decorator function
        """
        def decorator(evaluator_class: Type) -> Type:
            with cls._lock:
                if name in cls._evaluators:
                    raise ValueError(f"Evaluator '{name}' is already registered")
                cls._evaluators[name] = evaluator_class
            return evaluator_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """
        Get an evaluator class by name.

        Args:
            name: Evaluator identifier

        Returns:
            Evaluator class or None
        """
        return cls._evaluators.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered evaluator names."""
        return list(cls._evaluators.keys())

    @classmethod
    def get_by_category(cls, category: str) -> Dict[str, Type]:
        """
        Get all evaluators of a specific category.

        Args:
            category: Category name (e.g., "cognitive", "social")

        Returns:
            Dict of {name: evaluator_class}
        """
        result = {}
        for name, evaluator_cls in cls._evaluators.items():
            if hasattr(evaluator_cls, 'category'):
                if evaluator_cls.category.value.lower() == category.lower():
                    result[name] = evaluator_cls
        return result

    @classmethod
    def create(cls, name: str, **kwargs):
        """
        Create an evaluator instance by name.

        Args:
            name: Evaluator identifier
            **kwargs: Arguments to pass to evaluator constructor

        Returns:
            Evaluator instance

        Raises:
            ValueError: If evaluator not found
        """
        evaluator_cls = cls.get(name)
        if evaluator_cls is None:
            raise ValueError(f"Evaluator '{name}' not found. Available: {cls.list_all()}")
        return evaluator_cls(**kwargs)

    @classmethod
    def clear(cls):
        """Clear all registered evaluators (for testing)."""
        with cls._lock:
            cls._evaluators.clear()


# Convenience alias
evaluator_registry = EvaluatorRegistry
