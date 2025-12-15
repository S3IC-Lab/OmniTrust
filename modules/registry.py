"""
Module Registry
===============

Provides a centralized registry for managing evaluators, quantifiers,
and visualizers across all OmniTrust modules.
"""

from typing import Dict, Any, List, Type, Optional, Callable
import threading
import logging


class ModuleRegistry:
    """
    Centralized registry for OmniTrust components.

    Supports:
    - Evaluator registration and lookup
    - Quantifier registration and lookup
    - Visualizer registration and lookup
    - Component discovery and listing

    Usage:
        # Using decorators
        @ModuleRegistry.register_evaluator('my_evaluator')
        class MyEvaluator(BaseEvaluator):
            pass

        # Using class methods
        ModuleRegistry.add_evaluator('my_evaluator', MyEvaluator)

        # Lookup
        evaluator_class = ModuleRegistry.get_evaluator('my_evaluator')
    """

    _instance = None
    _lock = threading.Lock()

    # Storage for registered components
    _evaluators: Dict[str, Type] = {}
    _quantifiers: Dict[str, Type] = {}
    _visualizers: Dict[str, Type] = {}
    _pipelines: Dict[str, Type] = {}

    # Metadata storage
    _evaluator_metadata: Dict[str, Dict[str, Any]] = {}
    _quantifier_metadata: Dict[str, Dict[str, Any]] = {}
    _visualizer_metadata: Dict[str, Dict[str, Any]] = {}
    _pipeline_metadata: Dict[str, Dict[str, Any]] = {}

    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    # === Evaluator Registration ===

    @classmethod
    def register_evaluator(cls, name: str, **metadata) -> Callable:
        """
        Decorator to register an evaluator class.

        Args:
            name: Unique name for the evaluator
            **metadata: Additional metadata (description, version, etc.)

        Returns:
            Decorator function

        Example:
            @ModuleRegistry.register_evaluator('bias_evaluator', version='1.0.0')
            class BiasEvaluator(BaseEvaluator):
                pass
        """
        def decorator(cls_to_register: Type) -> Type:
            ModuleRegistry.add_evaluator(name, cls_to_register, **metadata)
            return cls_to_register
        return decorator

    @classmethod
    def add_evaluator(cls, name: str, evaluator_class: Type, **metadata) -> None:
        """
        Register an evaluator class.

        Args:
            name: Unique name for the evaluator
            evaluator_class: The evaluator class to register
            **metadata: Additional metadata
        """
        with cls._lock:
            if name in cls._evaluators:
                logging.warning(f"Overwriting existing evaluator: {name}")
            cls._evaluators[name] = evaluator_class
            cls._evaluator_metadata[name] = {
                "class_name": evaluator_class.__name__,
                "module": evaluator_class.__module__,
                **metadata
            }
            logging.debug(f"Registered evaluator: {name}")

    @classmethod
    def get_evaluator(cls, name: str) -> Optional[Type]:
        """
        Get an evaluator class by name.

        Args:
            name: Evaluator name

        Returns:
            Evaluator class or None
        """
        return cls._evaluators.get(name)

    @classmethod
    def list_evaluators(cls) -> List[str]:
        """
        List all registered evaluator names.

        Returns:
            List of evaluator names
        """
        return list(cls._evaluators.keys())

    @classmethod
    def get_evaluator_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Get metadata for an evaluator.

        Args:
            name: Evaluator name

        Returns:
            Metadata dictionary
        """
        return cls._evaluator_metadata.get(name, {})

    # === Quantifier Registration ===

    @classmethod
    def register_quantifier(cls, name: str, **metadata) -> Callable:
        """
        Decorator to register a quantifier class.

        Args:
            name: Unique name for the quantifier
            **metadata: Additional metadata

        Returns:
            Decorator function
        """
        def decorator(cls_to_register: Type) -> Type:
            ModuleRegistry.add_quantifier(name, cls_to_register, **metadata)
            return cls_to_register
        return decorator

    @classmethod
    def add_quantifier(cls, name: str, quantifier_class: Type, **metadata) -> None:
        """
        Register a quantifier class.

        Args:
            name: Unique name for the quantifier
            quantifier_class: The quantifier class to register
            **metadata: Additional metadata
        """
        with cls._lock:
            if name in cls._quantifiers:
                logging.warning(f"Overwriting existing quantifier: {name}")
            cls._quantifiers[name] = quantifier_class
            cls._quantifier_metadata[name] = {
                "class_name": quantifier_class.__name__,
                "module": quantifier_class.__module__,
                **metadata
            }
            logging.debug(f"Registered quantifier: {name}")

    @classmethod
    def get_quantifier(cls, name: str) -> Optional[Type]:
        """
        Get a quantifier class by name.

        Args:
            name: Quantifier name

        Returns:
            Quantifier class or None
        """
        return cls._quantifiers.get(name)

    @classmethod
    def list_quantifiers(cls) -> List[str]:
        """
        List all registered quantifier names.

        Returns:
            List of quantifier names
        """
        return list(cls._quantifiers.keys())

    @classmethod
    def get_quantifier_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Get metadata for a quantifier.

        Args:
            name: Quantifier name

        Returns:
            Metadata dictionary
        """
        return cls._quantifier_metadata.get(name, {})

    # === Visualizer Registration ===

    @classmethod
    def register_visualizer(cls, name: str, **metadata) -> Callable:
        """
        Decorator to register a visualizer class.

        Args:
            name: Unique name for the visualizer
            **metadata: Additional metadata

        Returns:
            Decorator function
        """
        def decorator(cls_to_register: Type) -> Type:
            ModuleRegistry.add_visualizer(name, cls_to_register, **metadata)
            return cls_to_register
        return decorator

    @classmethod
    def add_visualizer(cls, name: str, visualizer_class: Type, **metadata) -> None:
        """
        Register a visualizer class.

        Args:
            name: Unique name for the visualizer
            visualizer_class: The visualizer class to register
            **metadata: Additional metadata
        """
        with cls._lock:
            if name in cls._visualizers:
                logging.warning(f"Overwriting existing visualizer: {name}")
            cls._visualizers[name] = visualizer_class
            cls._visualizer_metadata[name] = {
                "class_name": visualizer_class.__name__,
                "module": visualizer_class.__module__,
                **metadata
            }
            logging.debug(f"Registered visualizer: {name}")

    @classmethod
    def get_visualizer(cls, name: str) -> Optional[Type]:
        """
        Get a visualizer class by name.

        Args:
            name: Visualizer name

        Returns:
            Visualizer class or None
        """
        return cls._visualizers.get(name)

    @classmethod
    def list_visualizers(cls) -> List[str]:
        """
        List all registered visualizer names.

        Returns:
            List of visualizer names
        """
        return list(cls._visualizers.keys())

    @classmethod
    def get_visualizer_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Get metadata for a visualizer.

        Args:
            name: Visualizer name

        Returns:
            Metadata dictionary
        """
        return cls._visualizer_metadata.get(name, {})

    # === Pipeline Registration ===

    @classmethod
    def register_pipeline(cls, name: str, **metadata) -> Callable:
        """
        Decorator to register a pipeline class.

        Args:
            name: Unique name for the pipeline
            **metadata: Additional metadata

        Returns:
            Decorator function
        """
        def decorator(cls_to_register: Type) -> Type:
            ModuleRegistry.add_pipeline(name, cls_to_register, **metadata)
            return cls_to_register
        return decorator

    @classmethod
    def add_pipeline(cls, name: str, pipeline_class: Type, **metadata) -> None:
        """
        Register a pipeline class.

        Args:
            name: Unique name for the pipeline
            pipeline_class: The pipeline class to register
            **metadata: Additional metadata
        """
        with cls._lock:
            if name in cls._pipelines:
                logging.warning(f"Overwriting existing pipeline: {name}")
            cls._pipelines[name] = pipeline_class
            cls._pipeline_metadata[name] = {
                "class_name": pipeline_class.__name__,
                "module": pipeline_class.__module__,
                **metadata
            }
            logging.debug(f"Registered pipeline: {name}")

    @classmethod
    def get_pipeline(cls, name: str) -> Optional[Type]:
        """
        Get a pipeline class by name.

        Args:
            name: Pipeline name

        Returns:
            Pipeline class or None
        """
        return cls._pipelines.get(name)

    @classmethod
    def list_pipelines(cls) -> List[str]:
        """
        List all registered pipeline names.

        Returns:
            List of pipeline names
        """
        return list(cls._pipelines.keys())

    # === Utility Methods ===

    @classmethod
    def clear_all(cls) -> None:
        """Clear all registered components."""
        with cls._lock:
            cls._evaluators.clear()
            cls._quantifiers.clear()
            cls._visualizers.clear()
            cls._pipelines.clear()
            cls._evaluator_metadata.clear()
            cls._quantifier_metadata.clear()
            cls._visualizer_metadata.clear()
            cls._pipeline_metadata.clear()
            logging.info("All registrations cleared")

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """
        Get a summary of all registered components.

        Returns:
            Dictionary with component counts and lists
        """
        return {
            "evaluators": {
                "count": len(cls._evaluators),
                "names": cls.list_evaluators()
            },
            "quantifiers": {
                "count": len(cls._quantifiers),
                "names": cls.list_quantifiers()
            },
            "visualizers": {
                "count": len(cls._visualizers),
                "names": cls.list_visualizers()
            },
            "pipelines": {
                "count": len(cls._pipelines),
                "names": cls.list_pipelines()
            }
        }


# Convenience functions for decorator-style registration
def register_evaluator(name: str, **metadata) -> Callable:
    """Decorator to register an evaluator."""
    return ModuleRegistry.register_evaluator(name, **metadata)


def register_quantifier(name: str, **metadata) -> Callable:
    """Decorator to register a quantifier."""
    return ModuleRegistry.register_quantifier(name, **metadata)


def register_visualizer(name: str, **metadata) -> Callable:
    """Decorator to register a visualizer."""
    return ModuleRegistry.register_visualizer(name, **metadata)


def register_pipeline(name: str, **metadata) -> Callable:
    """Decorator to register a pipeline."""
    return ModuleRegistry.register_pipeline(name, **metadata)
