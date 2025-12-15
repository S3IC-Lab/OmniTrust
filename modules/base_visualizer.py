"""
Base Visualizer Module
======================

Provides the abstract base class and interfaces for result visualizers
that generate charts, reports, and visual representations of evaluation results.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import uuid


class VisualizerInterface(ABC):
    """
    Abstract interface for all visualizers.

    Visualizers are responsible for:
    - Generating charts and plots
    - Creating visual reports
    - Exporting to various formats (PNG, PDF, HTML)
    """

    @abstractmethod
    def visualize(
        self,
        result: Any,
        output_path: str,
        **kwargs
    ) -> str:
        """
        Generate visualization from results.

        Args:
            result: QuantificationResult or EvaluationResult to visualize
            output_path: Path for output files
            **kwargs: Additional visualization parameters

        Returns:
            Path to the generated visualization
        """
        pass

    @abstractmethod
    def get_visualizer_info(self) -> Dict[str, Any]:
        """
        Get information about this visualizer.

        Returns:
            Dictionary containing visualizer metadata
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported output formats.

        Returns:
            List of format strings (e.g., ['png', 'pdf', 'html'])
        """
        pass


class BaseVisualizer(VisualizerInterface):
    """
    Base implementation of VisualizerInterface with common functionality.

    Provides:
    - Output directory management
    - Format handling
    - Common chart configuration
    - Template system

    Subclasses should override:
    - _generate_charts(): Chart generation logic
    - _generate_report(): Report generation (optional)
    - get_supported_formats(): List of supported formats

    Example:
        class BiasVisualizer(BaseVisualizer):
            def __init__(self, **kwargs):
                super().__init__(name="bias_visualizer", **kwargs)

            def _generate_charts(self, result, output_dir):
                # Generate bias distribution charts
                return ["chart1.png", "chart2.png"]
    """

    # Default chart style configuration
    DEFAULT_STYLE = {
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100,
    }

    # Default color palette
    DEFAULT_COLORS = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Yellow-green
        '#17becf',  # Cyan
    ]

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the base visualizer.

        Args:
            name: Visualizer name
            version: Version string
            description: Brief description
            config: Configuration dictionary
            **kwargs: Additional parameters
        """
        self._id = str(uuid.uuid4())[:8]
        self._name = name
        self._version = version
        self._description = description
        self._config = config or {}
        self._kwargs = kwargs

        # Default configuration
        self._config.setdefault('output_format', 'png')
        self._config.setdefault('dpi', 150)
        self._config.setdefault('style', self.DEFAULT_STYLE.copy())
        self._config.setdefault('colors', self.DEFAULT_COLORS.copy())

        # Initialize logging
        import logging
        self._logger = logging.getLogger(f"{self.__class__.__name__}[{self._id}]")

    @property
    def visualizer_id(self) -> str:
        """Get unique visualizer instance ID."""
        return self._id

    @property
    def name(self) -> str:
        """Get visualizer name."""
        return self._name

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Update visualizer configuration.

        Args:
            config: Configuration dictionary to merge
        """
        self._config.update(config)
        self._logger.debug(f"Configuration updated: {config}")

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()

    def visualize(
        self,
        result: Any,
        output_path: str,
        **kwargs
    ) -> str:
        """
        Generate visualization from results.

        This method orchestrates the visualization process:
        1. Creates output directory
        2. Extracts data from result
        3. Generates charts
        4. Generates report (if enabled)
        5. Returns output path

        Args:
            result: QuantificationResult or EvaluationResult to visualize
            output_path: Path for output files
            **kwargs: Additional visualization parameters

        Returns:
            Path to the generated visualization directory
        """
        self._logger.info(f"Starting visualization: {self._name}")

        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Merge kwargs with config
        vis_config = {**self._config, **kwargs}

        # Extract data
        data = self._extract_data(result)

        # Generate charts
        chart_paths = self._generate_charts(data, output_dir, vis_config)

        # Generate report if enabled
        report_path = None
        if vis_config.get('generate_report', True):
            report_path = self._generate_report(data, chart_paths, output_dir, vis_config)

        # Create manifest
        manifest = {
            "visualizer": self._name,
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(output_dir),
            "charts": [str(p) for p in chart_paths],
            "report": str(report_path) if report_path else None,
            "config": vis_config
        }

        self._save_manifest(manifest, output_dir)

        self._logger.info(f"Visualization completed. Output: {output_dir}")
        return str(output_dir)

    def _extract_data(self, result: Any) -> Dict[str, Any]:
        """
        Extract visualization data from result.

        Args:
            result: QuantificationResult or EvaluationResult

        Returns:
            Dictionary of data for visualization
        """
        if hasattr(result, 'to_dict'):
            return result.to_dict()
        elif isinstance(result, dict):
            return result
        return {"raw": result}

    @abstractmethod
    def _generate_charts(
        self,
        data: Dict[str, Any],
        output_dir: Path,
        config: Dict[str, Any]
    ) -> List[Path]:
        """
        Generate charts from data. Must be implemented by subclasses.

        Args:
            data: Data dictionary for visualization
            output_dir: Output directory path
            config: Visualization configuration

        Returns:
            List of paths to generated chart files
        """
        pass

    def _generate_report(
        self,
        data: Dict[str, Any],
        chart_paths: List[Path],
        output_dir: Path,
        config: Dict[str, Any]
    ) -> Optional[Path]:
        """
        Generate a summary report.

        Override in subclasses for custom report generation.

        Args:
            data: Data dictionary
            chart_paths: Paths to generated charts
            output_dir: Output directory
            config: Visualization configuration

        Returns:
            Path to report file, or None
        """
        return None

    def _save_manifest(self, manifest: Dict[str, Any], output_dir: Path) -> None:
        """
        Save visualization manifest.

        Args:
            manifest: Manifest dictionary
            output_dir: Output directory
        """
        import json
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    def get_visualizer_info(self) -> Dict[str, Any]:
        """Get information about this visualizer."""
        return {
            "id": self._id,
            "name": self._name,
            "version": self._version,
            "description": self._description,
            "supported_formats": self.get_supported_formats(),
            "config": self._config
        }

    # Utility methods for common chart types

    def _create_bar_chart(
        self,
        labels: List[str],
        values: List[float],
        title: str,
        xlabel: str,
        ylabel: str,
        output_path: Path,
        **kwargs
    ) -> Path:
        """
        Create a bar chart.

        Args:
            labels: X-axis labels
            values: Bar values
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            output_path: Output file path
            **kwargs: Additional matplotlib parameters

        Returns:
            Path to saved chart
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))

            colors = kwargs.get('colors', self._config.get('colors', self.DEFAULT_COLORS))
            ax.bar(labels, values, color=colors[:len(labels)])

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            plt.tight_layout()
            plt.savefig(output_path, dpi=self._config.get('dpi', 150))
            plt.close(fig)

            return output_path
        except ImportError:
            self._logger.warning("matplotlib not installed, skipping chart generation")
            return None

    def _create_line_chart(
        self,
        x_values: List,
        y_values: List[List[float]],
        labels: List[str],
        title: str,
        xlabel: str,
        ylabel: str,
        output_path: Path,
        **kwargs
    ) -> Path:
        """
        Create a line chart.

        Args:
            x_values: X-axis values
            y_values: List of Y-axis value lists (one per line)
            labels: Legend labels
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            output_path: Output file path
            **kwargs: Additional matplotlib parameters

        Returns:
            Path to saved chart
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))

            colors = kwargs.get('colors', self._config.get('colors', self.DEFAULT_COLORS))
            for i, (y, label) in enumerate(zip(y_values, labels)):
                ax.plot(x_values, y, label=label, color=colors[i % len(colors)])

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()

            plt.tight_layout()
            plt.savefig(output_path, dpi=self._config.get('dpi', 150))
            plt.close(fig)

            return output_path
        except ImportError:
            self._logger.warning("matplotlib not installed, skipping chart generation")
            return None

    def _create_heatmap(
        self,
        data: List[List[float]],
        x_labels: List[str],
        y_labels: List[str],
        title: str,
        output_path: Path,
        **kwargs
    ) -> Path:
        """
        Create a heatmap.

        Args:
            data: 2D data array
            x_labels: X-axis labels
            y_labels: Y-axis labels
            title: Chart title
            output_path: Output file path
            **kwargs: Additional parameters

        Returns:
            Path to saved chart
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))

            im = ax.imshow(np.array(data), cmap=kwargs.get('cmap', 'RdYlGn_r'))

            ax.set_xticks(range(len(x_labels)))
            ax.set_yticks(range(len(y_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.set_yticklabels(y_labels)

            ax.set_title(title)
            fig.colorbar(im)

            plt.tight_layout()
            plt.savefig(output_path, dpi=self._config.get('dpi', 150))
            plt.close(fig)

            return output_path
        except ImportError:
            self._logger.warning("matplotlib not installed, skipping chart generation")
            return None
