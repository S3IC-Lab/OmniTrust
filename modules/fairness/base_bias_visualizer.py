"""
Base Bias Visualizer
====================

Provides the abstract base class for bias visualizers that generate
charts and reports for bias analysis results.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .bias_types import BiasType, BiasEvaluationResult
from .base_bias_quantifier import BiasQuantificationResult
from ..base_visualizer import BaseVisualizer


class BiasVisualizerInterface(ABC):
    """
    Abstract interface for bias visualizers.
    """

    @abstractmethod
    def visualize(
        self,
        result: BiasQuantificationResult,
        output_path: str,
        **kwargs
    ) -> str:
        """
        Generate visualization from bias results.

        Args:
            result: BiasQuantificationResult to visualize
            output_path: Path for output files
            **kwargs: Additional visualization parameters

        Returns:
            Path to the generated visualization directory
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported output formats.

        Returns:
            List of format strings
        """
        pass


class BaseBiasVisualizer(BaseVisualizer, BiasVisualizerInterface):
    """
    Base implementation for bias visualizers.

    Provides common functionality for:
    - Bias distribution charts
    - Group comparison visualizations
    - Heatmaps for pairwise analysis
    - Summary reports

    Subclasses should override:
    - _generate_charts(): Chart generation logic
    - _generate_report(): Report generation
    - get_supported_formats(): List of supported formats

    Example:
        class DemographicBiasVisualizer(BaseBiasVisualizer):
            def __init__(self, **kwargs):
                super().__init__(name="demographic_visualizer", **kwargs)

            def _generate_charts(self, data, output_dir, config):
                # Generate demographic bias charts
                return [output_dir / "demographic_bias.png"]
    """

    # Bias-specific color schemes
    BIAS_COLORS = {
        'low': '#2ca02c',      # Green
        'moderate': '#ff7f0e',  # Orange
        'high': '#d62728',      # Red
        'severe': '#7f0000',    # Dark red
    }

    GROUP_COLORS = {
        'group_a': '#1f77b4',   # Blue
        'group_b': '#ff7f0e',   # Orange
        'group_c': '#2ca02c',   # Green
        'group_d': '#d62728',   # Red
    }

    def __init__(
        self,
        name: str = "bias_visualizer",
        version: str = "1.0.0",
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the bias visualizer.

        Args:
            name: Visualizer name
            version: Version string
            description: Brief description
            config: Configuration dictionary
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            version=version,
            description=description,
            config=config,
            **kwargs
        )

        # Default configuration for bias visualization
        self._config.setdefault('show_confidence_intervals', True)
        self._config.setdefault('show_significance', True)
        self._config.setdefault('chart_types', ['bar', 'heatmap', 'radar'])

    def _extract_data(self, result: Any) -> Dict[str, Any]:
        """
        Extract visualization data from bias result.

        Args:
            result: BiasQuantificationResult or BiasEvaluationResult

        Returns:
            Dictionary of data for visualization
        """
        if hasattr(result, 'to_dict'):
            data = result.to_dict()
        elif isinstance(result, dict):
            data = result
        else:
            data = {"raw": result}

        # Add bias-specific metadata
        if hasattr(result, 'get_bias_level'):
            data['bias_level'] = result.get_bias_level()

        return data

    def _generate_charts(
        self,
        data: Dict[str, Any],
        output_dir: Path,
        config: Dict[str, Any]
    ) -> List[Path]:
        """
        Generate bias visualization charts.

        Args:
            data: Data dictionary for visualization
            output_dir: Output directory path
            config: Visualization configuration

        Returns:
            List of paths to generated chart files
        """
        chart_paths = []
        chart_types = config.get('chart_types', ['bar', 'heatmap'])

        # Get metrics
        metrics = data.get('quantified_metrics', {})
        if not metrics:
            metrics = data.get('metrics', {})

        # Generate bar chart for metrics
        if 'bar' in chart_types and metrics:
            bar_path = self._generate_metrics_bar_chart(metrics, output_dir, config)
            if bar_path:
                chart_paths.append(bar_path)

        # Generate group comparison chart
        group_metrics = data.get('group_metrics', {})
        if group_metrics and 'bar' in chart_types:
            group_path = self._generate_group_comparison_chart(group_metrics, output_dir, config)
            if group_path:
                chart_paths.append(group_path)

        # Generate heatmap for pairwise comparisons
        comparisons = data.get('group_comparisons', {})
        if not comparisons:
            comparisons = data.get('pairwise_comparisons', {})
        if comparisons and 'heatmap' in chart_types:
            heatmap_path = self._generate_comparison_heatmap(comparisons, output_dir, config)
            if heatmap_path:
                chart_paths.append(heatmap_path)

        # Generate bias level indicator
        bias_level = data.get('bias_level', 'unknown')
        if isinstance(data.get('bias_summary'), dict):
            bias_level = data['bias_summary'].get('bias_level', bias_level)

        level_path = self._generate_bias_level_indicator(bias_level, output_dir, config)
        if level_path:
            chart_paths.append(level_path)

        return chart_paths

    def _generate_metrics_bar_chart(
        self,
        metrics: Dict[str, float],
        output_dir: Path,
        config: Dict[str, Any]
    ) -> Optional[Path]:
        """
        Generate bar chart for bias metrics.

        Args:
            metrics: Dictionary of metrics
            output_dir: Output directory
            config: Configuration

        Returns:
            Path to chart file
        """
        if not metrics:
            return None

        output_path = output_dir / "bias_metrics.png"

        labels = list(metrics.keys())
        values = list(metrics.values())

        return self._create_bar_chart(
            labels=labels,
            values=values,
            title="Bias Metrics",
            xlabel="Metric",
            ylabel="Value",
            output_path=output_path
        )

    def _generate_group_comparison_chart(
        self,
        group_metrics: Dict[str, Dict[str, float]],
        output_dir: Path,
        config: Dict[str, Any]
    ) -> Optional[Path]:
        """
        Generate chart comparing metrics across groups.

        Args:
            group_metrics: Metrics by group
            output_dir: Output directory
            config: Configuration

        Returns:
            Path to chart file
        """
        if not group_metrics:
            return None

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            output_path = output_dir / "group_comparison.png"

            groups = list(group_metrics.keys())
            if not groups:
                return None

            # Get all metrics
            all_metrics = set()
            for metrics in group_metrics.values():
                all_metrics.update(metrics.keys())
            metrics_list = sorted(all_metrics)

            # Prepare data
            x = np.arange(len(metrics_list))
            width = 0.8 / len(groups)

            fig, ax = plt.subplots(figsize=(12, 6))

            for i, group in enumerate(groups):
                values = [group_metrics[group].get(m, 0) for m in metrics_list]
                color = list(self.GROUP_COLORS.values())[i % len(self.GROUP_COLORS)]
                ax.bar(x + i * width, values, width, label=group, color=color)

            ax.set_xlabel('Metric')
            ax.set_ylabel('Value')
            ax.set_title('Bias Metrics by Group')
            ax.set_xticks(x + width * (len(groups) - 1) / 2)
            ax.set_xticklabels(metrics_list, rotation=45, ha='right')
            ax.legend()

            plt.tight_layout()
            plt.savefig(output_path, dpi=self._config.get('dpi', 150))
            plt.close(fig)

            return output_path

        except ImportError:
            self._logger.warning("matplotlib not installed")
            return None

    def _generate_comparison_heatmap(
        self,
        comparisons: Dict[str, Any],
        output_dir: Path,
        config: Dict[str, Any]
    ) -> Optional[Path]:
        """
        Generate heatmap for pairwise comparisons.

        Args:
            comparisons: Pairwise comparison data
            output_dir: Output directory
            config: Configuration

        Returns:
            Path to chart file
        """
        if not comparisons:
            return None

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            output_path = output_dir / "comparison_heatmap.png"

            # Extract comparison pairs
            pairs = list(comparisons.keys())
            if not pairs:
                return None

            # Get metrics from first comparison
            first_comp = comparisons[pairs[0]]
            if isinstance(first_comp, dict):
                metrics = list(first_comp.keys())
            else:
                return None

            # Build data matrix
            data = []
            for pair in pairs:
                row = []
                for metric in metrics:
                    val = comparisons[pair].get(metric, {})
                    if isinstance(val, dict):
                        row.append(abs(val.get('difference', 0)))
                    else:
                        row.append(abs(val) if isinstance(val, (int, float)) else 0)
                data.append(row)

            return self._create_heatmap(
                data=data,
                x_labels=metrics,
                y_labels=pairs,
                title="Pairwise Comparison Heatmap",
                output_path=output_path
            )

        except Exception as e:
            self._logger.warning(f"Failed to generate heatmap: {e}")
            return None

    def _generate_bias_level_indicator(
        self,
        bias_level: str,
        output_dir: Path,
        config: Dict[str, Any]
    ) -> Optional[Path]:
        """
        Generate visual indicator of overall bias level.

        Args:
            bias_level: Bias level string
            output_dir: Output directory
            config: Configuration

        Returns:
            Path to chart file
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            output_path = output_dir / "bias_level_indicator.png"

            fig, ax = plt.subplots(figsize=(8, 2))

            # Define levels and positions
            levels = ['none', 'low', 'moderate', 'high', 'severe']
            colors = ['#2ca02c', '#7fbf7f', '#ff7f0e', '#d62728', '#7f0000']

            # Draw level indicator
            for i, (level, color) in enumerate(zip(levels, colors)):
                rect = mpatches.Rectangle(
                    (i * 0.2, 0), 0.18, 1,
                    facecolor=color,
                    edgecolor='black' if level == bias_level else 'gray',
                    linewidth=3 if level == bias_level else 1
                )
                ax.add_patch(rect)
                ax.text(i * 0.2 + 0.09, -0.2, level.capitalize(),
                        ha='center', va='top', fontsize=10)

            # Add marker for current level
            if bias_level in levels:
                idx = levels.index(bias_level)
                ax.plot(idx * 0.2 + 0.09, 1.15, 'v', color='black', markersize=15)

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.5, 1.3)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title('Bias Level Indicator', fontsize=14, pad=20)

            plt.tight_layout()
            plt.savefig(output_path, dpi=self._config.get('dpi', 150))
            plt.close(fig)

            return output_path

        except ImportError:
            self._logger.warning("matplotlib not installed")
            return None

    def _generate_report(
        self,
        data: Dict[str, Any],
        chart_paths: List[Path],
        output_dir: Path,
        config: Dict[str, Any]
    ) -> Optional[Path]:
        """
        Generate a summary report for bias analysis.

        Args:
            data: Data dictionary
            chart_paths: Paths to generated charts
            output_dir: Output directory
            config: Configuration

        Returns:
            Path to report file
        """
        output_path = output_dir / "bias_report.md"

        try:
            report_lines = [
                "# Bias Evaluation Report",
                "",
                f"**Generated:** {data.get('timestamp', 'N/A')}",
                "",
                "## Summary",
                ""
            ]

            # Add bias summary if available
            summary = data.get('bias_summary', {})
            if summary:
                report_lines.extend([
                    f"- **Overall Bias Score:** {summary.get('overall_bias_score', 'N/A'):.4f}",
                    f"- **Bias Level:** {summary.get('bias_level', 'N/A')}",
                    f"- **Interpretation:** {summary.get('interpretation', 'N/A')}",
                    f"- **Statistically Significant:** {summary.get('statistically_significant', 'N/A')}",
                    ""
                ])

            # Add metrics section
            metrics = data.get('quantified_metrics', {})
            if not metrics:
                metrics = data.get('metrics', {})

            if metrics:
                report_lines.extend([
                    "## Metrics",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|"
                ])
                for k, v in metrics.items():
                    report_lines.append(f"| {k} | {v:.4f} |")
                report_lines.append("")

            # Add recommendations
            recommendations = summary.get('recommendations', [])
            if recommendations:
                report_lines.extend([
                    "## Recommendations",
                    ""
                ])
                for rec in recommendations:
                    report_lines.append(f"- {rec}")
                report_lines.append("")

            # Add chart references
            if chart_paths:
                report_lines.extend([
                    "## Visualizations",
                    ""
                ])
                for path in chart_paths:
                    if path:
                        report_lines.append(f"- [{path.name}]({path.name})")
                report_lines.append("")

            # Write report
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))

            return output_path

        except Exception as e:
            self._logger.error(f"Failed to generate report: {e}")
            return None

    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        return ['png', 'pdf', 'svg', 'html', 'md']
