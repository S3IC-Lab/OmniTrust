"""
Evaluation Utilities
====================

Provides utility functions for evaluation workflows, result processing,
and batch operations.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

logger = logging.getLogger(__name__)


def batch_evaluate(
    model: Any,
    datasets: List[Any],
    evaluator: Any,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict[str, Any]]:
    """
    Perform batch evaluation across multiple datasets.

    Args:
        model: Model to evaluate
        datasets: List of datasets
        evaluator: Evaluator instance
        max_workers: Maximum parallel workers
        progress_callback: Optional callback for progress updates

    Returns:
        List of evaluation results
    """
    results = []
    total = len(datasets)

    def evaluate_single(idx_dataset):
        idx, dataset = idx_dataset
        try:
            result = evaluator.evaluate(model, dataset)
            return idx, result, None
        except Exception as e:
            return idx, None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_single, (i, ds)): i
            for i, ds in enumerate(datasets)
        }

        completed = 0
        results = [None] * total

        for future in as_completed(futures):
            idx, result, error = future.result()
            if error:
                logger.error(f"Evaluation {idx} failed: {error}")
                results[idx] = {"error": error}
            else:
                results[idx] = result

            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    return results


def aggregate_results(
    results: List[Dict[str, Any]],
    aggregation_method: str = 'mean'
) -> Dict[str, Any]:
    """
    Aggregate multiple evaluation results.

    Args:
        results: List of evaluation result dictionaries
        aggregation_method: Method for aggregating metrics ('mean', 'median', 'max', 'min')

    Returns:
        Aggregated results dictionary
    """
    if not results:
        return {}

    # Extract metrics from all results
    all_metrics = {}
    for result in results:
        if isinstance(result, dict) and 'metrics' in result:
            for key, value in result['metrics'].items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

    # Aggregate metrics
    aggregated_metrics = {}
    for key, values in all_metrics.items():
        if not values:
            continue

        if aggregation_method == 'mean':
            aggregated_metrics[key] = sum(values) / len(values)
        elif aggregation_method == 'median':
            sorted_values = sorted(values)
            mid = len(sorted_values) // 2
            if len(sorted_values) % 2 == 0:
                aggregated_metrics[key] = (sorted_values[mid-1] + sorted_values[mid]) / 2
            else:
                aggregated_metrics[key] = sorted_values[mid]
        elif aggregation_method == 'max':
            aggregated_metrics[key] = max(values)
        elif aggregation_method == 'min':
            aggregated_metrics[key] = min(values)

    return {
        'aggregated_metrics': aggregated_metrics,
        'aggregation_method': aggregation_method,
        'num_results': len(results),
        'timestamp': datetime.now().isoformat()
    }


def format_results(
    results: Dict[str, Any],
    format_type: str = 'json',
    include_raw: bool = False
) -> str:
    """
    Format evaluation results for display or export.

    Args:
        results: Evaluation results dictionary
        format_type: Output format ('json', 'markdown', 'csv', 'table')
        include_raw: Whether to include raw results

    Returns:
        Formatted string representation
    """
    if format_type == 'json':
        return json.dumps(results, indent=2, ensure_ascii=False, default=str)

    elif format_type == 'markdown':
        lines = ["# Evaluation Results", ""]

        if 'timestamp' in results:
            lines.append(f"**Timestamp:** {results['timestamp']}")
            lines.append("")

        if 'metrics' in results:
            lines.extend(["## Metrics", "", "| Metric | Value |", "|--------|-------|"])
            for key, value in results['metrics'].items():
                if isinstance(value, float):
                    lines.append(f"| {key} | {value:.4f} |")
                else:
                    lines.append(f"| {key} | {value} |")
            lines.append("")

        if include_raw and 'raw_results' in results:
            lines.extend(["## Raw Results", "", "```json"])
            lines.append(json.dumps(results['raw_results'], indent=2, default=str))
            lines.append("```")

        return "\n".join(lines)

    elif format_type == 'csv':
        lines = []
        if 'metrics' in results:
            lines.append("metric,value")
            for key, value in results['metrics'].items():
                lines.append(f"{key},{value}")
        return "\n".join(lines)

    elif format_type == 'table':
        lines = []
        if 'metrics' in results:
            max_key_len = max(len(k) for k in results['metrics'].keys())
            lines.append("-" * (max_key_len + 20))
            lines.append(f"{'Metric':<{max_key_len}} | Value")
            lines.append("-" * (max_key_len + 20))
            for key, value in results['metrics'].items():
                if isinstance(value, float):
                    lines.append(f"{key:<{max_key_len}} | {value:.4f}")
                else:
                    lines.append(f"{key:<{max_key_len}} | {value}")
            lines.append("-" * (max_key_len + 20))
        return "\n".join(lines)

    return str(results)


def save_results(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    format_type: str = 'json'
) -> Path:
    """
    Save evaluation results to file.

    Args:
        results: Evaluation results dictionary
        output_path: Path to save results
        format_type: Output format ('json', 'markdown', 'csv')

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add extension if not present
    if not output_path.suffix:
        ext_map = {'json': '.json', 'markdown': '.md', 'csv': '.csv'}
        output_path = output_path.with_suffix(ext_map.get(format_type, '.json'))

    formatted = format_results(results, format_type)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted)

    logger.info(f"Results saved to: {output_path}")
    return output_path


def load_results(
    input_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Load evaluation results from file.

    Args:
        input_path: Path to results file

    Returns:
        Loaded results dictionary
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if input_path.suffix == '.json':
        return json.loads(content)
    else:
        # Return raw content for non-JSON files
        return {"raw_content": content}


def compare_results(
    results_a: Dict[str, Any],
    results_b: Dict[str, Any],
    metrics_to_compare: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare two evaluation results.

    Args:
        results_a: First evaluation results
        results_b: Second evaluation results
        metrics_to_compare: Optional list of specific metrics to compare

    Returns:
        Comparison results dictionary
    """
    metrics_a = results_a.get('metrics', {})
    metrics_b = results_b.get('metrics', {})

    if metrics_to_compare is None:
        metrics_to_compare = list(set(metrics_a.keys()) | set(metrics_b.keys()))

    comparison = {}
    for metric in metrics_to_compare:
        val_a = metrics_a.get(metric)
        val_b = metrics_b.get(metric)

        if val_a is not None and val_b is not None:
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                diff = val_a - val_b
                pct_diff = (diff / val_b * 100) if val_b != 0 else float('inf')
                comparison[metric] = {
                    'value_a': val_a,
                    'value_b': val_b,
                    'difference': diff,
                    'percent_difference': pct_diff,
                    'better': 'a' if val_a > val_b else ('b' if val_b > val_a else 'equal')
                }
            else:
                comparison[metric] = {
                    'value_a': val_a,
                    'value_b': val_b,
                    'equal': val_a == val_b
                }

    return {
        'comparison': comparison,
        'summary': {
            'metrics_compared': len(comparison),
            'a_better_count': sum(1 for v in comparison.values() if v.get('better') == 'a'),
            'b_better_count': sum(1 for v in comparison.values() if v.get('better') == 'b'),
            'equal_count': sum(1 for v in comparison.values() if v.get('better') == 'equal' or v.get('equal'))
        }
    }


def filter_results(
    results: List[Dict[str, Any]],
    filter_fn: Callable[[Dict[str, Any]], bool]
) -> List[Dict[str, Any]]:
    """
    Filter evaluation results based on criteria.

    Args:
        results: List of evaluation results
        filter_fn: Function that returns True for results to keep

    Returns:
        Filtered list of results
    """
    return [r for r in results if filter_fn(r)]


def sort_results(
    results: List[Dict[str, Any]],
    sort_key: str,
    descending: bool = True
) -> List[Dict[str, Any]]:
    """
    Sort evaluation results by a metric.

    Args:
        results: List of evaluation results
        sort_key: Metric name to sort by
        descending: Sort in descending order

    Returns:
        Sorted list of results
    """
    def get_sort_value(result):
        metrics = result.get('metrics', {})
        return metrics.get(sort_key, 0)

    return sorted(results, key=get_sort_value, reverse=descending)


def merge_results(
    results_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge multiple result dictionaries.

    Args:
        results_list: List of result dictionaries to merge

    Returns:
        Merged results dictionary
    """
    merged = {
        'merged_from': len(results_list),
        'timestamp': datetime.now().isoformat(),
        'individual_results': results_list,
        'combined_metrics': {}
    }

    # Combine all metrics
    for result in results_list:
        if 'metrics' in result:
            for key, value in result['metrics'].items():
                if key not in merged['combined_metrics']:
                    merged['combined_metrics'][key] = []
                merged['combined_metrics'][key].append(value)

    return merged
