#!/usr/bin/env python3
"""
Simple script to compute accuracy for ranking and matching tasks,
both including and excluding 'Unknown' answers.
"""

import json
import argparse
from typing import Dict, Any


def analyze_results(results_file: str) -> None:
    """Analyze results and compute accuracy metrics."""
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Initialize counters
    stats = {
        'ranking': {
            'total': 0,
            'correct': 0,
            'unknown': 0,
            'total_excluding_unknown': 0,
            'correct_excluding_unknown': 0
        },
        'matching': {
            'total': 0,
            'correct': 0,
            'unknown': 0,
            'total_excluding_unknown': 0,
            'correct_excluding_unknown': 0
        }
    }
    
    # Process each result
    for result in results:
        task = result['task']
        is_correct = result['is_correct']
        extracted_answer = result['extracted_answer']
        
        # Update total counts
        stats[task]['total'] += 1
        
        # Check if answer is unknown
        is_unknown = extracted_answer == 'Unknown'
        
        if is_unknown:
            stats[task]['unknown'] += 1
        else:
            stats[task]['total_excluding_unknown'] += 1
            if is_correct:
                stats[task]['correct_excluding_unknown'] += 1
        
        # Update correct counts (including unknowns)
        if is_correct:
            stats[task]['correct'] += 1
    
    # Calculate accuracies
    def safe_divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator > 0 else 0.0
    
    print("=" * 60)
    print("HUMOR EVALUATION RESULTS ANALYSIS")
    print("=" * 60)
    print()
    
    # Overall summary
    total_all = sum(stats[task]['total'] for task in stats)
    correct_all = sum(stats[task]['correct'] for task in stats)
    unknown_all = sum(stats[task]['unknown'] for task in stats)
    
    print(f"OVERALL SUMMARY:")
    print(f"  Total entries: {total_all}")
    print(f"  Unknown answers: {unknown_all} ({unknown_all/total_all*100:.1f}%)")
    print(f"  Accuracy (including unknowns): {correct_all}/{total_all} = {safe_divide(correct_all, total_all):.3f} ({safe_divide(correct_all, total_all)*100:.1f}%)")
    print(f"  Accuracy (excluding unknowns): {correct_all}/{total_all - unknown_all} = {safe_divide(correct_all, total_all - unknown_all):.3f} ({safe_divide(correct_all, total_all - unknown_all)*100:.1f}%)")
    print()
    
    # Task-specific analysis
    for task_name in ['ranking', 'matching']:
        task_stats = stats[task_name]
        
        print(f"{task_name.upper()} TASK:")
        print(f"  Total entries: {task_stats['total']}")
        print(f"  Unknown answers: {task_stats['unknown']} ({safe_divide(task_stats['unknown'], task_stats['total'])*100:.1f}%)")
        print(f"  Accuracy (including unknowns): {task_stats['correct']}/{task_stats['total']} = {safe_divide(task_stats['correct'], task_stats['total']):.3f} ({safe_divide(task_stats['correct'], task_stats['total'])*100:.1f}%)")
        print(f"  Accuracy (excluding unknowns): {task_stats['correct_excluding_unknown']}/{task_stats['total_excluding_unknown']} = {safe_divide(task_stats['correct_excluding_unknown'], task_stats['total_excluding_unknown']):.3f} ({safe_divide(task_stats['correct_excluding_unknown'], task_stats['total_excluding_unknown'])*100:.1f}%)")
        print()
    
    # Summary table
    print("SUMMARY TABLE:")
    print("-" * 80)
    print(f"{'Task':<10} | {'Total':<6} | {'Unknown':<8} | {'Acc (w/ Unknown)':<15} | {'Acc (w/o Unknown)':<16}")
    print("-" * 80)
    
    for task_name in ['ranking', 'matching']:
        task_stats = stats[task_name]
        acc_with = safe_divide(task_stats['correct'], task_stats['total'])
        acc_without = safe_divide(task_stats['correct_excluding_unknown'], task_stats['total_excluding_unknown'])
        
        print(f"{task_name:<10} | {task_stats['total']:<6} | {task_stats['unknown']:<8} | {acc_with:.3f} ({acc_with*100:.1f}%) | {acc_without:.3f} ({acc_without*100:.1f}%)")
    
    # Overall row
    acc_with_all = safe_divide(correct_all, total_all)
    acc_without_all = safe_divide(correct_all, total_all - unknown_all)
    print("-" * 80)
    print(f"{'OVERALL':<10} | {total_all:<6} | {unknown_all:<8} | {acc_with_all:.3f} ({acc_with_all*100:.1f}%) | {acc_without_all:.3f} ({acc_without_all*100:.1f}%)")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze humor evaluation results")
    parser.add_argument("results_file", help="Path to the results JSON file", default="results.json")
    args = parser.parse_args()
    
    try:
        analyze_results(args.results_file)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{args.results_file}'.")
    except KeyError as e:
        print(f"Error: Missing expected key in results file: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
