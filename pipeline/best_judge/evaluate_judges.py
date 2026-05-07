#!/usr/bin/env python3
"""
Script to evaluate judge performance on positive and negative constraint satisfaction.

This script:
1. Iterates over all judge folders in the input directory
2. Evaluates judgments on positive.jsonl (gold: Yes) and negative.jsonl (gold: No)
3. Computes performance metrics for each judge
4. Saves results to output directory
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load all lines from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_verdict(judge_text: str) -> Optional[str]:
    """
    Extract the final verdict from judge text.
    
    Looks for the last occurrence of "Final Verdict: [[Yes/No]]".
    If not found, allows for missing brackets and colon.
    
    Args:
        judge_text: The judge's output text
        
    Returns:
        "Yes", "No", or None if no verdict found
    """
    # 1. Try to find the strict pattern (Last occurrence)
    strict_pattern = r"Final Verdict:\s*\[\[(Yes|No)\]\]"
    strict_matches = re.findall(strict_pattern, judge_text, re.IGNORECASE)
    
    if strict_matches:
        return strict_matches[-1].capitalize()
    
    # 2. If no strict match, try more flexible pattern (Last occurrence)
    # Allows missing brackets and/or colon
    flexible_pattern = r"Final Verdict:?\s*\[*\[*(Yes|No)\]*\]*"
    flexible_matches = re.findall(flexible_pattern, judge_text, re.IGNORECASE)
    
    if flexible_matches:
        return flexible_matches[-1].capitalize()
    
    return None


def evaluate_file(file_path: Path, gold_label: str) -> Dict:
    """
    Evaluate judgments in a file against the gold label.
    
    Args:
        file_path: Path to the JSONL file with judgments
        gold_label: The gold label for all entries ("Yes" or "No")
        
    Returns:
        Dictionary with evaluation metrics
    """
    data = load_jsonl(file_path)
    
    total = len(data)
    correct = 0
    incorrect = 0
    no_verdict = 0
    
    predictions = []
    gold_labels = []
    
    for item in data:
        judge_text = item.get('judge', '')
        verdict = extract_verdict(judge_text)
        
        if verdict is None:
            no_verdict += 1
            # For metrics calculation, treat no verdict as incorrect
            predictions.append("Unknown")
            gold_labels.append(gold_label)
        else:
            predictions.append(verdict)
            gold_labels.append(gold_label)
            
            if verdict == gold_label:
                correct += 1
            else:
                incorrect += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate confusion matrix elements
    tp = sum(1 for p, g in zip(predictions, gold_labels) if p == "Yes" and g == "Yes")
    tn = sum(1 for p, g in zip(predictions, gold_labels) if p == "No" and g == "No")
    fp = sum(1 for p, g in zip(predictions, gold_labels) if p == "Yes" and g == "No")
    fn = sum(1 for p, g in zip(predictions, gold_labels) if p == "No" and g == "Yes")
    
    # Calculate precision, recall, F1 for "Yes" class
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'no_verdict': no_verdict,
        'accuracy': accuracy,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_judge(judge_dir: Path) -> Dict:
    """
    Evaluate a judge's performance on positive and negative files.
    
    Args:
        judge_dir: Path to the judge's directory
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    positive_file = judge_dir / "positive.jsonl"
    negative_file = judge_dir / "negative.jsonl"
    
    results = {
        'judge_name': judge_dir.name,
        'has_positive': positive_file.exists(),
        'has_negative': negative_file.exists()
    }
    
    # Evaluate positive file (gold: Yes)
    if positive_file.exists():
        results['positive'] = evaluate_file(positive_file, "Yes")
    else:
        results['positive'] = None
    
    # Evaluate negative file (gold: No)
    if negative_file.exists():
        results['negative'] = evaluate_file(negative_file, "No")
    else:
        results['negative'] = None
    
    # Calculate overall metrics
    if results['positive'] and results['negative']:
        pos = results['positive']
        neg = results['negative']
        
        total_correct = pos['correct'] + neg['correct']
        total_samples = pos['total'] + neg['total']
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # Overall confusion matrix
        total_tp = pos['true_positives'] + neg['true_positives']
        total_tn = pos['true_negatives'] + neg['true_negatives']
        total_fp = pos['false_positives'] + neg['false_positives']
        total_fn = pos['false_negatives'] + neg['false_negatives']
        
        # Overall precision, recall, F1
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        results['overall'] = {
            'total': total_samples,
            'correct': total_correct,
            'incorrect': pos['incorrect'] + neg['incorrect'],
            'no_verdict': pos['no_verdict'] + neg['no_verdict'],
            'accuracy': overall_accuracy,
            'true_positives': total_tp,
            'true_negatives': total_tn,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        }
    else:
        results['overall'] = None
    
    return results


def format_metrics(metrics: Optional[Dict], indent: str = "") -> str:
    """Format metrics dictionary as a readable string."""
    if metrics is None:
        return f"{indent}No data available\n"
    
    lines = []
    lines.append(f"{indent}Total samples:     {metrics['total']}")
    lines.append(f"{indent}Correct:           {metrics['correct']} ({metrics['accuracy']*100:.2f}%)")
    lines.append(f"{indent}Incorrect:         {metrics['incorrect']}")
    lines.append(f"{indent}No verdict:        {metrics['no_verdict']}")
    lines.append(f"{indent}")
    lines.append(f"{indent}Confusion Matrix:")
    lines.append(f"{indent}  True Positives:  {metrics['true_positives']}")
    lines.append(f"{indent}  True Negatives:  {metrics['true_negatives']}")
    lines.append(f"{indent}  False Positives: {metrics['false_positives']}")
    lines.append(f"{indent}  False Negatives: {metrics['false_negatives']}")
    lines.append(f"{indent}")
    lines.append(f"{indent}Precision:         {metrics['precision']:.4f}")
    lines.append(f"{indent}Recall:            {metrics['recall']:.4f}")
    lines.append(f"{indent}F1 Score:          {metrics['f1']:.4f}")
    
    return "\n".join(lines)


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all judge directories (directories containing positive.jsonl or negative.jsonl)
    judge_dirs = []
    for path in sorted(input_dir.iterdir()):
        if path.is_dir() and path.name != 'logs':
            # Check if it contains positive or negative files
            if (path / "positive.jsonl").exists() or (path / "negative.jsonl").exists():
                judge_dirs.append(path)
    
    if not judge_dirs:
        raise ValueError(f"No judge directories found in {input_dir}")
    
    print("=" * 80)
    print("Judge Performance Evaluation")
    print("=" * 80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(judge_dirs)} judge(s):")
    for d in judge_dirs:
        print(f"  - {d.name}")
    print()
    
    # Evaluate each judge
    all_results = []
    for judge_dir in judge_dirs:
        print(f"\nEvaluating judge: {judge_dir.name}")
        results = evaluate_judge(judge_dir)
        all_results.append(results)
        
        # Print summary
        if results['has_positive']:
            print(f"  Positive file: {results['positive']['accuracy']*100:.2f}% accuracy")
        if results['has_negative']:
            print(f"  Negative file: {results['negative']['accuracy']*100:.2f}% accuracy")
        if results['overall']:
            print(f"  Overall: {results['overall']['accuracy']*100:.2f}% accuracy")
    
    # Save detailed results to JSON
    output_json = output_dir / "judge_performance.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_json}")
    
    # Save human-readable report
    output_txt = output_dir / "judge_performance.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("JUDGE PERFORMANCE EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for results in all_results:
            f.write("-" * 80 + "\n")
            f.write(f"JUDGE: {results['judge_name']}\n")
            f.write("-" * 80 + "\n\n")
            
            # Positive file results
            f.write("POSITIVE FILE (Gold: Yes - Constraint should be satisfied):\n")
            f.write(format_metrics(results['positive'], "  ") + "\n\n")
            
            # Negative file results
            f.write("NEGATIVE FILE (Gold: No - Constraint should NOT be satisfied):\n")
            f.write(format_metrics(results['negative'], "  ") + "\n\n")
            
            # Overall results
            f.write("OVERALL PERFORMANCE:\n")
            f.write(format_metrics(results['overall'], "  ") + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Evaluation complete. Processed {len(all_results)} judge(s).\n")
        f.write("=" * 80 + "\n")
    
    print(f"Human-readable report saved to: {output_txt}")
    
    # Save summary CSV
    output_csv = output_dir / "judge_performance_summary.csv"
    with open(output_csv, 'w', encoding='utf-8') as f:
        # Header
        f.write("judge_name,positive_accuracy,negative_accuracy,overall_accuracy,")
        f.write("overall_precision,overall_recall,overall_f1,")
        f.write("total_samples,correct,incorrect,no_verdict\n")
        
        # Data rows
        for results in all_results:
            judge_name = results['judge_name']
            
            pos_acc = f"{results['positive']['accuracy']:.4f}" if results['positive'] else "N/A"
            neg_acc = f"{results['negative']['accuracy']:.4f}" if results['negative'] else "N/A"
            
            if results['overall']:
                ov = results['overall']
                f.write(f"{judge_name},{pos_acc},{neg_acc},{ov['accuracy']:.4f},")
                f.write(f"{ov['precision']:.4f},{ov['recall']:.4f},{ov['f1']:.4f},")
                f.write(f"{ov['total']},{ov['correct']},{ov['incorrect']},{ov['no_verdict']}\n")
            else:
                f.write(f"{judge_name},{pos_acc},{neg_acc},N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A\n")
    
    print(f"Summary CSV saved to: {output_csv}")
    
    print()
    print("=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate judge performance on positive and negative constraint satisfaction")
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory containing judge folders (e.g., best_judge/outputs_judge)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory to save evaluation results')  
    args = parser.parse_args()
    main(args)
