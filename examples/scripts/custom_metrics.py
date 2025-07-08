#!/usr/bin/env python3
"""
Custom Metrics Script for Promptomatix

This script demonstrates how to implement and use custom evaluation metrics:
- Custom quality metrics
- Cost efficiency metrics
- Performance tracking
- Metric comparison and analysis

Usage:
    python custom_metrics.py
"""

import os
import sys
import json
import time
import re
from typing import Dict, List, Any, Callable, Optional
from dotenv import load_dotenv

# Add the src directory to Python path
sys.path.append('../src')

# Import Promptomatix functions
from promptomatix.main import process_input

class CustomMetrics:
    """Custom metrics for prompt optimization evaluation."""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_prompt_clarity_score(self, prompt: str) -> float:
        """Calculate clarity score based on prompt characteristics."""
        score = 0.0
        
        # Length factor (optimal length between 50-200 characters)
        length = len(prompt)
        if 50 <= length <= 200:
            score += 0.3
        elif 20 <= length <= 300:
            score += 0.2
        else:
            score += 0.1
        
        # Specificity indicators
        specificity_indicators = [
            'specific', 'exactly', 'precisely', 'clearly', 'concisely',
            'format', 'structure', 'example', 'sample', 'template'
        ]
        specificity_count = sum(1 for indicator in specificity_indicators if indicator in prompt.lower())
        score += min(specificity_count * 0.1, 0.3)
        
        # Action words (good for clarity)
        action_words = [
            'analyze', 'classify', 'summarize', 'extract', 'generate',
            'identify', 'compare', 'evaluate', 'explain', 'describe'
        ]
        action_count = sum(1 for word in action_words if word in prompt.lower())
        score += min(action_count * 0.05, 0.2)
        
        # Punctuation and structure
        if prompt.count('.') >= 1 and prompt.count(',') >= 1:
            score += 0.1
        
        # Capitalization consistency
        if prompt[0].isupper() and not prompt.isupper():
            score += 0.1
        
        return min(score, 1.0)
    
    def calculate_cost_efficiency_score(self, cost: float, synthetic_data_count: int) -> float:
        """Calculate cost efficiency score."""
        if synthetic_data_count == 0:
            return 0.0
        
        cost_per_example = cost / synthetic_data_count
        
        # Score based on cost per example (lower is better)
        if cost_per_example <= 0.01:
            return 1.0
        elif cost_per_example <= 0.02:
            return 0.8
        elif cost_per_example <= 0.05:
            return 0.6
        elif cost_per_example <= 0.10:
            return 0.4
        else:
            return 0.2
    
    def calculate_time_efficiency_score(self, time_taken: float, synthetic_data_count: int) -> float:
        """Calculate time efficiency score."""
        if synthetic_data_count == 0:
            return 0.0
        
        time_per_example = time_taken / synthetic_data_count
        
        # Score based on time per example (lower is better)
        if time_per_example <= 10:
            return 1.0
        elif time_per_example <= 20:
            return 0.8
        elif time_per_example <= 30:
            return 0.6
        elif time_per_example <= 60:
            return 0.4
        else:
            return 0.2
    
    def calculate_synthetic_data_quality_score(self, synthetic_data: List[Dict]) -> float:
        """Calculate quality score for synthetic data."""
        if not synthetic_data:
            return 0.0
        
        score = 0.0
        
        # Diversity score (different examples)
        unique_examples = len(set(str(example) for example in synthetic_data))
        diversity_score = unique_examples / len(synthetic_data)
        score += diversity_score * 0.4
        
        # Completeness score (examples have both input and output)
        complete_examples = 0
        for example in synthetic_data:
            if isinstance(example, dict) and len(example) >= 2:
                complete_examples += 1
        
        completeness_score = complete_examples / len(synthetic_data)
        score += completeness_score * 0.3
        
        # Length consistency score
        if len(synthetic_data) >= 3:
            lengths = [len(str(example)) for example in synthetic_data]
            length_variance = sum((l - sum(lengths)/len(lengths))**2 for l in lengths) / len(lengths)
            consistency_score = 1.0 / (1.0 + length_variance / 1000)  # Normalize
            score += consistency_score * 0.3
        
        return min(score, 1.0)
    
    def calculate_overall_quality_score(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive quality scores."""
        prompt = result.get('result', '')
        metrics = result.get('metrics', {})
        synthetic_data = result.get('synthetic_data', [])
        
        scores = {
            'clarity_score': self.calculate_prompt_clarity_score(prompt),
            'cost_efficiency_score': self.calculate_cost_efficiency_score(
                metrics.get('cost', 0), len(synthetic_data)
            ),
            'time_efficiency_score': self.calculate_time_efficiency_score(
                metrics.get('time_taken', 0), len(synthetic_data)
            ),
            'synthetic_data_quality_score': self.calculate_synthetic_data_quality_score(synthetic_data)
        }
        
        # Overall weighted score
        weights = {
            'clarity_score': 0.4,
            'cost_efficiency_score': 0.2,
            'time_efficiency_score': 0.2,
            'synthetic_data_quality_score': 0.2
        }
        
        overall_score = sum(scores[key] * weights[key] for key in scores.keys())
        scores['overall_score'] = overall_score
        
        return scores
    
    def add_result(self, result: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Add optimization result for tracking."""
        scores = self.calculate_overall_quality_score(result)
        
        entry = {
            'timestamp': time.time(),
            'session_id': result.get('session_id'),
            'original_input': result.get('raw_input', ''),
            'optimized_prompt': result.get('result', ''),
            'task_type': result.get('task_type', 'unknown'),
            'metrics': result.get('metrics', {}),
            'synthetic_data_count': len(result.get('synthetic_data', [])),
            'custom_scores': scores,
            'metadata': metadata or {}
        }
        
        self.metrics_history.append(entry)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {"error": "No metrics history available"}
        
        # Calculate averages
        avg_scores = {}
        score_keys = ['clarity_score', 'cost_efficiency_score', 'time_efficiency_score', 
                     'synthetic_data_quality_score', 'overall_score']
        
        for key in score_keys:
            values = [entry['custom_scores'][key] for entry in self.metrics_history]
            avg_scores[f'avg_{key}'] = sum(values) / len(values)
        
        # Best and worst performers
        best_entry = max(self.metrics_history, key=lambda x: x['custom_scores']['overall_score'])
        worst_entry = min(self.metrics_history, key=lambda x: x['custom_scores']['overall_score'])
        
        # Task type breakdown
        task_breakdown = {}
        for entry in self.metrics_history:
            task_type = entry['task_type']
            if task_type not in task_breakdown:
                task_breakdown[task_type] = {
                    'count': 0, 'avg_overall_score': 0, 'total_cost': 0
                }
            
            task_breakdown[task_type]['count'] += 1
            task_breakdown[task_type]['avg_overall_score'] += entry['custom_scores']['overall_score']
            task_breakdown[task_type]['total_cost'] += entry['metrics'].get('cost', 0)
        
        # Calculate averages for task breakdown
        for task_type in task_breakdown:
            count = task_breakdown[task_type]['count']
            task_breakdown[task_type]['avg_overall_score'] /= count
        
        return {
            'total_optimizations': len(self.metrics_history),
            'average_scores': avg_scores,
            'best_performer': {
                'session_id': best_entry['session_id'],
                'overall_score': best_entry['custom_scores']['overall_score'],
                'task_type': best_entry['task_type']
            },
            'worst_performer': {
                'session_id': worst_entry['session_id'],
                'overall_score': worst_entry['custom_scores']['overall_score'],
                'task_type': worst_entry['task_type']
            },
            'task_breakdown': task_breakdown
        }
    
    def export_metrics(self, filename: str = "custom_metrics.json"):
        """Export metrics to JSON file."""
        data = {
            'metrics_history': self.metrics_history,
            'performance_summary': self.get_performance_summary(),
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ Metrics exported to {filename}")

def run_optimization_with_metrics(api_key: str, input_prompt: str, task_type: str = "classification") -> Dict[str, Any]:
    """Run optimization and calculate custom metrics."""
    config = {
        "raw_input": input_prompt,
        "model_name": "gpt-3.5-turbo",
        "model_api_key": api_key,
        "model_provider": "openai",
        "backend": "simple_meta_prompt",
        "synthetic_data_size": 5,
        "task_type": task_type
    }
    
    result = process_input(**config)
    return result

def main():
    """Main function for custom metrics demonstration."""
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("   Please set your API key: export OPENAI_API_KEY='your_api_key'")
        return 1
    
    print("🚀 Promptomatix Custom Metrics Example")
    print("=" * 60)
    
    # Create custom metrics tracker
    metrics_tracker = CustomMetrics()
    
    # Test prompts for different task types
    test_cases = [
        {
            "input": "Classify customer feedback as positive, negative, or neutral",
            "task_type": "classification",
            "description": "Sentiment Classification"
        },
        {
            "input": "Summarize a long article in exactly 3 sentences",
            "task_type": "summarization",
            "description": "Text Summarization"
        },
        {
            "input": "Extract key information from customer reviews",
            "task_type": "extraction",
            "description": "Information Extraction"
        },
        {
            "input": "Generate creative marketing copy for a product",
            "task_type": "generation",
            "description": "Creative Generation"
        },
        {
            "input": "Answer questions about a given text accurately",
            "task_type": "qa",
            "description": "Question Answering"
        }
    ]
    
    print(f"🧪 Running {len(test_cases)} optimizations with custom metrics...")
    
    # Run optimizations and track metrics
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}/{len(test_cases)}: {test_case['description']}")
        print(f"Input: {test_case['input']}")
        
        try:
            result = run_optimization_with_metrics(
                api_key, 
                test_case['input'], 
                test_case['task_type']
            )
            
            # Calculate and display custom metrics
            scores = metrics_tracker.calculate_overall_quality_score(result)
            
            print(f"✅ Optimization completed!")
            print(f"Optimized Prompt: {result['result'][:100]}...")
            print(f"Custom Scores:")
            print(f"  Clarity: {scores['clarity_score']:.3f}")
            print(f"  Cost Efficiency: {scores['cost_efficiency_score']:.3f}")
            print(f"  Time Efficiency: {scores['time_efficiency_score']:.3f}")
            print(f"  Synthetic Data Quality: {scores['synthetic_data_quality_score']:.3f}")
            print(f"  Overall Score: {scores['overall_score']:.3f}")
            print(f"Cost: ${result['metrics']['cost']:.4f}")
            
            # Add to metrics tracker
            metrics_tracker.add_result(result, {
                'test_case': test_case['description'],
                'test_index': i
            })
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
    
    # Display comprehensive summary
    print(f"\n📊 Custom Metrics Summary")
    print("=" * 50)
    
    summary = metrics_tracker.get_performance_summary()
    
    print(f"Total Optimizations: {summary['total_optimizations']}")
    print(f"\nAverage Scores:")
    for key, value in summary['average_scores'].items():
        print(f"  {key}: {value:.3f}")
    
    print(f"\n🏆 Best Performer:")
    best = summary['best_performer']
    print(f"  Task Type: {best['task_type']}")
    print(f"  Overall Score: {best['overall_score']:.3f}")
    print(f"  Session ID: {best['session_id']}")
    
    print(f"\n📉 Worst Performer:")
    worst = summary['worst_performer']
    print(f"  Task Type: {worst['task_type']}")
    print(f"  Overall Score: {worst['overall_score']:.3f}")
    print(f"  Session ID: {worst['session_id']}")
    
    print(f"\n📊 Task Type Breakdown:")
    for task_type, stats in summary['task_breakdown'].items():
        print(f"  {task_type}:")
        print(f"    Count: {stats['count']}")
        print(f"    Avg Overall Score: {stats['avg_overall_score']:.3f}")
        print(f"    Total Cost: ${stats['total_cost']:.4f}")
    
    # Export metrics
    print(f"\n💾 Exporting Metrics...")
    metrics_tracker.export_metrics("custom_metrics_results.json")
    
    # Show detailed analysis
    print(f"\n🔍 Detailed Analysis:")
    print("=" * 30)
    
    # Find highest clarity score
    best_clarity = max(metrics_tracker.metrics_history, 
                      key=lambda x: x['custom_scores']['clarity_score'])
    print(f"Highest Clarity Score: {best_clarity['custom_scores']['clarity_score']:.3f}")
    print(f"Prompt: {best_clarity['optimized_prompt'][:80]}...")
    
    # Find most cost efficient
    best_cost_efficiency = max(metrics_tracker.metrics_history, 
                              key=lambda x: x['custom_scores']['cost_efficiency_score'])
    print(f"\nMost Cost Efficient: {best_cost_efficiency['custom_scores']['cost_efficiency_score']:.3f}")
    print(f"Cost: ${best_cost_efficiency['metrics']['cost']:.4f}")
    
    print(f"\n✅ Custom metrics analysis completed successfully!")
    print(f"📁 Check custom_metrics_results.json for detailed metrics")
    
    return 0

if __name__ == "__main__":
    exit(main()) 