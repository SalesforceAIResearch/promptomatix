#!/usr/bin/env python3
"""
Batch Processing Script for Promptomatix

This script demonstrates how to process multiple prompts efficiently:
- Sequential and parallel processing
- Cost tracking and optimization
- Error handling and retry logic
- Performance monitoring

Usage:
    python batch_processing.py
"""

import os
import sys
import json
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add the src directory to Python path
sys.path.append('../src')

# Import Promptomatix functions
from promptomatix.main import process_input

class BatchProcessor:
    """Batch processor for multiple prompt optimizations."""
    
    def __init__(self, api_key: str, max_workers: int = 3):
        self.api_key = api_key
        self.max_workers = max_workers
        self.base_config = {
            "model_name": "gpt-3.5-turbo",
            "model_api_key": api_key,
            "model_provider": "openai",
            "backend": "simple_meta_prompt",
            "synthetic_data_size": 5
        }
        self.results = []
        self.errors = []
        self.cost_tracker = []
        
    def process_sequential(self, prompts: List[Dict[str, Any]]) -> List[Dict]:
        """Process prompts sequentially."""
        print(f"🔄 Processing {len(prompts)} prompts sequentially...")
        
        results = []
        for i, prompt_data in enumerate(prompts, 1):
            print(f"\n📝 Processing {i}/{len(prompts)}: {prompt_data['input'][:50]}...")
            
            config = self.base_config.copy()
            config.update({
                'raw_input': prompt_data['input'],
                'task_type': prompt_data.get('task_type', 'classification')
            })
            
            try:
                start_time = time.time()
                result = process_input(**config)
                end_time = time.time()
                
                # Add metadata
                result['original_input'] = prompt_data['input']
                result['task_type'] = prompt_data.get('task_type', 'classification')
                result['processing_time'] = end_time - start_time
                
                results.append(result)
                self.cost_tracker.append(result['metrics']['cost'])
                
                print(f"✅ Completed - Cost: ${result['metrics']['cost']:.4f}, Time: {result['processing_time']:.2f}s")
                
            except Exception as e:
                error_info = {
                    'input': prompt_data['input'],
                    'task_type': prompt_data.get('task_type', 'classification'),
                    'error': str(e),
                    'index': i
                }
                self.errors.append(error_info)
                print(f"❌ Failed: {str(e)}")
        
        self.results = results
        return results
    
    def process_parallel(self, prompts: List[Dict[str, Any]]) -> List[Dict]:
        """Process prompts in parallel (limited by API rate limits)."""
        print(f"🔄 Processing {len(prompts)} prompts in parallel (max {self.max_workers} workers)...")
        
        def process_single(prompt_data):
            config = self.base_config.copy()
            config.update({
                'raw_input': prompt_data['input'],
                'task_type': prompt_data.get('task_type', 'classification')
            })
            
            try:
                start_time = time.time()
                result = process_input(**config)
                end_time = time.time()
                
                # Add metadata
                result['original_input'] = prompt_data['input']
                result['task_type'] = prompt_data.get('task_type', 'classification')
                result['processing_time'] = end_time - start_time
                
                return result
                
            except Exception as e:
                return {
                    'input': prompt_data['input'],
                    'task_type': prompt_data.get('task_type', 'classification'),
                    'error': str(e)
                }
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_single, prompt_data) for prompt_data in prompts]
            
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                
                if 'error' in result:
                    self.errors.append(result)
                    print(f"❌ Failed {i}/{len(prompts)}: {result['error']}")
                else:
                    results.append(result)
                    self.cost_tracker.append(result['metrics']['cost'])
                    print(f"✅ Completed {i}/{len(prompts)} - Cost: ${result['metrics']['cost']:.4f}")
        
        self.results = results
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.results:
            return {"error": "No results available"}
        
        total_cost = sum(self.cost_tracker)
        total_time = sum(r['processing_time'] for r in self.results)
        success_rate = len(self.results) / (len(self.results) + len(self.errors))
        
        # Task type breakdown
        task_breakdown = {}
        for result in self.results:
            task_type = result['task_type']
            if task_type not in task_breakdown:
                task_breakdown[task_type] = {'count': 0, 'total_cost': 0}
            task_breakdown[task_type]['count'] += 1
            task_breakdown[task_type]['total_cost'] += result['metrics']['cost']
        
        return {
            'total_prompts': len(self.results) + len(self.errors),
            'successful': len(self.results),
            'failed': len(self.errors),
            'success_rate': success_rate,
            'total_cost': total_cost,
            'total_time': total_time,
            'average_cost_per_prompt': total_cost / len(self.results) if self.results else 0,
            'task_breakdown': task_breakdown
        }
    
    def export_results(self, filename: str = "batch_results.json"):
        """Export results to JSON file."""
        data = {
            'results': self.results,
            'errors': self.errors,
            'summary': self.get_summary(),
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ Results exported to {filename}")
    
    def export_csv(self, filename: str = "batch_results.csv"):
        """Export results to CSV file."""
        if not self.results:
            print("❌ No results to export")
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Input', 'Task Type', 'Optimized Prompt', 'Cost', 'Time', 
                'Session ID', 'Success'
            ])
            
            # Write results
            for result in self.results:
                writer.writerow([
                    result['original_input'][:100],  # Truncate long inputs
                    result['task_type'],
                    result['result'][:100],  # Truncate long prompts
                    result['metrics']['cost'],
                    result['processing_time'],
                    result['session_id'],
                    'Yes'
                ])
            
            # Write errors
            for error in self.errors:
                writer.writerow([
                    error['input'][:100],
                    error.get('task_type', 'unknown'),
                    '',
                    0,
                    0,
                    '',
                    'No'
                ])
        
        print(f"✅ Results exported to {filename}")

def load_sample_prompts() -> List[Dict[str, Any]]:
    """Load sample prompts for demonstration."""
    return [
        {
            "input": "Classify customer feedback as positive, negative, or neutral",
            "task_type": "classification"
        },
        {
            "input": "Summarize a long article in 3 sentences",
            "task_type": "summarization"
        },
        {
            "input": "Extract key information from customer reviews",
            "task_type": "extraction"
        },
        {
            "input": "Generate creative marketing copy for a product",
            "task_type": "generation"
        },
        {
            "input": "Answer questions about a given text accurately",
            "task_type": "qa"
        },
        {
            "input": "Translate English text to Spanish",
            "task_type": "translation"
        },
        {
            "input": "Identify sentiment in social media posts",
            "task_type": "classification"
        },
        {
            "input": "Create professional email responses",
            "task_type": "generation"
        }
    ]

def main():
    """Main function for batch processing demonstration."""
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("   Please set your API key: export OPENAI_API_KEY='your_api_key'")
        return 1
    
    print("🚀 Promptomatix Batch Processing Example")
    print("=" * 60)
    
    # Load sample prompts
    prompts = load_sample_prompts()
    print(f"📝 Loaded {len(prompts)} sample prompts")
    
    # Create batch processor
    processor = BatchProcessor(api_key, max_workers=2)  # Conservative parallel processing
    
    # Process prompts sequentially
    print("\n1️⃣  Sequential Processing")
    print("-" * 40)
    sequential_results = processor.process_sequential(prompts)
    
    # Get summary
    summary = processor.get_summary()
    print(f"\n📊 Sequential Processing Summary:")
    print(f"Total Prompts: {summary['total_prompts']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Total Cost: ${summary['total_cost']:.4f}")
    print(f"Total Time: {summary['total_time']:.2f} seconds")
    print(f"Average Cost per Prompt: ${summary['average_cost_per_prompt']:.4f}")
    
    # Task breakdown
    print(f"\n📊 Task Type Breakdown:")
    for task_type, stats in summary['task_breakdown'].items():
        avg_cost = stats['total_cost'] / stats['count']
        print(f"  {task_type}: {stats['count']} prompts, ${avg_cost:.4f} avg cost")
    
    # Export results
    print(f"\n💾 Exporting Results...")
    processor.export_results("batch_processing_results.json")
    processor.export_csv("batch_processing_results.csv")
    
    # Show some example results
    print(f"\n📋 Example Results:")
    for i, result in enumerate(sequential_results[:3], 1):
        print(f"\n{i}. {result['original_input'][:50]}...")
        print(f"   Optimized: {result['result'][:80]}...")
        print(f"   Cost: ${result['metrics']['cost']:.4f}")
    
    print(f"\n✅ Batch processing completed successfully!")
    print(f"📁 Check batch_processing_results.json and batch_processing_results.csv for detailed results")
    
    return 0

if __name__ == "__main__":
    exit(main()) 