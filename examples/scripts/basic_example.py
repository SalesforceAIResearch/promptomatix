#!/usr/bin/env python3
"""
Basic Example Script for Promptomatix

This script demonstrates the three core workflows:
1. Prompt Optimization
2. Feedback Generation  
3. Optimization with Feedback

Usage:
    python basic_example.py
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to Python path
sys.path.append('../src')

# Import Promptomatix functions
from promptomatix.main import process_input, generate_feedback, optimize_with_feedback

def main():
    """Main function demonstrating basic Promptomatix usage."""
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("   Please set your API key: export OPENAI_API_KEY='your_api_key'")
        return 1
    
    print("🚀 Promptomatix Basic Example")
    print("=" * 50)
    
    # Example 1: Prompt Optimization
    print("\n1️⃣  Prompt Optimization")
    print("-" * 30)
    
    config = {
        "raw_input": "Classify if a text is positive, negative, or neutral",
        "model_name": "gpt-3.5-turbo",
        "model_api_key": api_key,
        "model_provider": "openai",
        "backend": "simple_meta_prompt",
        "synthetic_data_size": 5,
        "task_type": "classification"
    }
    
    print(f"Input: {config['raw_input']}")
    print("Optimizing...")
    
    try:
        result = process_input(**config)
        
        print("✅ Optimization completed!")
        print(f"Optimized Prompt: {result['result']}")
        print(f"Session ID: {result['session_id']}")
        print(f"Cost: ${result['metrics']['cost']:.4f}")
        print(f"Time: {result['metrics']['time_taken']:.2f} seconds")
        
        # Example 2: Feedback Generation
        print("\n2️⃣  Feedback Generation")
        print("-" * 30)
        
        print("Generating feedback...")
        
        feedback_result = generate_feedback(
            optimized_prompt=result['result'],
            input_fields=result['input_fields'],
            output_fields=result['output_fields'],
            model_name="gpt-3.5-turbo",
            model_api_key=api_key,
            synthetic_data=result['synthetic_data'],
            session_id=result['session_id']
        )
        
        print("✅ Feedback generated!")
        print(f"Comprehensive Feedback: {feedback_result['comprehensive_feedback'][:200]}...")
        print(f"Individual Feedbacks: {len(feedback_result['individual_feedbacks'])} samples")
        
        # Example 3: Optimization with Feedback
        print("\n3️⃣  Optimization with Feedback")
        print("-" * 30)
        
        print("Optimizing with feedback...")
        
        improved_result = optimize_with_feedback(result['session_id'])
        
        print("✅ Optimization with feedback completed!")
        print(f"Original Prompt: {result['result']}")
        print(f"Improved Prompt: {improved_result['result']}")
        print(f"Additional Cost: ${improved_result['metrics']['cost']:.4f}")
        
        # Summary
        print("\n📊 Summary")
        print("-" * 30)
        print(f"Total Cost: ${result['metrics']['cost'] + improved_result['metrics']['cost']:.4f}")
        print(f"Total Time: {result['metrics']['time_taken'] + improved_result['metrics']['time_taken']:.2f} seconds")
        print(f"Session ID: {result['session_id']}")
        print("\n✅ Basic example completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 