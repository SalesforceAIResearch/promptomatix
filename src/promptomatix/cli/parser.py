"""
Command-line interface parser for the prompt optimization tool.
"""

from argparse import ArgumentParser
from typing import Dict

def parse_args() -> Dict:
    """
    Parse and validate command line arguments.
    
    Returns:
        Dict: Configuration dictionary with all parsed arguments
    """
    parser = ArgumentParser(description="Prompt Optimization Tool")
    
    # Basic configuration
    input_source = parser.add_mutually_exclusive_group(required=True)
    input_source.add_argument("--raw_input", type=str, 
                            help="Initial human input for prompt optimization")
    input_source.add_argument("--huggingface_dataset_name", type=str, 
                            help="Name of the HuggingFace dataset to use")
    # optional arguments
    additional_args = parser.add_argument_group('Additional Arguments')
    additional_args.add_argument("--task", type=str, help="original task")
    additional_args.add_argument("--few_shot_examples", type=str, help="few shot examples")
    additional_args.add_argument("--task_context", type=str, help="context")
    additional_args.add_argument("--question", type=str, help="question")
    
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--model_name", type=str, help="Model name")
    model_group.add_argument("--model_api_key", type=str, help="Model API key")
    model_group.add_argument("--model_api_base", type=str, help="Model API base")
    model_group.add_argument("--model_provider", type=str, help="Model provider")
    model_group.add_argument("--temperature", type=float, 
                           help="Model temperature (default: 0.7)")
    model_group.add_argument("--max_tokens", type=int, 
                           help="Maximum tokens for model output (default: 4000)")
    model_group.add_argument("--dspy_module", type=str, help="DSPy module")
    model_group.add_argument("--backend", type=str, 
                           help="Optimization backend ('dspy' or 'simple_meta_prompt', default: 'simple_meta_prompt')")
    
    # Data configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--input_fields', nargs='+', help='Input fields')
    data_group.add_argument('--output_fields', nargs='+', help='Output fields')
    data_group.add_argument("--train_data", type=str, help="Train data")
    data_group.add_argument("--valid_data", type=str, help="Validation data")
    data_group.add_argument("--valid_data_full", type=str, help="Full validation data")
    data_group.add_argument("--synthetic_data_size", type=int, help="Synthetic data size")
    data_group.add_argument("--train_data_size", type=int, help="Train data size")
    data_group.add_argument("--valid_data_size", type=int, help="Validation data size")
    data_group.add_argument("--train_ratio", type=float, help="Train ratio")
    data_group.add_argument('--load_data_local', action="store_true", 
                           help="Load data from local file")
    data_group.add_argument('--local_train_data_path', type=str, 
                           help="Path to local train data")
    data_group.add_argument('--local_test_data_path', type=str, 
                           help="Path to local test data")
    data_group.add_argument("--sample_data", type=str, help="Sample data")
    
    # Task configuration
    task_group = parser.add_argument_group('Task Configuration')
    task_group.add_argument("--task_type", type=str, help="Task type")
    task_group.add_argument("--task_description", type=str, help="Task description")
    task_group.add_argument("--output_format", type=str, help="Output format")
    task_group.add_argument("--style_guide", type=str, help="Style guide")
    task_group.add_argument("--constraints", type=str, help="Constraints")
    task_group.add_argument("--context", type=str, help="Context")
    task_group.add_argument("--tools", type=str, help="Tools")
    
    # Training configuration
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument("--metrics", nargs='+', help="Metrics")
    training_group.add_argument("--trainer", type=str, help="Trainer")
    training_group.add_argument("--search_type", type=str, help="Search type")
    
    # Feedback management
    feedback_group = parser.add_argument_group('Feedback Management')
    feedback_group.add_argument("--list_feedbacks", action="store_true", 
                              help="List all feedbacks")
    feedback_group.add_argument("--analyze_feedbacks", action="store_true", 
                              help="Analyze feedbacks")
    feedback_group.add_argument("--export_feedbacks", type=str, 
                              help="Export feedbacks to file")
    feedback_group.add_argument("--prompt_id", type=str, help="Filter by prompt ID")
    
    args = parser.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None} 

def main():
    """Main CLI entry point."""
    try:
        args = parse_args()
        
        # Import here to avoid circular imports
        from ..main import process_input  # Fixed import path
        
        if args.get('raw_input'):
            print(f" Optimizing prompt: {args['raw_input']}")
            result = process_input(human_input=args['raw_input'])
            print(f"\n✅ Optimized prompt:\n{result['result']}")
            print(f"\n📝 Session ID: {result['session_id']}")
        else:
            print("❌ Please provide input using --raw_input")
            print("Example: promtomatic --raw_input 'Classify text sentiment'")
            return 1
        
        return 0
    except KeyboardInterrupt:
        print("\n Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 