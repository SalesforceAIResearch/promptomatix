import warnings
import os
import sys
import contextlib
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from typing import Callable, Dict, Any, Union, List
import numpy as np
from collections import Counter
import re
import math
from langdetect import detect  # You'll need to: pip install langdetect
from ..core.config import LambdaPenalty  # Fix the import path

# Set environment variables to suppress warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Comprehensive warning suppression
warnings.filterwarnings("ignore")

# Suppress all warnings from specific modules
warnings.filterwarnings("ignore", module="transformers")
warnings.filterwarnings("ignore", module="bert_score")

# Create a context manager for suppressing stderr during BERTScore calls
@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Now import BERTScore after setting up suppression
from bert_score import score as bert_score_original

# Create a silent wrapper for BERTScore
def bert_score_silent(*args, **kwargs):
    """BERTScore wrapper that suppresses all stderr output."""
    with suppress_stderr():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bert_score_original(*args, **kwargs)

# Replace the original import
bert_score_metric = bert_score_silent

class MetricsManager:
    _output_fields = None  # Class-level storage for output fields

    @staticmethod
    def configure(output_fields: List[str]) -> None:
        """Configure the MetricsManager with output fields.
        
        Args:
            output_fields (List[str]): List of output field names
        """
        MetricsManager._output_fields = output_fields

    @staticmethod
    def _get_output_value(item: Any) -> str:
        """Helper method to get the output value from an item using the first output field"""
        if not MetricsManager._output_fields:
            return ''
            # raise ValueError("MetricsManager not configured with output_fields")
        
        # For each output field, check if it is present in the given dict i.e. `item`
        # Collect all such existing fields and return a concatenated string
        fields = [field for field in MetricsManager._output_fields if field in item]
        return ' '.join([str(getattr(item, field, '')).lower().strip() for field in fields])

    @staticmethod
    def get_metrics_for_task(task_type: str) -> Callable:
        """Returns appropriate evaluation metric function based on task type.
        
        Args:
            task_type (str): Type of the task
            
        Returns:
            Callable: A function that computes the metric for the given task type
        """
        task_type = task_type.lower()
        
        metrics_map = {
            'qa': MetricsManager._qa_metrics,
            'classification': MetricsManager._classification_metrics,
            'generation': MetricsManager._generation_metrics,
            'summarization': MetricsManager._summarization_metrics,
            'translation': MetricsManager._translation_metrics,
            
            'multi_label_classification': MetricsManager._multi_label_classification_metrics,
            
            'information_extraction': MetricsManager._information_extraction_metrics,
            
            'paraphrasing': MetricsManager._paraphrasing_metrics,
            
            'conversation': MetricsManager._conversation_metrics,
            'negotiation': MetricsManager._negotiation_metrics,
            
            'code_generation': MetricsManager._code_generation_metrics,
            'code_explanation': MetricsManager._code_explanation_metrics,
            'code_completion': MetricsManager._code_completion_metrics,
            'code_debugging': MetricsManager._code_debugging_metrics,
            
            'planning': MetricsManager._planning_metrics,
            'tool_use': MetricsManager._tool_use_metrics,
            'decision_making': MetricsManager._decision_making_metrics,
            'process_automation': MetricsManager._process_automation_metrics,
            
            'recommendation': MetricsManager._recommendation_metrics,
            'data_analysis': MetricsManager._data_analysis_metrics
        }
        
        return metrics_map.get(task_type, MetricsManager._qa_metrics)
    
    @staticmethod
    def get_final_eval_metrics(task_type: str) -> Callable:
        """Returns appropriate final evaluation metric function based on task type.
        
        Args:
            task_type (str): Type of the task
        """
        task_type = task_type.lower()

        metrics_map = {
            'qa': MetricsManager._qa_metrics_final_eval,
            'classification': MetricsManager._classification_metrics_final_eval,
            'generation': MetricsManager._generation_metrics_final_eval,
            'summarization': MetricsManager._summarization_metrics_final_eval,
            'translation': MetricsManager._translation_metrics_final_eval,
            
            'multi_label_classification': MetricsManager._multi_label_classification_metrics_final_eval,
            
            'information_extraction': MetricsManager._information_extraction_metrics_final_eval,
            
            'paraphrasing': MetricsManager._paraphrasing_metrics_final_eval,
            
            'conversation': MetricsManager._conversation_metrics_final_eval,
            'negotiation': MetricsManager._negotiation_metrics_final_eval,
            
            'code_generation': MetricsManager._code_generation_metrics_final_eval,
            'code_explanation': MetricsManager._code_explanation_metrics_final_eval,
            'code_completion': MetricsManager._code_completion_metrics_final_eval,
            'code_debugging': MetricsManager._code_debugging_metrics_final_eval,
            
            'planning': MetricsManager._planning_metrics_final_eval,
            'tool_use': MetricsManager._tool_use_metrics_final_eval,
            'decision_making': MetricsManager._decision_making_metrics_final_eval,
            'process_automation': MetricsManager._process_automation_metrics_final_eval,
            
            'recommendation': MetricsManager._recommendation_metrics_final_eval,
            'data_analysis': MetricsManager._data_analysis_metrics_final_eval
        }

        return metrics_map.get(task_type, MetricsManager._qa_metrics_final_eval)

    @staticmethod
    def _qa_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """
        Evaluates QA predictions using BERT Score.
        
        Args:
            example: The example containing the gold answer
            pred: The prediction containing the model's answer
        """
        try:
            # Extract answers using configured output fields
            pred_answer = MetricsManager._get_output_value(pred)
            gold_answer = MetricsManager._get_output_value(example)

            if '####' in gold_answer:
                gold_answer = gold_answer.split('####')[-1].strip()
            if '####' in pred_answer:
                pred_answer = pred_answer.split('####')[-1].strip()

            em_score = float(pred_answer.lower().strip() == gold_answer.lower().strip())
            
            # # BERT Score expects lists of strings
            P, R, F1 = bert_score_metric([pred_answer], [gold_answer], lang="en", rescale_with_baseline=False)
            
            # # Convert tensor to float and take mean
            score = float(F1.mean())


            # exatact prompt length
            prompt_length = len(instructions.split()) if instructions else 0
            lambda_penalty = LambdaPenalty.get_value()

            # exponential decay penalty
            length_penalty = math.exp(-lambda_penalty * prompt_length)  # Exponential decay


            return ((em_score + score) / 2) * length_penalty
            
        except Exception as e:
            print(f"--------------------------------")
            print(f"pred_answer: {pred_answer}")
            print(f"--------------------------------")
            print(f"gold_answer: {gold_answer}")
            print(f"--------------------------------")
            # Return 0 score for failed comparisons
            return 0.0

    @staticmethod
    def _classification_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for Classification tasks.
        Returns accuracy (1 for correct, 0 for incorrect).
        
        Returns:
            float: Score between 0 and 1
        """
        gold_label = MetricsManager._get_output_value(example)
        pred_label = MetricsManager._get_output_value(pred)
        em_score = float(gold_label.lower().strip() == pred_label.lower().strip())

        # exatact prompt length
        prompt_length = len(instructions.split()) if instructions else 0
        lambda_penalty = LambdaPenalty.get_value()

        # exponential decay penalty
        length_penalty = math.exp(-lambda_penalty * prompt_length)  # Exponential decay

        return em_score * length_penalty
    
    @staticmethod
    def _generation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for Text Generation tasks.
        Returns average of BLEU and ROUGE scores.
        
        Returns:
            float: Score between 0 and 1
        """
        gold_text = MetricsManager._get_output_value(example)
        pred_text = MetricsManager._get_output_value(pred)

        fluent_ref = "This is a well-written, grammatically correct, and flowing text."
        creative_ref = "This is a unique, novel, and imaginative piece of writing with original ideas."
        
        _, _, fluency = bert_score_metric([pred_text], [fluent_ref], lang='en', rescale_with_baseline=False)
        _, _, creativity = bert_score_metric([pred_text], [creative_ref], lang='en', rescale_with_baseline=False)
        _, _, similarity = bert_score_metric([pred_text], [gold_text], lang='en', rescale_with_baseline=False)
   
        combined_score = (fluency.mean().item() + creativity.mean().item() + similarity.mean().item()) / 3

        # exatact prompt length
        prompt_length = len(instructions.split()) if instructions else 0
        lambda_penalty = LambdaPenalty.get_value()

        # exponential decay penalty
        length_penalty = math.exp(-lambda_penalty * prompt_length)  # Exponential decay

        return combined_score * length_penalty

    @staticmethod
    def _summarization_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for Summarization tasks.
        Returns average of ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
        
        Returns:
            float: Score between 0 and 1
        """
        gold_summary = MetricsManager._get_output_value(example)
        pred_summary = MetricsManager._get_output_value(pred)

        if isinstance(pred_summary, list):
            pred_summary = pred_summary
        else:
            pred_summary = [str(pred_summary)]
        
        if isinstance(gold_summary, list):
            gold_summary = gold_summary
        else:
            gold_summary = [str(gold_summary)]

        P, R, F1 = bert_score_metric(pred_summary, gold_summary, lang="en", rescale_with_baseline=False)
        score = F1.mean().item()

        # exatact prompt length
        prompt_length = len(instructions.split()) if instructions else 0
        lambda_penalty = LambdaPenalty.get_value()

        # exponential decay penalty
        length_penalty = math.exp(-lambda_penalty * prompt_length)  # Exponential decay

        return score * length_penalty

    @staticmethod
    def _translation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """
        Evaluates translation predictions using BERT Score with automatic language detection.
        """
        try:
            # Extract translations
            pred_translation = MetricsManager._get_output_value(pred)
            gold_translation = MetricsManager._get_output_value(example)
            
            # Try to detect language from the prediction
            try:
                # Detect language from prediction or gold (prediction might be more reliable)
                detected_lang = detect(pred_translation)
                # Convert ISO 639-1 codes to BERT Score supported languages
                lang_mapping = {
                    'de': 'de',  # German
                    'fr': 'fr',  # French
                    'es': 'es',  # Spanish
                    'it': 'it',  # Italian
                    'nl': 'nl',  # Dutch
                    'zh': 'zh',  # Chinese
                    'ja': 'ja',  # Japanese
                    'ko': 'ko',  # Korean
                    'ru': 'ru',  # Russian
                    # Add more mappings as needed
                }
                bert_lang = lang_mapping.get(detected_lang, 'en')  # Default to English if language not in mapping
            except:
                # Fallback to English if detection fails
                bert_lang = 'en'
            
            # Calculate BERT Score with detected language
            P, R, F1 = bert_score_metric([pred_translation], [gold_translation], lang=bert_lang, model_type='distilbert-base-multilingual-cased')
            
            # Convert tensor to float and take mean
            score = float(F1.mean())

            # exatact prompt length
            prompt_length = len(instructions.split()) if instructions else 0
            lambda_penalty = LambdaPenalty.get_value()

            # exponential decay penalty
            length_penalty = math.exp(-lambda_penalty * prompt_length)  # Exponential decay

            return score * length_penalty
                            
        except Exception as e:
            print(f"Error in translation metrics: {str(e)}")
            print(f"Prediction: {pred_translation}")
            print(f"Gold: {gold_translation}")
            # Return 0 score for failed comparisons
            return 0.0

    @staticmethod
    def _default_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Default metrics when task type is unknown.
        Returns exact match score.
        
        Returns:
            float: Score between 0 and 1
        """
        

        gold_output = MetricsManager._get_output_value(example)
        pred_output = MetricsManager._get_output_value(pred)
        exact_match = float(gold_output.lower().strip() == pred_output.lower().strip())

        # exatact prompt length
        prompt_length = len(instructions.split()) if instructions else 0
        lambda_penalty = LambdaPenalty.get_value()

        # exponential decay penalty
        length_penalty = math.exp(-lambda_penalty * prompt_length)  # Exponential decay

        return exact_match * length_penalty
    
    @staticmethod
    def _final_eval_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics."""
        gold_summary = MetricsManager._get_output_value(example)
        pred_summary = MetricsManager._get_output_value(pred)

        if isinstance(pred_summary, list):
            pred_summary = pred_summary
        else:
            pred_summary = [str(pred_summary)]
        
        if isinstance(gold_summary, list):
            gold_summary = gold_summary
        else:
            gold_summary = [str(gold_summary)]

        P, R, F1 = bert_score_metric(pred_summary, gold_summary, lang="en", rescale_with_baseline=False)
        score = F1.mean().item()

        return score

    @staticmethod
    def get_detailed_metrics(task_type: str, example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> Dict[str, float]:
        """Returns detailed breakdown of all metrics for analysis purposes."""
        task_type = task_type.lower()
        
        if task_type == 'qa':
            gold_answer = MetricsManager._get_output_value(example)
            pred_answer = MetricsManager._get_output_value(pred)
            
            exact_match = float(gold_answer == pred_answer)
            pred_tokens = set(pred_answer.split())
            gold_tokens = set(gold_answer.split())
            
            if not pred_tokens and not gold_tokens:
                f1 = 1.0
            elif not pred_tokens or not gold_tokens:
                f1 = 0.0
            else:
                common = pred_tokens.intersection(gold_tokens)
                precision = len(common) / len(pred_tokens)
                recall = len(common) / len(gold_tokens)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'exact_match': exact_match,
                'f1': f1,
                'combined_score': (exact_match + f1) / 2
            }
            
        elif task_type in ['generation', 'summarization']:
            gold_text = MetricsManager._get_output_value(example)
            pred_text = MetricsManager._get_output_value(pred)
            
            rouge = Rouge()
            scores = rouge.get_scores(pred_text, gold_text)[0]
            
            return {
                'rouge-1': scores['rouge-1']['f'],
                'rouge-2': scores['rouge-2']['f'],
                'rouge-l': scores['rouge-l']['f'],
                'rouge_avg': np.mean([scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']])
            }
            
        else:
            gold_output = MetricsManager._get_output_value(example)
            pred_output = MetricsManager._get_output_value(pred)
            return {
                'accuracy': float(gold_output == pred_output)
            }       
        
    @staticmethod
    def _qa_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """
        Evaluates QA predictions using BERT Score.
        
        Args:
            example: The example containing the gold answer
            pred: The prediction containing the model's answer
        """
        try:
            # Extract answers using configured output fields
            pred_answer = MetricsManager._get_output_value(pred)
            gold_answer = MetricsManager._get_output_value(example)

            if '####' in gold_answer:
                gold_answer = gold_answer.split('####')[-1].strip()
            if '####' in pred_answer:
                pred_answer = pred_answer.split('####')[-1].strip()

            em_score = float(pred_answer.lower().strip() == gold_answer.lower().strip())
            
            # # BERT Score expects lists of strings
            P, R, F1 = bert_score_metric([pred_answer], [gold_answer], lang="en", rescale_with_baseline=False)
            
            # # Convert tensor to float and take mean
            score = float(F1.mean())


            return ((em_score + score) / 2)
            
        except Exception as e:
            print(f"--------------------------------")
            print(f"pred_answer: {pred_answer}")
            print(f"--------------------------------")
            print(f"gold_answer: {gold_answer}")
            print(f"--------------------------------")
            # Return 0 score for failed comparisons
            return 0.0

    @staticmethod
    def _classification_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for Classification tasks.
        Returns accuracy (1 for correct, 0 for incorrect).
        
        Returns:
            float: Score between 0 and 1
        """
        gold_label = MetricsManager._get_output_value(example)
        pred_label = MetricsManager._get_output_value(pred)
        em_score = float(gold_label.lower().strip() == pred_label.lower().strip())

        return em_score 
    
    @staticmethod
    def _generation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for Text Generation tasks.
        Returns average of BLEU and ROUGE scores.
        
        Returns:
            float: Score between 0 and 1
        """
        gold_text = MetricsManager._get_output_value(example)
        pred_text = MetricsManager._get_output_value(pred)

        fluent_ref = "This is a well-written, grammatically correct, and flowing text."
        creative_ref = "This is a unique, novel, and imaginative piece of writing with original ideas."
        
        _, _, fluency = bert_score_metric([pred_text], [fluent_ref], lang='en', rescale_with_baseline=False)
        _, _, creativity = bert_score_metric([pred_text], [creative_ref], lang='en', rescale_with_baseline=False)
        _, _, similarity = bert_score_metric([pred_text], [gold_text], lang='en', rescale_with_baseline=False)
   
        combined_score = (fluency.mean().item() + creativity.mean().item() + similarity.mean().item()) / 3

        return combined_score 

    @staticmethod
    def _summarization_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for Summarization tasks.
        Returns average of ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
        
        Returns:
            float: Score between 0 and 1
        """
        gold_summary = MetricsManager._get_output_value(example)
        pred_summary = MetricsManager._get_output_value(pred)

        if isinstance(pred_summary, list):
            pred_summary = pred_summary
        else:
            pred_summary = [str(pred_summary)]
        
        if isinstance(gold_summary, list):
            gold_summary = gold_summary
        else:
            gold_summary = [str(gold_summary)]

        P, R, F1 = bert_score_metric(pred_summary, gold_summary, lang="en", rescale_with_baseline=False)
        score = F1.mean().item()

        return score 

    @staticmethod
    def _translation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """
        Evaluates translation predictions using BERT Score with automatic language detection.
        """
        try:
            # Extract translations
            pred_translation = MetricsManager._get_output_value(pred)
            gold_translation = MetricsManager._get_output_value(example)
            
            # Try to detect language from the prediction
            try:
                # Detect language from prediction or gold (prediction might be more reliable)
                detected_lang = detect(pred_translation)
                # Convert ISO 639-1 codes to BERT Score supported languages
                lang_mapping = {
                    'de': 'de',  # German
                    'fr': 'fr',  # French
                    'es': 'es',  # Spanish
                    'it': 'it',  # Italian
                    'nl': 'nl',  # Dutch
                    'zh': 'zh',  # Chinese
                    'ja': 'ja',  # Japanese
                    'ko': 'ko',  # Korean
                    'ru': 'ru',  # Russian
                    # Add more mappings as needed
                }
                bert_lang = lang_mapping.get(detected_lang, 'en')  # Default to English if language not in mapping
            except:
                # Fallback to English if detection fails
                bert_lang = 'en'
            
            # Calculate BERT Score with detected language
            P, R, F1 = bert_score_metric([pred_translation], [gold_translation], lang=bert_lang, model_type='distilbert-base-multilingual-cased')
            
            # Convert tensor to float and take mean
            score = float(F1.mean())

            return score 
                            
        except Exception as e:
            print(f"Error in translation metrics: {str(e)}")
            print(f"Prediction: {pred_translation}")
            print(f"Gold: {gold_translation}")
            # Return 0 score for failed comparisons
            return 0.0

    @staticmethod
    def _default_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Default metrics when task type is unknown.
        Returns exact match score.
        
        Returns:
            float: Score between 0 and 1
        """
        

        gold_output = MetricsManager._get_output_value(example)
        pred_output = MetricsManager._get_output_value(pred)
        exact_match = float(gold_output.lower().strip() == pred_output.lower().strip())

        return exact_match 
    
    @staticmethod
    def _multi_label_classification_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for multi-label classification tasks.
        Uses label-based F1 score and Hamming distance.
        
        Returns:
            float: Score between 0 and 1
        """
        try:

            gold_labels = set(MetricsManager._get_output_value(example).split(','))
            pred_labels = set(MetricsManager._get_output_value(pred).split(','))

            gold_labels = set([label.replace('[', '').replace(']', '').replace('"', '').replace("'", '').strip() for label in gold_labels])
            pred_labels = set([label.replace('[', '').replace(']', '').replace('"', '').replace("'", '').strip() for label in pred_labels])
            
            # Calculate F1 score
            true_positives = len(gold_labels.intersection(pred_labels))
            precision = true_positives / len(pred_labels) if pred_labels else 0
            recall = true_positives / len(gold_labels) if gold_labels else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate Hamming similarity (complement of Hamming distance)
            all_labels = gold_labels.union(pred_labels)
            correct_labels = sum(1 for label in all_labels if (label in gold_labels) == (label in pred_labels))
            hamming_similarity = correct_labels / len(all_labels) if all_labels else 1
            
            # Combine metrics
            score = (f1 + hamming_similarity) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in multi-label classification metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _information_extraction_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for information extraction tasks.
        Uses F1 score for extracted entities/relations and structure accuracy.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_info = MetricsManager._get_output_value(example)
            pred_info = MetricsManager._get_output_value(pred)
            
            # Convert to sets of key-value pairs for comparison
            gold_items = set(re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', gold_info))
            pred_items = set(re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', pred_info))
            
            # Calculate F1 score for extracted items
            true_positives = len(gold_items.intersection(pred_items))
            precision = true_positives / len(pred_items) if pred_items else 0
            recall = true_positives / len(gold_items) if gold_items else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate structure similarity using BERT Score
            P, R, F1 = bert_score_metric([pred_info], [gold_info], lang="en", rescale_with_baseline=False)
            structure_score = float(F1.mean())
            
            # Combine metrics
            score = (f1 + structure_score) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in information extraction metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _paraphrasing_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for paraphrasing tasks.
        Combines semantic similarity with diversity from original text.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_text = MetricsManager._get_output_value(example)
            pred_text = MetricsManager._get_output_value(pred)
            
            # Calculate semantic similarity using BERT Score
            P, R, F1 = bert_score_metric([pred_text], [gold_text], lang="en", rescale_with_baseline=False)
            semantic_similarity = float(F1.mean())
            
            # Calculate lexical diversity (complement of word overlap)
            gold_words = set(gold_text.lower().split())
            pred_words = set(pred_text.lower().split())
            word_overlap = len(gold_words.intersection(pred_words)) / len(gold_words.union(pred_words))
            lexical_diversity = 1 - word_overlap
            
            # Combine metrics (weight semantic similarity more heavily)
            score = (0.7 * semantic_similarity + 0.3 * lexical_diversity)
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in paraphrasing metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _conversation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for conversation tasks.
        Evaluates response relevance, coherence, and context preservation.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_response = MetricsManager._get_output_value(example)
            pred_response = MetricsManager._get_output_value(pred)
            
            # Calculate response similarity using BERT Score
            P, R, F1 = bert_score_metric([pred_response], [gold_response], lang="en", rescale_with_baseline=False)
            response_similarity = float(F1.mean())
            
            # Evaluate coherence using reference to well-formed response
            coherent_ref = "This is a coherent, contextually appropriate, and well-structured response."
            _, _, coherence = bert_score_metric([pred_response], [coherent_ref], lang="en", rescale_with_baseline=False)
            coherence_score = float(coherence.mean())
            
            # Combine metrics
            score = (response_similarity + coherence_score) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in conversation metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _negotiation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for negotiation tasks.
        Evaluates goal achievement, fairness, and strategy effectiveness.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_response = MetricsManager._get_output_value(example)
            pred_response = MetricsManager._get_output_value(pred)
            
            # Calculate response appropriateness using BERT Score
            P, R, F1 = bert_score_metric([pred_response], [gold_response], lang="en", rescale_with_baseline=False)
            response_score = float(F1.mean())
            
            # Evaluate negotiation effectiveness using reference to ideal negotiation
            effective_ref = "This response demonstrates effective negotiation strategy, maintains fairness, and works toward mutual agreement."
            _, _, effectiveness = bert_score_metric([pred_response], [effective_ref], lang="en", rescale_with_baseline=False)
            effectiveness_score = float(effectiveness.mean())
            
            # Combine metrics
            score = (response_score + effectiveness_score) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in negotiation metrics: {str(e)}")
            return 0.0
    
    @staticmethod
    def _code_generation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for code generation tasks.
        Evaluates code correctness, style, and efficiency.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_code = MetricsManager._get_output_value(example)
            pred_code = MetricsManager._get_output_value(pred)
            
            # Calculate code similarity using BERT Score
            P, R, F1 = bert_score_metric([pred_code], [gold_code], lang="en", rescale_with_baseline=False)
            code_similarity = float(F1.mean())
            
            # Evaluate code quality using reference to well-structured code
            quality_ref = "This code follows best practices, is well-documented, and efficiently implements the required functionality."
            _, _, quality = bert_score_metric([pred_code], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            # Combine metrics
            score = (code_similarity + quality_score) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in code generation metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _code_explanation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for code explanation tasks.
        Evaluates explanation clarity, completeness, and accuracy.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_explanation = MetricsManager._get_output_value(example)
            pred_explanation = MetricsManager._get_output_value(pred)
            
            # Calculate explanation similarity using BERT Score
            P, R, F1 = bert_score_metric([pred_explanation], [gold_explanation], lang="en", rescale_with_baseline=False)
            explanation_similarity = float(F1.mean())
            
            # Evaluate explanation quality using reference
            quality_ref = "This explanation is clear, complete, and accurately describes the code's functionality and implementation details."
            _, _, quality = bert_score_metric([pred_explanation], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            # Combine metrics
            score = (explanation_similarity + quality_score) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in code explanation metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _code_completion_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for code completion tasks.
        Evaluates completion correctness and consistency with context.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_completion = MetricsManager._get_output_value(example)
            pred_completion = MetricsManager._get_output_value(pred)
            
            # Calculate completion similarity using BERT Score
            P, R, F1 = bert_score_metric([pred_completion], [gold_completion], lang="en", rescale_with_baseline=False)
            completion_similarity = float(F1.mean())
            
            # Evaluate completion quality using reference
            quality_ref = "This code completion is syntactically correct and consistent with the surrounding context."
            _, _, quality = bert_score_metric([pred_completion], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            # Combine metrics
            score = (completion_similarity + quality_score) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in code completion metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _code_debugging_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for code debugging tasks.
        Evaluates bug identification and fix correctness.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_debug = MetricsManager._get_output_value(example)
            pred_debug = MetricsManager._get_output_value(pred)
            
            # Calculate debug solution similarity using BERT Score
            P, R, F1 = bert_score_metric([pred_debug], [gold_debug], lang="en", rescale_with_baseline=False)
            debug_similarity = float(F1.mean())
            
            # Evaluate debug quality using reference
            quality_ref = "This debug solution correctly identifies and fixes the bugs while maintaining code functionality."
            _, _, quality = bert_score_metric([pred_debug], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            # Combine metrics
            score = (debug_similarity + quality_score) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in code debugging metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _planning_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for planning tasks.
        Evaluates plan completeness, feasibility, and structure.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_plan = MetricsManager._get_output_value(example)
            pred_plan = MetricsManager._get_output_value(pred)
            
            # Calculate plan similarity using BERT Score
            P, R, F1 = bert_score_metric([pred_plan], [gold_plan], lang="en", rescale_with_baseline=False)
            plan_similarity = float(F1.mean())
            
            # Evaluate plan quality using reference
            quality_ref = "This plan is complete, well-structured, and presents feasible steps to achieve the goal."
            _, _, quality = bert_score_metric([pred_plan], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            # Combine metrics
            score = (plan_similarity + quality_score) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in planning metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _tool_use_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for tool use tasks.
        Evaluates appropriate tool selection and usage.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_usage = MetricsManager._get_output_value(example)
            pred_usage = MetricsManager._get_output_value(pred)
            
            # Calculate tool usage similarity using BERT Score
            P, R, F1 = bert_score_metric([pred_usage], [gold_usage], lang="en", rescale_with_baseline=False)
            usage_similarity = float(F1.mean())
            
            # Evaluate tool usage quality using reference
            quality_ref = "This solution demonstrates appropriate tool selection and effective usage to accomplish the task."
            _, _, quality = bert_score_metric([pred_usage], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            # Combine metrics
            score = (usage_similarity + quality_score) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in tool use metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _decision_making_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for decision making tasks.
        Evaluates decision quality and reasoning.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_decision = MetricsManager._get_output_value(example)
            pred_decision = MetricsManager._get_output_value(pred)
            
            # Calculate decision similarity using BERT Score
            P, R, F1 = bert_score_metric([pred_decision], [gold_decision], lang="en", rescale_with_baseline=False)
            decision_similarity = float(F1.mean())
            
            # Evaluate decision quality using reference
            quality_ref = "This decision is well-reasoned, considers relevant factors, and provides clear justification."
            _, _, quality = bert_score_metric([pred_decision], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            # Combine metrics
            score = (decision_similarity + quality_score) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in decision making metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _process_automation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for process automation tasks.
        Evaluates workflow efficiency and completeness.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_process = MetricsManager._get_output_value(example)
            pred_process = MetricsManager._get_output_value(pred)
            
            # Calculate process similarity using BERT Score
            P, R, F1 = bert_score_metric([pred_process], [gold_process], lang="en", rescale_with_baseline=False)
            process_similarity = float(F1.mean())
            
            # Evaluate process quality using reference
            quality_ref = "This automation process is efficient, complete, and properly handles all required steps and conditions."
            _, _, quality = bert_score_metric([pred_process], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            # Combine metrics
            score = (process_similarity + quality_score) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in process automation metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _reasoning_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for reasoning tasks.
        Evaluates logical consistency and step-by-step explanation.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_reasoning = MetricsManager._get_output_value(example)
            pred_reasoning = MetricsManager._get_output_value(pred)
            
            # Calculate reasoning similarity using BERT Score
            P, R, F1 = bert_score_metric([pred_reasoning], [gold_reasoning], lang="en", rescale_with_baseline=False)
            reasoning_similarity = float(F1.mean())
            
            # Evaluate reasoning quality using reference
            quality_ref = "This reasoning is logically sound, well-structured, and clearly explains each step of the thought process."
            _, _, quality = bert_score_metric([pred_reasoning], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            # Combine metrics
            score = (reasoning_similarity + quality_score) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in reasoning metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _recommendation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for recommendation tasks.
        Evaluates recommendation relevance and personalization.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_recommendation = MetricsManager._get_output_value(example)
            pred_recommendation = MetricsManager._get_output_value(pred)
            
            # Calculate recommendation similarity using BERT Score
            P, R, F1 = bert_score_metric([pred_recommendation], [gold_recommendation], lang="en", rescale_with_baseline=False)
            recommendation_similarity = float(F1.mean())
            
            # Evaluate recommendation quality using reference
            quality_ref = "This recommendation is relevant, personalized, and provides clear justification for the suggestions."
            _, _, quality = bert_score_metric([pred_recommendation], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            # Combine metrics
            score = (recommendation_similarity + quality_score) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in recommendation metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _data_analysis_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Compute metrics for data analysis tasks.
        Evaluates analysis accuracy and insight quality.
        
        Returns:
            float: Score between 0 and 1
        """
        try:
            gold_analysis = MetricsManager._get_output_value(example)
            pred_analysis = MetricsManager._get_output_value(pred)
            
            # Calculate analysis similarity using BERT Score
            P, R, F1 = bert_score_metric([pred_analysis], [gold_analysis], lang="en", rescale_with_baseline=False)
            analysis_similarity = float(F1.mean())
            
            # Evaluate analysis quality using reference
            quality_ref = "This analysis is thorough, accurate, and provides meaningful insights supported by the data."
            _, _, quality = bert_score_metric([pred_analysis], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            # Combine metrics
            score = (analysis_similarity + quality_score) / 2
            
            # Apply length penalty
            prompt_length = len(instructions.split()) if instructions else 0
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            
            return score * length_penalty
        except Exception as e:
            print(f"Error in data analysis metrics: {str(e)}")
            return 0.0
    
    @staticmethod
    def _multi_label_classification_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for multi-label classification tasks."""
        try:
            gold_labels = set(MetricsManager._get_output_value(example).split(','))
            pred_labels = set(MetricsManager._get_output_value(pred).split(','))

            gold_labels = set([label.replace('[', '').replace(']', '').replace('"', '').replace("'", '').strip() for label in gold_labels])
            pred_labels = set([label.replace('[', '').replace(']', '').replace('"', '').replace("'", '').strip() for label in pred_labels])
            
            
            # Calculate F1 score
            true_positives = len(gold_labels.intersection(pred_labels))
            precision = true_positives / len(pred_labels) if pred_labels else 0
            recall = true_positives / len(gold_labels) if gold_labels else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate Hamming similarity
            all_labels = gold_labels.union(pred_labels)
            correct_labels = sum(1 for label in all_labels if (label in gold_labels) == (label in pred_labels))
            hamming_similarity = correct_labels / len(all_labels) if all_labels else 1
            
            return (f1 + hamming_similarity) / 2
        except Exception as e:
            print(f"Error in multi-label classification metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _information_extraction_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for information extraction tasks."""
        try:
            gold_info = MetricsManager._get_output_value(example)
            pred_info = MetricsManager._get_output_value(pred)
            
            # Convert to sets of key-value pairs
            gold_items = set(re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', gold_info))
            pred_items = set(re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', pred_info))
            
            # Calculate F1 score
            true_positives = len(gold_items.intersection(pred_items))
            precision = true_positives / len(pred_items) if pred_items else 0
            recall = true_positives / len(gold_items) if gold_items else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate structure similarity
            P, R, F1 = bert_score_metric([pred_info], [gold_info], lang="en", rescale_with_baseline=False)
            structure_score = float(F1.mean())
            
            return (f1 + structure_score) / 2
        except Exception as e:
            print(f"Error in information extraction metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _paraphrasing_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for paraphrasing tasks."""
        try:
            gold_text = MetricsManager._get_output_value(example)
            pred_text = MetricsManager._get_output_value(pred)
            
            # Calculate semantic similarity
            P, R, F1 = bert_score_metric([pred_text], [gold_text], lang="en", rescale_with_baseline=False)
            semantic_similarity = float(F1.mean())
            
            # Calculate lexical diversity
            gold_words = set(gold_text.lower().split())
            pred_words = set(pred_text.lower().split())
            word_overlap = len(gold_words.intersection(pred_words)) / len(gold_words.union(pred_words))
            lexical_diversity = 1 - word_overlap
            
            return (0.7 * semantic_similarity + 0.3 * lexical_diversity)
        except Exception as e:
            print(f"Error in paraphrasing metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _conversation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for conversation tasks."""
        try:
            gold_response = MetricsManager._get_output_value(example)
            pred_response = MetricsManager._get_output_value(pred)
            
            # Calculate response similarity
            P, R, F1 = bert_score_metric([pred_response], [gold_response], lang="en", rescale_with_baseline=False)
            response_similarity = float(F1.mean())
            
            # Evaluate coherence
            coherent_ref = "This is a coherent, contextually appropriate, and well-structured response."
            _, _, coherence = bert_score_metric([pred_response], [coherent_ref], lang="en", rescale_with_baseline=False)
            coherence_score = float(coherence.mean())
            
            return (response_similarity + coherence_score) / 2
        except Exception as e:
            print(f"Error in conversation metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _negotiation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for negotiation tasks."""
        try:
            gold_response = MetricsManager._get_output_value(example)
            pred_response = MetricsManager._get_output_value(pred)
            
            # Calculate response appropriateness
            P, R, F1 = bert_score_metric([pred_response], [gold_response], lang="en", rescale_with_baseline=False)
            response_score = float(F1.mean())
            
            # Evaluate negotiation effectiveness
            effective_ref = "This response demonstrates effective negotiation strategy, maintains fairness, and works toward mutual agreement."
            _, _, effectiveness = bert_score_metric([pred_response], [effective_ref], lang="en", rescale_with_baseline=False)
            effectiveness_score = float(effectiveness.mean())
            
            return (response_score + effectiveness_score) / 2
        except Exception as e:
            print(f"Error in negotiation metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _code_generation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for code generation tasks."""
        try:
            gold_code = MetricsManager._get_output_value(example)
            pred_code = MetricsManager._get_output_value(pred)
            
            # Calculate code similarity
            P, R, F1 = bert_score_metric([pred_code], [gold_code], lang="en", rescale_with_baseline=False)
            code_similarity = float(F1.mean())
            
            # Evaluate code quality
            quality_ref = "This code follows best practices, is well-documented, and efficiently implements the required functionality."
            _, _, quality = bert_score_metric([pred_code], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            return (code_similarity + quality_score) / 2
        except Exception as e:
            print(f"Error in code generation metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _code_explanation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for code explanation tasks."""
        try:
            gold_explanation = MetricsManager._get_output_value(example)
            pred_explanation = MetricsManager._get_output_value(pred)
            
            # Calculate explanation similarity
            P, R, F1 = bert_score_metric([pred_explanation], [gold_explanation], lang="en", rescale_with_baseline=False)
            explanation_similarity = float(F1.mean())
            
            # Evaluate explanation quality
            quality_ref = "This explanation is clear, complete, and accurately describes the code's functionality and implementation details."
            _, _, quality = bert_score_metric([pred_explanation], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            return (explanation_similarity + quality_score) / 2
        except Exception as e:
            print(f"Error in code explanation metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _code_completion_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for code completion tasks."""
        try:
            gold_completion = MetricsManager._get_output_value(example)
            pred_completion = MetricsManager._get_output_value(pred)
            
            # Calculate completion similarity
            P, R, F1 = bert_score_metric([pred_completion], [gold_completion], lang="en", rescale_with_baseline=False)
            completion_similarity = float(F1.mean())
            
            # Evaluate completion quality
            quality_ref = "This code completion is syntactically correct and consistent with the surrounding context."
            _, _, quality = bert_score_metric([pred_completion], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            return (completion_similarity + quality_score) / 2
        except Exception as e:
            print(f"Error in code completion metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _code_debugging_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for code debugging tasks."""
        try:
            gold_debug = MetricsManager._get_output_value(example)
            pred_debug = MetricsManager._get_output_value(pred)
            
            # Calculate debug solution similarity
            P, R, F1 = bert_score_metric([pred_debug], [gold_debug], lang="en", rescale_with_baseline=False)
            debug_similarity = float(F1.mean())
            
            # Evaluate debug quality
            quality_ref = "This debug solution correctly identifies and fixes the bugs while maintaining code functionality."
            _, _, quality = bert_score_metric([pred_debug], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            return (debug_similarity + quality_score) / 2
        except Exception as e:
            print(f"Error in code debugging metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _planning_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for planning tasks."""
        try:
            gold_plan = MetricsManager._get_output_value(example)
            pred_plan = MetricsManager._get_output_value(pred)
            
            # Calculate plan similarity
            P, R, F1 = bert_score_metric([pred_plan], [gold_plan], lang="en", rescale_with_baseline=False)
            plan_similarity = float(F1.mean())
            
            # Evaluate plan quality
            quality_ref = "This plan is complete, well-structured, and presents feasible steps to achieve the goal."
            _, _, quality = bert_score_metric([pred_plan], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            return (plan_similarity + quality_score) / 2
        except Exception as e:
            print(f"Error in planning metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _tool_use_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for tool use tasks."""
        try:
            gold_usage = MetricsManager._get_output_value(example)
            pred_usage = MetricsManager._get_output_value(pred)
            
            # Calculate tool usage similarity
            P, R, F1 = bert_score_metric([pred_usage], [gold_usage], lang="en", rescale_with_baseline=False)
            usage_similarity = float(F1.mean())
            
            # Evaluate tool usage quality
            quality_ref = "This solution demonstrates appropriate tool selection and effective usage to accomplish the task."
            _, _, quality = bert_score_metric([pred_usage], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            return (usage_similarity + quality_score) / 2
        except Exception as e:
            print(f"Error in tool use metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _decision_making_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for decision making tasks."""
        try:
            gold_decision = MetricsManager._get_output_value(example)
            pred_decision = MetricsManager._get_output_value(pred)
            
            # Calculate decision similarity
            P, R, F1 = bert_score_metric([pred_decision], [gold_decision], lang="en", rescale_with_baseline=False)
            decision_similarity = float(F1.mean())
            
            # Evaluate decision quality
            quality_ref = "This decision is well-reasoned, considers relevant factors, and provides clear justification."
            _, _, quality = bert_score_metric([pred_decision], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            return (decision_similarity + quality_score) / 2
        except Exception as e:
            print(f"Error in decision making metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _process_automation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for process automation tasks."""
        try:
            gold_process = MetricsManager._get_output_value(example)
            pred_process = MetricsManager._get_output_value(pred)
            
            # Calculate process similarity
            P, R, F1 = bert_score_metric([pred_process], [gold_process], lang="en", rescale_with_baseline=False)
            process_similarity = float(F1.mean())
            
            # Evaluate process quality
            quality_ref = "This automation process is efficient, complete, and properly handles all required steps and conditions."
            _, _, quality = bert_score_metric([pred_process], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            return (process_similarity + quality_score) / 2
        except Exception as e:
            print(f"Error in process automation metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _reasoning_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for reasoning tasks."""
        try:
            gold_reasoning = MetricsManager._get_output_value(example)
            pred_reasoning = MetricsManager._get_output_value(pred)
            
            # Calculate reasoning similarity
            P, R, F1 = bert_score_metric([pred_reasoning], [gold_reasoning], lang="en", rescale_with_baseline=False)
            reasoning_similarity = float(F1.mean())
            
            # Evaluate reasoning quality
            quality_ref = "This reasoning is logically sound, well-structured, and clearly explains each step of the thought process."
            _, _, quality = bert_score_metric([pred_reasoning], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            return (reasoning_similarity + quality_score) / 2
        except Exception as e:
            print(f"Error in reasoning metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _recommendation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for recommendation tasks."""
        try:
            gold_recommendation = MetricsManager._get_output_value(example)
            pred_recommendation = MetricsManager._get_output_value(pred)
            
            # Calculate recommendation similarity
            P, R, F1 = bert_score_metric([pred_recommendation], [gold_recommendation], lang="en", rescale_with_baseline=False)
            recommendation_similarity = float(F1.mean())
            
            # Evaluate recommendation quality
            quality_ref = "This recommendation is relevant, personalized, and provides clear justification for the suggestions."
            _, _, quality = bert_score_metric([pred_recommendation], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            return (recommendation_similarity + quality_score) / 2
        except Exception as e:
            print(f"Error in recommendation metrics: {str(e)}")
            return 0.0

    @staticmethod
    def _data_analysis_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
        """Final evaluation metrics for data analysis tasks."""
        try:
            gold_analysis = MetricsManager._get_output_value(example)
            pred_analysis = MetricsManager._get_output_value(pred)
            
            # Calculate analysis similarity
            P, R, F1 = bert_score_metric([pred_analysis], [gold_analysis], lang="en", rescale_with_baseline=False)
            analysis_similarity = float(F1.mean())
            
            # Evaluate analysis quality
            quality_ref = "This analysis is thorough, accurate, and provides meaningful insights supported by the data."
            _, _, quality = bert_score_metric([pred_analysis], [quality_ref], lang="en", rescale_with_baseline=False)
            quality_score = float(quality.mean())
            
            return (analysis_similarity + quality_score) / 2
        except Exception as e:
            print(f"Error in data analysis metrics: {str(e)}")
            return 0.0
    
    