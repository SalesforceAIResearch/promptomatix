from typing import Optional
import dspy
from dspy.backends import OpenAI, Anthropic, Cohere

class LMManager:
    """Manages Language Model initialization and configuration for different providers"""
    
    SUPPORTED_PROVIDERS = {
        'openai': OpenAI,
        'anthropic': Anthropic,
        'cohere': Cohere,
        # Add more providers as needed
    }

    @classmethod
    def get_lm(cls, 
               provider: str, 
               model_name: str, 
               api_key: str, 
               api_base: Optional[str] = None,
               temperature: float = 0.7,
               max_tokens: int = 4000,
               **kwargs):
        """
        Initialize and return appropriate language model based on provider.
        
        Args:
            provider: The model provider (e.g., 'openai', 'anthropic')
            model_name: Name of the model to use
            api_key: API key for the provider
            api_base: Optional API base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Initialized language model instance
            
        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower()
        if provider not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers are: {list(cls.SUPPORTED_PROVIDERS.keys())}"
            )

        # Get the appropriate LM class
        lm_class = cls.SUPPORTED_PROVIDERS[provider]
        
        # Prepare base arguments
        lm_args = {
            "model": model_name,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Add provider-specific configurations
        if provider == 'openai' and api_base:
            lm_args["api_base"] = api_base
            
        # Initialize the LM
        try:
            return lm_class(**lm_args)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {provider} LM: {str(e)}")

    @staticmethod
    def configure_dspy(lm) -> None:
        """Configure DSPy with the given language model"""
        dspy.configure(lm=lm) 