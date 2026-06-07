# --- Helper Script for AI-Driven Software Engineering Course ---
# Description: This script provides a unified interface for interacting with
#              multiple LLM providers (OpenAI, Anthropic, Hugging Face, Google Gemini)
#              and simplifies common tasks like artifact management.
# -----------------------------------------------------------------
import os
import json
import requests
from PIL import Image
from io import BytesIO
import re
import base64
import mimetypes
import time # For loading indicator
from urllib.parse import quote

# --- Dynamic Library Installation ---
try:
    from dotenv import load_dotenv
    from IPython.display import display, Markdown, Image as IPyImage
    from plantuml import PlantUML
except ImportError:
    # Provide safe fallbacks so the module can be imported even when optional deps are missing.
    print("Warning: Optional core dependencies not found. Some features will be degraded.")
    print("To enable full functionality run: pip install python-dotenv ipython plantuml")

    # noop load_dotenv fallback
    def load_dotenv(*args, **kwargs):
        print("Warning: python-dotenv not installed; .env will not be loaded.")

    # minimal IPython.display fallbacks
    def display(*args, **kwargs):
        # no-op in non-notebook environments
        return None

    def Markdown(text):
        return text

    class IPyImage:
        def __init__(self, *args, **kwargs):
            # placeholder for notebook image object
            pass

    # PlantUML fallback
    class PlantUML:
        def __init__(self, url=None):
            print("Warning: plantuml not installed; rendering disabled.")
        def processes(self, *args, **kwargs):
            print("PlantUML rendering skipped (plantuml not installed).")


# --- Model & Provider Configuration ---
RECOMMENDED_MODELS = {
    "gpt-5-nano-2025-08-07": {"provider": "openai", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 400_000, "output_tokens": 128_000},
    "gpt-5-mini-2025-08-07": {"provider": "openai", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 400_000, "output_tokens": 128_000},
    "gpt-5-2025-08-07": {"provider": "openai", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 400_000, "output_tokens": 128_000},
    "gpt-4o": {"provider": "openai", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 128_000, "output_tokens": 16_384},
    "gpt-4o-mini": {"provider": "openai", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 128_000, "output_tokens": 16_384},
    "gpt-4.1": {"provider": "openai", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 1_000_000, "output_tokens": 32_768},
    "gpt-4.1-mini": {"provider": "openai", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 1_000_000, "output_tokens": 32_000},
    "gpt-4.1-nano": {"provider": "openai", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 1_000_000, "output_tokens": 32_000},
    "o3": {"provider": "openai", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 200_000, "output_tokens": 100_000},
    "o4-mini": {"provider": "openai", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 200_000, "output_tokens": 100_000},
    "dall-e-3": {"provider": "openai", "vision": False, "text_generation": False, "image_generation": True, "image_modification": False, "audio_transcription": False, "context_window_tokens": None, "output_tokens": None},
    "whisper-1": {"provider": "openai", "vision": False, "text_generation": False, "image_generation": False, "image_modification": False, "audio_transcription": True, "context_window_tokens": None, "output_tokens": None},
    "claude-opus-4-1-20250805": {"provider": "anthropic", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 200_000, "output_tokens": 100_000},
    "claude-opus-4-20250514": {"provider": "anthropic", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 200_000, "output_tokens": 100_000},
    "claude-sonnet-4-20250514": {"provider": "anthropic", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 1_000_000, "output_tokens": 100_000},
    "gemini-2.5-pro": {"provider": "google", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 1_048_576, "output_tokens": 65_536},
    "gemini-2.5-flash": {"provider": "google", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 1_048_576, "output_tokens": 65_536},
    "gemini-2.5-flash-lite": {"provider": "google", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 1_048_576, "output_tokens": 65_536},
    "gemini-live-2.5-flash-preview": {"provider": "google", "vision": False, "text_generation": False, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 1_048_576, "output_tokens": 8_192},
    "gemini-2.5-flash-image-preview": {"provider": "google", "vision": False, "text_generation": False, "image_generation": True, "image_modification": False, "audio_transcription": False, "context_window_tokens": 32_768, "output_tokens": 32_768},
    "gemini-2.0-flash-preview-image-generation": {"provider": "google", "vision": False, "text_generation": False, "image_generation": True, "image_modification": False, "audio_transcription": False, "context_window_tokens": 32_000, "output_tokens": 8_192},
    "gemini-1.5-pro": {"provider": "google", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 2_000_000, "output_tokens": 8_192},
    "gemini-1.5-flash": {"provider": "google", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 1_000_000, "output_tokens": 8_192},
    "gemini-2.0-flash-exp": {"provider": "google", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 1_048_576, "output_tokens": 8_192},
    "veo-3.0-generate-preview": {"provider": "google", "vision": False, "text_generation": False, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 1_024, "output_tokens": None},
    "veo-3.0-fast-generate-preview": {"provider": "google", "vision": False, "text_generation": False, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 1_024, "output_tokens": None},
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {"provider": "huggingface", "vision": False, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 10_000_000, "output_tokens": 100_000},
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": {"provider": "huggingface", "vision": False, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 1_000_000, "output_tokens": 100_000},
    "meta-llama/Llama-3.3-70B-Instruct": {"provider": "huggingface", "vision": False, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 8_192, "output_tokens": 4_096},
    "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5": {"provider": "huggingface", "vision": False, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 4_096, "output_tokens": 1_024},
    "mistralai/Mistral-7B-Instruct-v0.3": {"provider": "huggingface", "vision": False, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 32_768, "output_tokens": 8_192},
    "deepseek-ai/DeepSeek-V3.1": {"provider": "huggingface", "vision": False, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 128_000, "output_tokens": 100_000},
    "deepseek-chat": {"provider": "deepseek", "vision": False, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 4_000_000, "output_tokens": 8_000},
    "deepseek-reasoner": {"provider": "deepseek", "vision": False, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": 32_000, "output_tokens": 8_000},
    "deepseek-vision": {"provider": "deepseek", "vision": True, "text_generation": False, "image_generation": True, "image_modification": False, "audio_transcription": False, "context_window_tokens": 4_000_000, "output_tokens": 8_000},
    "openai/gpt-5.2": {"provider": "apifree", "vision": False, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": None, "output_tokens": None},
    "openai/gpt-4o": {"provider": "apifree", "vision": False, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": None, "output_tokens": None},
    "google/gemini-2.5-pro": {"provider": "apifree", "vision": True, "text_generation": True, "image_generation": False, "image_modification": False, "audio_transcription": False, "context_window_tokens": None, "output_tokens": None},
    "Qwen/Qwen-Image": {"provider": "huggingface", "vision": False, "text_generation": False, "image_generation": True, "image_modification": False, "audio_transcription": False, "context_window_tokens": None, "output_tokens": None},
    "Qwen/Qwen-Image-Edit": {"provider": "huggingface", "vision": False, "text_generation": False, "image_generation": False, "image_modification": True, "audio_transcription": False, "context_window_tokens": None, "output_tokens": None},
    "qwen/qwen-image-2512": {"provider": "apifree", "vision": False, "text_generation": False, "image_generation": True, "image_modification": False, "audio_transcription": False, "context_window_tokens": None, "output_tokens": None},
    "qwen/qwen-image-edit-2511": {"provider": "apifree", "vision": False, "text_generation": False, "image_generation": False, "image_modification": True, "audio_transcription": False, "context_window_tokens": None, "output_tokens": None},
    "stabilityai/stable-diffusion-3.5-large": {"provider": "huggingface", "vision": False, "text_generation": False, "image_generation": True, "image_modification": False, "audio_transcription": False, "context_window_tokens": None, "output_tokens": None},
    "black-forest-labs/FLUX.1-Kontext-dev": {"provider": "huggingface", "vision": False, "text_generation": False, "image_generation": False, "image_modification": True, "audio_transcription": False, "context_window_tokens": None, "output_tokens": None},
}


def recommended_models_table(task=None, provider=None, text_generation=None, vision=None, image_generation=None,
                             audio_transcription=None, min_context=None, min_output_tokens=None,
                             image_modification=None):
    """
    Return a markdown table of recommended models, optionally filtered by attributes.

    Args:
        task (str, optional): High level task to filter models. Accepts values like
            'text', 'vision', 'image', 'audio'. These set sensible defaults for
            the corresponding capability flags unless explicitly provided.
        provider (str, optional): Filter models by provider name (e.g. ``'openai'``).
        text_generation (bool, optional): If set, include only models supporting text generation.
        vision (bool, optional): If set, include only models that match vision capability.
        image_generation (bool, optional): If set, include only image generation models.
        audio_transcription (bool, optional): If set, include only models supporting
            audio transcription.
        min_context (int, optional): Minimum context window size required.
        min_output_tokens (int, optional): Minimum max output tokens required.
        image_modification (bool, optional): If set, include only models supporting
            image editing/modification. Currently flagged editing models are
            "Qwen/Qwen-Image-Edit" and "black-forest-labs/FLUX.1-Kontext-dev".

    Returns:
        str: Markdown formatted table.
    """
    # Interpret task shortcuts
    if task:
        t = task.lower()
        if t in {"vision", "multimodal", "vl"} and vision is None:
            vision = True
        elif t in {"image", "image_generation", "image-generation"} and image_generation is None:
            image_generation = True
        elif t in {"image_modification", "image-edit", "image_edit", "image-editing", "editing"} and image_modification is None:
            image_modification = True
        elif t in {"audio", "speech", "audio_transcription", "stt"} and audio_transcription is None:
            audio_transcription = True
        elif t == "text" and text_generation is None:
            text_generation = True
            # Also ensure other modalities are off unless specified
            vision = False if vision is None else vision
            image_generation = False if image_generation is None else image_generation
            image_modification = False if image_modification is None else image_modification
            audio_transcription = False if audio_transcription is None else audio_transcription

    rows = []
    for model_name in sorted(RECOMMENDED_MODELS.keys()):
        cfg = RECOMMENDED_MODELS[model_name]
        model_provider = (cfg.get("provider") or "").lower()
        model_text = cfg.get("text_generation", False)
        model_vision = cfg.get("vision", False)
        model_image = cfg.get("image_generation", False)
        model_image_mod = cfg.get("image_modification", False)
        model_audio = cfg.get("audio_transcription", False)

        # Prefer canonical integer fields used in RECOMMENDED_MODELS
        context = cfg.get("context_window_tokens")
        if context is None:
            # Backwards-compat: allow older key name
            context = cfg.get("context_window")

        max_tokens = cfg.get("output_tokens")
        if max_tokens is None:
            max_tokens = cfg.get("max_output_tokens")

        # Normalize provider filter to be case-insensitive
        if provider and model_provider != provider.lower():
            continue
        if text_generation is not None and bool(model_text) != bool(text_generation):
            continue
        if vision is not None and bool(model_vision) != bool(vision):
            continue
        if image_generation is not None and bool(model_image) != bool(image_generation):
            continue
        if image_modification is not None and bool(model_image_mod) != bool(image_modification):
            continue
        if audio_transcription is not None and bool(model_audio) != bool(audio_transcription):
            continue
        if min_context and (context is None or (isinstance(context, int) and context < min_context)):
            continue
        if min_output_tokens and (max_tokens is None or (isinstance(max_tokens, int) and max_tokens < min_output_tokens)):
            continue

        def _fmt_num(x):
            if x is None:
                return "-"
            try:
                # format large ints with commas
                return f"{int(x):,}"
            except Exception:
                return str(x)

        rows.append(
            f"| {model_name} | {model_provider or '-'} | {'✅' if model_text else '❌'} | {'✅' if model_vision else '❌'} | "
            f"{'✅' if model_image else '❌'} | {'✅' if model_image_mod else '❌'} | {'✅' if model_audio else '❌'} | "
            f"{_fmt_num(context)} | {_fmt_num(max_tokens)} |"
        )

    if not rows:
        return "No models match the specified criteria."

    header = (
        "| Model | Provider | Text | Vision | Image Gen | Image Edit | Audio Transcription | Context Window | Max Output Tokens |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    table = header + "\n".join(rows)
    display(Markdown(table))
    return table

# --- Environment and API Client Setup ---

def load_environment():
    """
    Loads environment variables from a .env file in the project root.
    
    This function searches upward from the current working directory to find the
    project root by looking for specific markers (.env file or .git directory).
    Once found, it loads all environment variables defined in the .env file,
    making them available to the application through os.getenv().
    
    Args:
        None
    
    Returns:
        None: This function doesn't return a value but has the side effect of
            loading environment variables into the process environment.
    
    Raises:
        None: This function handles all errors gracefully and prints warnings
            instead of raising exceptions.
    
    Notes:
        - Searches upward from current directory until it finds .env or .git
        - Falls back to current directory if no markers are found
        - Uses python-dotenv library to parse and load the .env file
        - Prints a warning if .env file is not found
        - Environment variables loaded are accessible via os.getenv()
    
    Example:
        >>> load_environment()
        >>> api_key = os.getenv('OPENAI_API_KEY')
        
        # Typical .env file content:
        # OPENAI_API_KEY=sk-...
        # ANTHROPIC_API_KEY=sk-ant-...
        # GOOGLE_API_KEY=AIza...
    
    Dependencies:
        - python-dotenv: For parsing and loading .env files
        - os: For file system operations and environment variable access
    """
    path = os.getcwd()
    while path != os.path.dirname(path):
        if os.path.exists(os.path.join(path, '.env')) or os.path.exists(os.path.join(path, '.git')):
            project_root = path
            break
        path = os.path.dirname(path)
    else:
        project_root = os.getcwd()

    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print("Warning: .env file not found. API keys may not be loaded.")


def setup_llm_client(model_name="gpt-4o"):
    """
    Configures and returns an LLM client based on the specified model name.
    
    This function initializes the appropriate API client for the specified model,
    handling authentication and configuration for multiple providers including
    OpenAI, Anthropic, Hugging Face, and Google (Gemini/Speech-to-Text).
    It automatically loads environment variables and validates API keys.
    
    Args:
        model_name (str, optional): The identifier of the model to use. Must be
            a key in the RECOMMENDED_MODELS dictionary. Defaults to "gpt-4o".
            Examples: "gpt-4o", "claude-3-opus-20240229", "gemini-2.5-pro"
    
    Returns:
        tuple: A 3-element tuple containing:
            - client: The initialized API client object (varies by provider)
                - OpenAI: OpenAI client instance
                - Anthropic: Anthropic client instance
                - Hugging Face: InferenceClient instance
                - Google: GenerativeModel, genai module, or SpeechClient
            - model_name (str): The model name (echoed back)
            - api_provider (str): The provider name ("openai", "anthropic", etc.)
            
            Returns (None, None, None) if initialization fails.
    
    Raises:
        None: This function handles all errors gracefully and prints error messages
            instead of raising exceptions.
    
    Notes:
        - Automatically calls load_environment() to load .env file
        - Validates that the model exists in RECOMMENDED_MODELS
        - Checks for required API keys in environment variables
        - Handles ImportError if provider libraries aren't installed
        - Special handling for Google models based on their capabilities:
            - Audio transcription models use google.cloud.speech
            - Image generation models return the genai module
            - Text/vision models return a GenerativeModel instance
        - Prints success/error messages to console
    
    Example:
        >>> # Initialize OpenAI client
        >>> client, model, provider = setup_llm_client("gpt-4o")
        ✅ LLM Client configured: Using 'openai' with model 'gpt-4o'
        
        >>> # Initialize Anthropic client
        >>> client, model, provider = setup_llm_client("claude-3-opus-20240229")
        ✅ LLM Client configured: Using 'anthropic' with model 'claude-3-opus-20240229'
        
        >>> # Handle missing API key
        >>> client, model, provider = setup_llm_client("gpt-4o")
        ERROR: OPENAI_API_KEY not found in .env file.
    
    Dependencies:
        - Provider-specific libraries (installed as needed):
            - openai: For OpenAI models
            - anthropic: For Anthropic models
            - huggingface_hub: For Hugging Face models
            - google.generativeai: For Google Gemini
            - google.cloud.speech: For Google Speech-to-Text
        - RECOMMENDED_MODELS: Global dictionary with model configurations
    """
    load_environment()
    if model_name not in RECOMMENDED_MODELS:
        print(f"ERROR: Model '{model_name}' is not in the list of recommended models.")
        return None, None, None
    config = RECOMMENDED_MODELS[model_name]
    api_provider = config["provider"]
    client = None
    try:
        if api_provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key: raise ValueError("OPENAI_API_KEY not found in .env file.")
            client = OpenAI(api_key=api_key)
        elif api_provider == "anthropic":
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key: raise ValueError("ANTHROPIC_API_KEY not found in .env file.")
            client = Anthropic(api_key=api_key)
        elif api_provider == "huggingface":
            from huggingface_hub import InferenceClient
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key: raise ValueError("HUGGINGFACE_API_KEY not found in .env file.")
            client = InferenceClient(model=model_name, token=api_key)
        elif api_provider == "gemini" or api_provider == "google": # Google for text/vision, Imagen, or STT
            if config.get("audio_transcription"):
                from google.cloud import speech
                client = speech.SpeechClient()
            else:
                # Decide based on model family
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key: raise ValueError("GOOGLE_API_KEY not found in .env file.")
                if "imagen" in model_name:
                    # Use the new google-genai low-level Client for Imagen
                    from google import genai as google_genai
                    client = google_genai.Client(api_key=api_key)
                else:
                    # Use google-generativeai GenerativeModel for Gemini text/vision
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    client = genai.GenerativeModel(model_name)
        elif api_provider == "deepseek":
            from openai import OpenAI
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key: raise ValueError("DEEPSEEK_API_KEY not found in .env file.")
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        elif api_provider == "apifree":
            api_key = os.getenv("APIFREE_API_KEY")
            if not api_key: raise ValueError("APIFREE_API_KEY not found in .env file.")
            client = {"provider": "apifree"}
    except ImportError:
        print(f"ERROR: The required library for '{api_provider}' is not installed.")
        return None, None, None
    except ValueError as e:
        print(f"ERROR: {e}")
        return None, None, None
    print(f"✅ LLM Client configured: Using '{api_provider}' with model '{model_name}'")
    return client, model_name, api_provider

# --- Core Interaction Functions ---

def get_completion(prompt, client, model_name, api_provider, temperature=0.7):
    """
    Sends a text-only prompt to the LLM and returns the completion.
    
    This function provides a unified interface for getting text completions from
    various LLM providers. It handles provider-specific API differences and
    error cases, including special handling for newer OpenAI models that may
    use different endpoints or not support temperature parameters.
    
    Args:
        prompt (str): The text prompt to send to the model. This is the user's
            input or question that the model should respond to.
        client: The initialized API client object from setup_llm_client().
            Type varies by provider (OpenAI, Anthropic, InferenceClient, etc.)
        model_name (str): The identifier of the model to use for completion.
        api_provider (str): The provider name ("openai", "anthropic", "huggingface",
            "gemini", or "google").
        temperature (float, optional): Controls randomness in the output. Higher
            values (e.g., 1.0) make output more random, lower values (e.g., 0.1)
            make it more deterministic. Defaults to 0.7. Range typically 0.0-2.0.
    
    Returns:
        str: The generated text completion from the model. Returns an error
            message string if the API call fails.
    
    Raises:
        None: This function catches all exceptions and returns error messages
            as strings instead of raising exceptions.
    
    Notes:
        - Handles different API structures for each provider
        - OpenAI: Tries chat completions first, falls back to responses endpoint
        - Anthropic: Uses messages API with max_tokens=4096
        - Hugging Face: Uses chat_completion with minimum temperature of 0.1
        - Google/Gemini: Uses generate_content method
        - Special error handling for OpenAI models that don't support temperature
        - Returns descriptive error messages if API calls fail
    
    Example:
        >>> client, model, provider = setup_llm_client("gpt-4o")
        >>> response = get_completion(
        ...     "What is the capital of France?",
        ...     client, model, provider, temperature=0.5
        ... )
        >>> print(response)
        "The capital of France is Paris."
        
        >>> # Handle API errors gracefully
        >>> response = get_completion("Hello", None, "gpt-4o", "openai")
        >>> print(response)
        "API client not initialized."
    
    Dependencies:
        - Provider-specific client libraries
        - RECOMMENDED_MODELS: For model capability validation
    """
    if not client and api_provider != "apifree":
        return "API client not initialized."
    try:
        if api_provider == "apifree":
            api_key = os.getenv("APIFREE_API_KEY")
            if not api_key:
                return "Error: APIFREE_API_KEY not found in .env file."
            if not model_name:
                return "Error: model_name is required for APIFREE requests."

            base_url = os.getenv("APIFREE_API_BASE", "https://api.apifree.ai")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            def _extract_text(response_data):
                choices = response_data.get("choices")
                if not choices and isinstance(response_data.get("resp_data"), dict):
                    choices = response_data["resp_data"].get("choices")
                if not choices:
                    return None
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            parts.append(item)
                    content = "".join(parts).strip()
                return content

            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
                "temperature": temperature,
            }

            try:
                resp = requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=payload, timeout=120)
                if not resp.ok:
                    return f"APIFREE error: {resp.status_code} {resp.text}"
                data = resp.json()
                content = _extract_text(data)
                if isinstance(content, str) and content.strip():
                    return content

                retry_payload = dict(payload)
                retry_payload["max_tokens"] = 512
                retry_resp = requests.post(
                    f"{base_url}/v1/chat/completions",
                    headers=headers,
                    json=retry_payload,
                    timeout=120,
                )
                if not retry_resp.ok:
                    return f"APIFREE error: {retry_resp.status_code} {retry_resp.text}"
                retry_data = retry_resp.json()
                retry_content = _extract_text(retry_data)
                if isinstance(retry_content, str) and retry_content.strip():
                    return retry_content

                # Last resort: return abbreviated prompt reminder with explicit instruction
                return f"[SYSTEM: Model failed to generate sufficient content. Please rephrase request and retry. Original prompt was: {prompt[:200]}...]"
            except Exception as e:
                return f"APIFREE request failed: {e}"

        if api_provider == "openai":
            # Some newer models use different endpoints
            try:
                # Try chat completions first (standard endpoint)
                response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=temperature)
                return response.choices[0].message.content
            except Exception as api_error:
                error_message = str(api_error).lower()
                
                if "temperature" in error_message and "unsupported" in error_message:
                    # Retry without temperature parameter
                    try:
                        response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
                        return response.choices[0].message.content
                    except Exception as retry_error:
                        if "v1/responses" in str(retry_error):
                            # Use the responses endpoint for certain models
                            response = client.responses.create(model=model_name, input=prompt)
                            return response.choices[0].text
                        else:
                            raise retry_error
                elif "v1/responses" in str(api_error):
                    # Use the responses endpoint for certain models
                    try:
                        response = client.responses.create(model=model_name, input=prompt, temperature=temperature)
                        return response.text
                    except Exception:
                        # Try responses endpoint without temperature
                        response = client.responses.create(model=model_name, input=prompt)
                        return response.text
                else:
                    raise api_error
        elif api_provider == "anthropic":
            response = client.messages.create(
                model=model_name,
                max_tokens=4096,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif api_provider == "huggingface":
            response = client.chat_completion(messages=[{"role": "user", "content": prompt}], temperature=max(0.1, temperature), max_tokens=4096)
            return response.choices[0].message.content
        elif api_provider == "gemini" or api_provider == "google":
            response = client.generate_content(prompt)
            return response.text
        elif api_provider == "deepseek":
            # DeepSeek uses OpenAI-compatible API
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"An API error occurred: {e}"

def get_vision_completion(prompt, image_path_or_url, client, model_name, api_provider):
    """
    Sends an image and a text prompt to a vision-capable LLM and returns the completion.
    
    This function enables multimodal AI interactions by processing both text and image
    inputs together. It handles the different image processing requirements for each
    provider, including URL-based and base64-encoded image formats. It can accept
    either a public URL to an image or a local file path.
    
    Args:
        prompt (str): The text prompt or question about the image. This provides
            context or specific instructions for analyzing the image.
        image_path_or_url (str): URL of the image to analyze or a local file path.
            If a URL, it must be publicly accessible.
        client: The initialized API client object from setup_llm_client().
            Type varies by provider.
        model_name (str): The identifier of the vision-capable model to use.
        api_provider (str): The provider name ("openai", "anthropic", "huggingface",
            "gemini", "google", "freepik", or "apifree").
    
    Returns:
        str: The model's response analyzing the image based on the prompt.
            Returns an error message string if the model doesn't support vision
            or if the API call fails.
    
    Raises:
        None: This function catches all exceptions and returns error messages
            as strings instead of raising exceptions.
    
    Notes:
        - Validates that the model supports vision using RECOMMENDED_MODELS
        - Handles both URLs and local file paths for images.
        - Different providers require different image formats:
            - OpenAI: Can use image URLs directly or base64-encoded data URLs.
            - Anthropic: Requires base64-encoded image data with MIME type.
            - Google/Gemini: Requires PIL Image object.
            - Hugging Face: Requires PIL Image object.
            - Freepik Mystic: Generates images from prompt using FREEPIK_API_KEY and
                optionally uses the provided image as a structure reference.
            - APIFREE: Sends a prompt + image_urls payload to the model endpoint using
                APIFREE_API_KEY and returns the raw response.
        - Automatically downloads images from URLs or reads from disk and converts as needed.
        - Sets max_tokens to 4096 for providers that support it.
        - Handles HTTP errors when fetching images and file I/O errors.
    
    Example:
        >>> client, model, provider = setup_llm_client("gpt-4o")
        >>> # Using a URL
        >>> response_url = get_vision_completion(
        ...     "What objects do you see in this image?",
        ...     "https://example.com/image.jpg",
        ...     client, model, provider
        ... )
        >>> # Using a local file
        >>> response_local = get_vision_completion(
        ...     "Describe this local image.",
        ...     "artifacts/screens/my_image.png",
        ...     client, model, provider
        ... )
        
    Dependencies:
        - requests: For downloading images from URLs
        - PIL (Pillow): For image processing
        - base64: For encoding images
        - mimetypes: For determining image type from file extension
        - io.BytesIO: For converting image bytes to PIL Images
        - RECOMMENDED_MODELS: For vision capability validation
    """
    if api_provider not in {"freepik", "apifree"}:
        if not client:
            return "API client not initialized."
        if not RECOMMENDED_MODELS.get(model_name, {}).get("vision"):
            return f"Error: Model '{model_name}' does not support vision."

    if api_provider == "apifree":
        api_key = os.getenv("APIFREE_API_KEY")
        if not api_key:
            return "Error: APIFREE_API_KEY not found in .env file."
        if not image_path_or_url:
            return "Error: image_path_or_url is required for APIFREE requests."

        is_url = image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://")
        if is_url:
            image_urls = [image_path_or_url]
        else:
            project_root = _find_project_root()
            resolved_path = os.path.join(project_root, image_path_or_url)
            if not os.path.exists(resolved_path):
                return f"Error: Local image file not found at {image_path_or_url}"
            with open(resolved_path, "rb") as f:
                img_content = f.read()
            mime_type = mimetypes.guess_type(resolved_path)[0] or "image/png"
            data_url = f"data:{mime_type};base64,{base64.b64encode(img_content).decode('utf-8')}"
            image_urls = [data_url]

        if not model_name:
            return "Error: model_name is required for APIFREE requests."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        base_url = os.getenv("APIFREE_API_BASE", "https://api.apifree.ai")

        width = None
        height = None
        if not is_url:
            try:
                with Image.open(resolved_path) as im:
                    width, height = im.size
            except Exception:
                width = None
                height = None

        try:
            # Some APIFREE OpenAI models (e.g., gpt-5.2) expect chat-style inputs
            if model_name.startswith("openai/gpt-5.2"):
                def _extract_text(response_data):
                    choices = response_data.get("choices")
                    if not choices and isinstance(response_data.get("resp_data"), dict):
                        choices = response_data["resp_data"].get("choices")
                    if not choices:
                        return None
                    message = choices[0].get("message") or {}
                    content = message.get("content")
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                parts.append(item.get("text", ""))
                            elif isinstance(item, str):
                                parts.append(item)
                        content = "".join(parts).strip()
                    return content

                chat_payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_urls[0]}},
                            ],
                        }
                    ],
                    "max_tokens": 512,
                }
                resp = requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=chat_payload, timeout=120)
                if not resp.ok:
                    return f"APIFREE error: {resp.status_code} {resp.text}"
                data = resp.json()
                content = _extract_text(data)
                if isinstance(content, str) and content.strip():
                    return content

                # Retry with a simpler response format if content is empty
                retry_payload = dict(chat_payload)
                retry_payload["max_tokens"] = 256
                retry_payload["response_format"] = {"type": "text"}
                retry_resp = requests.post(
                    f"{base_url}/v1/chat/completions",
                    headers=headers,
                    json=retry_payload,
                    timeout=120,
                )
                if not retry_resp.ok:
                    return f"APIFREE error: {retry_resp.status_code} {retry_resp.text}"
                retry_data = retry_resp.json()
                retry_content = _extract_text(retry_data)
                if isinstance(retry_content, str) and retry_content.strip():
                    return retry_content
                return retry_data

            payload = {
                "model": model_name,
                "prompt": prompt,
                "image_urls": image_urls,
                "num_images": 1,
                "num_inference_steps": 28,
            }
            if width and height:
                payload["width"] = width
                payload["height"] = height

            submit_url = f"{base_url}/v1/image/submit"
            resp = requests.post(submit_url, headers=headers, json=payload, timeout=120)
            if not resp.ok:
                return f"APIFREE error: {resp.status_code} {resp.text}"

            data = resp.json()
            if data.get("code") != 200:
                return f"APIFREE error: {data.get('error') or data}"

            request_id = data.get("resp_data", {}).get("request_id")
            if not request_id:
                return f"APIFREE error: Missing request_id. Response: {data}"

            result_url = f"{base_url}/v1/image/{request_id}/result"
            for _ in range(30):
                time.sleep(2)
                check_resp = requests.get(result_url, headers=headers, timeout=60)
                if not check_resp.ok:
                    return f"APIFREE error: {check_resp.status_code} {check_resp.text}"
                check_data = check_resp.json()
                if check_data.get("code") != 200:
                    return f"APIFREE error: {check_data.get('code_msg') or check_data}"
                status = check_data.get("resp_data", {}).get("status")
                if status == "success":
                    return check_data.get("resp_data", {})
                if status in {"error", "failed"}:
                    return f"APIFREE error: {check_data.get('resp_data', {}).get('error') or check_data}"
            return "APIFREE error: timed out waiting for result"
        except Exception as e:
            return f"APIFREE request failed: {e}"

    # Freepik generation does not require an input image
    if api_provider == "freepik":
        api_key = os.getenv("FREEPIK_API_KEY")
        if not api_key:
            return "Error: FREEPIK_API_KEY not found in .env file."

        payload = {"prompt": prompt}

        if image_path_or_url:
            try:
                is_url = image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://')
                if is_url:
                    response_img = requests.get(image_path_or_url)
                    response_img.raise_for_status()
                    img_content = response_img.content
                else:
                    project_root = _find_project_root()
                    resolved_path = os.path.join(project_root, image_path_or_url)
                    if not os.path.exists(resolved_path):
                        return f"Error: Local image file not found at {image_path_or_url}"
                    with open(resolved_path, "rb") as f:
                        img_content = f.read()
                payload["structure_reference"] = base64.b64encode(img_content).decode("utf-8")
                payload["structure_strength"] = 50
            except Exception as img_error:
                return f"Error preparing structure_reference for Freepik: {img_error}"

        headers = {
            "Content-Type": "application/json",
            "x-freepik-api-key": api_key,
        }

        try:
            resp = requests.post("https://api.freepik.com/v1/ai/mystic", json=payload, headers=headers, timeout=60)
            if not resp.ok:
                return f"Freepik Mystic error: {resp.status_code} {resp.text}"

            data = resp.json().get("data", {})
            generated = data.get("generated") or []
            if generated:
                first_url = generated[0]
                try:
                    img_resp = requests.get(first_url)
                    img_resp.raise_for_status()
                    mime_type = img_resp.headers.get("Content-Type", "image/png")
                    image_data_base64 = base64.b64encode(img_resp.content).decode("utf-8")

                    return f"data:{mime_type};base64,{image_data_base64}"
                except Exception as dl_error:
                    return f"Freepik image URL generated but could not download: {dl_error}. URL: {first_url}"

            task_id = data.get("task_id")
            status = data.get("status")
            if not task_id:
                return f"Freepik task created but no task_id returned. status={status}"

            # Poll for completion so the image is available
            poll_url = f"https://api.freepik.com/v1/ai/mystic/{task_id}"
            max_wait_seconds = 120
            poll_interval = 5
            waited = 0

            while waited < max_wait_seconds:
                time.sleep(poll_interval)
                waited += poll_interval
                try:
                    poll_resp = requests.get(poll_url, headers={"x-freepik-api-key": api_key}, timeout=30)
                    if not poll_resp.ok:
                        continue
                    poll_data = poll_resp.json().get("data", {})
                    poll_status = poll_data.get("status")
                    poll_generated = poll_data.get("generated") or []
                    if poll_generated:
                        first_url = poll_generated[0]
                        try:
                            img_resp = requests.get(first_url)
                            img_resp.raise_for_status()
                            mime_type = img_resp.headers.get("Content-Type", "image/png")
                            image_data_base64 = base64.b64encode(img_resp.content).decode("utf-8")

                            # Save the image like other providers
                            timestamp = int(time.time() * 1000)
                            ext = mimetypes.guess_extension(mime_type) or ".png"
                            rel_path = f"artifacts/screens/image_{timestamp}{ext}"
                            full_path = _resolve_artifact_path(rel_path, ensure_dir=True)
                            with open(full_path, "wb") as f:
                                f.write(base64.b64decode(image_data_base64))
                            print(f"✅ Image saved to: {os.path.relpath(full_path, _find_project_root())}")

                            return f"data:{mime_type};base64,{image_data_base64}"
                        except Exception as dl_error:
                            return f"Freepik image URL generated but could not download: {dl_error}. URL: {first_url}"

                    if poll_status and str(poll_status).upper() in {"FAILED", "ERROR"}:
                        return f"Freepik task failed. status={poll_status} task_id={task_id}"
                except Exception:
                    continue

            return f"Freepik task still in progress after {max_wait_seconds}s. status={status} task_id={task_id}"
        except Exception as freepik_error:
            return f"An API error occurred during Freepik image generation: {freepik_error}"

    is_url = image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://')

    # Resolve local path relative to project root
    if not is_url:
        project_root = _find_project_root()
        resolved_path = os.path.join(project_root, image_path_or_url)
        if not os.path.exists(resolved_path):
            return f"Error: Local image file not found at {image_path_or_url}"
        image_path_or_url = resolved_path

    try:
        if api_provider == "openai":
            image_url_data = {}
            if is_url:
                image_url_data = {"url": image_path_or_url}
            else:
                base64_image = _encode_image_to_base64(image_path_or_url)
                image_url_data = {"url": base64_image}

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": image_url_data}
                        ]
                    }]
                )
                return response.choices[0].message.content
            except Exception as e:
                # Surface the provider error clearly
                return f"An API error occurred during vision completion: {e}"

        elif api_provider == "anthropic":
            if is_url:
                response_img = requests.get(image_path_or_url)
                response_img.raise_for_status()
                img_content = response_img.content
                mime_type = response_img.headers.get('Content-Type', 'image/jpeg')
            else:
                with open(image_path_or_url, "rb") as f:
                    img_content = f.read()
                mime_type, _ = mimetypes.guess_type(image_path_or_url)
                if not mime_type:
                    return f"Error: Could not determine mime type for {image_path_or_url}"

            img_data = base64.b64encode(img_content).decode('utf-8')
            
            response = client.messages.create(
                model=model_name,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": img_data}}
                    ]
                }]
            )
            return response.content[0].text

        elif api_provider == "gemini" or api_provider == "google":
            if is_url:
                response_img = requests.get(image_path_or_url)
                response_img.raise_for_status()
                img = Image.open(BytesIO(response_img.content))
            else:
                img = Image.open(image_path_or_url)
            
            response = client.generate_content([prompt, img])
            return response.text

        elif api_provider == "huggingface":
            if is_url:
                response_img = requests.get(image_path_or_url)
                response_img.raise_for_status()
                img = Image.open(BytesIO(response_img.content))
            else:
                img = Image.open(image_path_or_url)
                
            # Note: HuggingFace's image_to_text might not support a separate text prompt in all cases.
            # The prompt may need to be embedded in the task definition for some models.
            # This implementation assumes the client handles the combination correctly.
            response = client.image_to_text(image=img, prompt=prompt)
            return response

    except Exception as e:
        return f"An API error occurred during vision completion: {e}"

def get_image_generation_completion(prompt, client, model_name, api_provider):
    """
    Generates an image from a text prompt using an image generation LLM.
    
    This function provides a unified interface for text-to-image generation across
    different providers. It handles the API differences between providers and
    returns the generated image as a base64-encoded data URL that can be displayed
    directly in web browsers or Jupyter notebooks.
    
    Args:
        prompt (str): The text description of the image to generate. Should be
            detailed and specific for best results. Example: "A serene mountain
            landscape at sunset with a lake in the foreground".
        client: The initialized API client object from setup_llm_client().
        model_name (str): The identifier of the image generation model to use.
            Examples: "dall-e-3", "gemini-2.5-flash-image-preview".
        api_provider (str): The provider name ("openai", "google", or "apifree").
    
    Returns:
        tuple[str, str]: A tuple containing:
            - file_path (str): The local path to the saved image file.
            - image_url (str): A data URL string like "data:image/png;base64,{...}" suitable for HTML/Jupyter.
            On error returns (None, error_message).
    
    Raises:
        None: This function catches all exceptions and returns error messages
            as strings instead of raising exceptions.
    
    Notes:
        - Validates that the model supports image generation using RECOMMENDED_MODELS
        - Displays a loading indicator during generation (can take 10-30 seconds)
        - Tracks and reports generation time
        - Provider-specific handling:
            - OpenAI: Uses images.generate() API, returns base64 directly
            - Google Gemini: Uses generate_content or google-genai streaming for preview models
            - APIFREE: Uses REST API with APIFREE_API_KEY
        - The returned data URL can be used directly in HTML or markdown
        - Loading indicators are shown in console and Jupyter environments
    
    Example:
        >>> client, model, provider = setup_llm_client("dall-e-3")
        >>> file_path, image_url = get_image_generation_completion(
        ...     "A futuristic city with flying cars and neon lights",
        ...     client, model, provider
        ... )
        Generating image... This may take a moment.
        ⏳ Generating image...
        ✅ Image generated in 15.32 seconds.
        ✅ Image saved to: artifacts/screens/image_1662586800.png
        >>> # Display in Jupyter: display(Image(url=image_url))
        
        >>> # Error handling
        >>> response = get_image_generation_completion(
        ...     "A cat", client, "gpt-4o", "openai"
        ... )
        >>> print(response)
        (None, "Error: Model 'gpt-4o' does not support image generation.")
    
    Dependencies:
        - time: For tracking generation duration
        - IPython.display: For showing loading indicators in Jupyter
        - RECOMMENDED_MODELS: For image generation capability validation
    """
    if api_provider != "apifree" and not client: 
        return None, "API client not initialized."
        
    if not RECOMMENDED_MODELS.get(model_name, {}).get("image_generation"):
        return None, f"Error: Model '{model_name}' does not support image generation."

    # Display a loading indicator
    print("Generating image... This may take a moment.")
    display(Markdown("⏳ Generating image..."))
    start_time = time.time()

    try:
        image_data_base64 = None
        image_mime = None
        
        if api_provider == "openai":
            params = {
                "model": model_name,
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
            }
            # gpt-image-1 doesn't support b64_json response format, it returns a URL by default
            if model_name != 'gpt-image-1':
                params["response_format"] = "b64_json"

            response = client.images.generate(**params)

            if model_name == 'gpt-image-1' and response.data[0].url:
                # Download image from URL and convert to base64
                img_resp = requests.get(response.data[0].url)
                img_resp.raise_for_status()
                image_data_base64 = base64.b64encode(img_resp.content).decode('utf-8')
            else:
                image_data_base64 = response.data[0].b64_json
        elif api_provider == "deepseek":
            # DeepSeek uses OpenAI-compatible API, but has different image generation capabilities
            # DeepSeek's image generation is experimental; attempt the call
            try:
                response = client.images.generate(
                    model=model_name,
                    prompt=prompt,
                    n=1,
                    size="1024x1024",
                    response_format="b64_json"
                )
                image_data_base64 = response.data[0].b64_json
            except Exception as deepseek_error:
                # If image generation not available, provide helpful message
                error_msg = str(deepseek_error).lower()
                if "not available" in error_msg or "not supported" in error_msg or "unknown" in error_msg:
                    return None, f"DeepSeek's image generation capability may not be available yet. Error: {deepseek_error}"
                else:
                    raise deepseek_error
        elif api_provider == "huggingface":
            pil_image = client.text_to_image(prompt)
            
            # Convert PIL image to base64
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            image_data_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        elif api_provider == "apifree":
            api_key = os.getenv("APIFREE_API_KEY")
            if not api_key:
                return None, "Error: APIFREE_API_KEY not found in .env file."
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            base_url = os.getenv("APIFREE_API_BASE", "https://api.apifree.ai")
            
            # APIFREE uses async task submission and polling
            payload = {
                "model": model_name,
                "prompt": prompt,
                "num_images": 1,
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 50,
            }
            
            try:
                # Step 1: Submit the request
                submit_response = requests.post(
                    f"{base_url}/v1/image/submit",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                submit_response.raise_for_status()
                submit_data = submit_response.json()
                
                if submit_data.get("code") != 200:
                    return None, f"APIFREE submission failed: {submit_data.get('error', submit_data.get('code_msg'))}"
                
                request_id = submit_data.get("resp_data", {}).get("request_id")
                if not request_id:
                    return None, f"APIFREE did not return request_id: {submit_data}"
                
                # Step 2: Poll for result
                max_polls = 60
                poll_interval = 2
                for poll_count in range(max_polls):
                    time.sleep(poll_interval)
                    
                    check_response = requests.get(
                        f"{base_url}/v1/image/{request_id}/result",
                        headers=headers,
                        timeout=30,
                    )
                    check_response.raise_for_status()
                    check_data = check_response.json()
                    
                    if check_data.get("code") != 200:
                        return None, f"APIFREE check failed: {check_data.get('code_msg')}"
                    
                    status = check_data.get("resp_data", {}).get("status", "unknown")
                    
                    if status == "success":
                        image_list = check_data.get("resp_data", {}).get("image_list", [])
                        if not image_list:
                            return None, "APIFREE returned no images"
                        
                        # Download the first image
                        img_url = image_list[0]
                        img_response = requests.get(img_url, timeout=30)
                        img_response.raise_for_status()
                        image_data_base64 = base64.b64encode(img_response.content).decode('utf-8')
                        break
                        
                    elif status == "error" or status == "failed":
                        error_msg = check_data.get("resp_data", {}).get("error", "Unknown error")
                        return None, f"APIFREE task failed: {error_msg}"
                    
                    # Still processing, continue polling
                    if (poll_count + 1) % 5 == 0:
                        print(f"Still generating image... (poll {poll_count + 1}/{max_polls})")
                
                if poll_count >= max_polls - 1:
                    return None, "APIFREE task timed out after 2 minutes"
                    
            except requests.exceptions.RequestException as e:
                return None, f"APIFREE API error: {e}"

        elif api_provider == "google":
            if "imagen" in model_name:
                # Google Imagen via google-genai Client
                try:
                    from google import genai as google_genai
                    from google.genai import types as google_types
                except ImportError:
                    return None, "google-genai package not installed. Run: pip install google-genai"

                # Prefer provided client if it's a google-genai Client; else create one
                gg_client = None
                if client is not None and hasattr(client, "models") and hasattr(client.models, "generate_images"):
                    gg_client = client
                else:
                    api_key = os.environ.get("GOOGLE_API_KEY")
                    if not api_key:
                        return None, "GOOGLE_API_KEY not found in environment. Please set it to use Google Imagen."
                    gg_client = google_genai.Client(api_key=api_key)

                try:
                    resp = gg_client.models.generate_images(
                        model=model_name,
                        prompt=prompt,
                        config=google_types.GenerateImagesConfig(number_of_images=1),
                    )
                except Exception as model_error:
                    return None, f"Google Imagen generation failed: {model_error}"

                # Extract first image as PNG bytes
                img_obj = None
                if hasattr(resp, "generated_images") and resp.generated_images:
                    first = resp.generated_images[0]
                    img_obj = getattr(first, "image", None)

                if img_obj is None:
                    return None, "Google Imagen returned no image data."

                try:
                    buf = BytesIO()
                    # PIL Image object expected
                    img_obj.save(buf, format="PNG")
                    image_data_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    image_mime = "image/png"
                except Exception as encode_error:
                    return None, f"Failed to serialize Imagen output: {encode_error}"

            elif "gemini" in model_name:
                # Prefer the new google-genai client for Gemini image preview models
                if model_name in ("gemini-2.5-flash-image-preview", "gemini-2.0-flash-preview-image-generation"):
                    try:
                        # Lazy-import to avoid hard dependency if user doesn't use this path
                        from google import genai as google_genai
                        
                        # Use GOOGLE_API_KEY for Gemini image preview
                        api_key = os.environ.get("GOOGLE_API_KEY")
                        if not api_key:
                            return None, "GOOGLE_API_KEY not found in environment. Please set it to use Gemini image preview."

                        gg_client = google_genai.Client(api_key=api_key)

                        # Build request using plain dicts with correct shapes.
                        # parts items must be {"text": ...}, no extra "type" key.
                        contents = [{
                            "role": "user",
                            "parts": [{"text": prompt}],
                        }]

                        generate_content_config = {"response_modalities": ["IMAGE", "TEXT"]}

                        saved_img_bytes = None
                        saved_mime = None

                        for chunk in gg_client.models.generate_content_stream(
                            model=model_name,
                            contents=contents,
                            config=generate_content_config,
                        ):
                            # Validate structure defensively
                            if (
                                not getattr(chunk, "candidates", None)
                                or not chunk.candidates
                                or not getattr(chunk.candidates[0], "content", None)
                                or not getattr(chunk.candidates[0].content, "parts", None)
                            ):
                                continue

                            part0 = chunk.candidates[0].content.parts[0]
                            # Image bytes come via inline_data
                            if getattr(part0, "inline_data", None) and getattr(part0.inline_data, "data", None):
                                saved_img_bytes = part0.inline_data.data
                                saved_mime = getattr(part0.inline_data, "mime_type", None) or "image/png"
                            # Occasionally the stream also includes text; surface it for debugging
                            elif getattr(part0, "text", None):
                                print(part0.text)

                        if not saved_img_bytes:
                            return None, "Gemini image preview returned no image data."

                        image_data_base64 = base64.b64encode(saved_img_bytes).decode("utf-8")
                        image_mime = saved_mime

                    except ImportError:
                        return None, "google-genai package not installed. Run: pip install google-genai"
                    except Exception as model_error:
                        return None, f"Gemini image generation (preview) failed: {model_error}"
                else:
                    # Fallback for other Gemini models via google-generativeai GenerativeModel
                    try:
                        response = client.generate_content(prompt)

                        if response and hasattr(response, 'parts') and response.parts:
                            part = response.parts[0]

                            if hasattr(part, 'inline_data') and part.inline_data and hasattr(part.inline_data, 'data') and part.inline_data.data:
                                img_bytes = part.inline_data.data
                                image_data_base64 = base64.b64encode(img_bytes).decode('utf-8')
                            elif hasattr(part, 'text'):
                                text_response = part.text
                                return None, f"The model '{model_name}' generated text instead of image data. Try 'gemini-2.5-flash-image-preview' or 'dall-e-3'. Description: {text_response[:200]}..."
                            else:
                                return None, "Gemini response contained no usable image data or text."
                        else:
                            return None, "Invalid or empty response from Gemini."

                    except Exception as model_error:
                        return None, f"Gemini image generation failed: {model_error}"

        if not image_data_base64:
            return None, "Image generation failed or returned no data."

        # Save and display the image
        duration = time.time() - start_time
        print(f"✅ Image generated in {duration:.2f} seconds.")
        
        image_bytes = base64.b64decode(image_data_base64)
        
        # Create a unique filename
        timestamp = int(time.time() * 1000)
        # Prefer mime-derived extension when available
        ext = ".png"
        mime_for_url = "image/png"
        if image_mime:
            guessed = mimetypes.guess_extension(image_mime)
            if guessed:
                ext = guessed
                mime_for_url = image_mime
        rel_path = f"artifacts/screens/image_{timestamp}{ext}"
        full_path = _resolve_artifact_path(rel_path, ensure_dir=True)

        with open(full_path, "wb") as f:
            f.write(image_bytes)
        print(f"✅ Image saved to: {os.path.relpath(full_path, _find_project_root())}")
        
        # Return the data URL using the appropriate mime type
        return full_path, f"data:{mime_for_url};base64,{image_data_base64}"

    except Exception as e:
        return None, f"An API error occurred during image generation: {e}"

def get_image_edit_completion(
    prompt: str,
    image_path: str,
    client,
    model_name: str,
    api_provider: str,
    **edit_params,
):
    """
    Edits an existing image based on a text prompt using an image-to-image model.

    This function provides a unified interface for image editing across different
    providers. It takes a local image path and a prompt, sends them to the model,
    and returns the edited image as a base64-encoded data URL.

    Args:
        prompt (str): The text description of the edits to apply to the image.
        image_path (str): The local file path to the image to be edited.
        client: The initialized API client object from setup_llm_client().
        model_name (str): The identifier of the image editing model to use.
        api_provider (str): The provider name (e.g., "huggingface").

    Returns:
        tuple[str, str]: A tuple containing:
            - file_path (str): The local path to the saved edited image file.
            - image_url (str): A data URL string for the edited image.
            Returns (None, None) if an error occurs.

    Raises:
        None: This function catches all exceptions and returns error messages
            as strings instead of raising exceptions.

    Notes:
        - Validates that the model supports image editing using RECOMMENDED_MODELS
        - Displays a loading indicator during editing (can take 5-15 seconds)
        - Tracks and reports editing time
        - Provider-specific handling:
            - Hugging Face: Uses image_to_image() method
        - Supported editing models: "Qwen/Qwen-Image-Edit" and
          "black-forest-labs/FLUX.1-Kontext-dev".
        - The returned data URL can be used directly in HTML or markdown
        - Loading indicators are shown in console and Jupyter environments

    Example:
        >>> client, model, provider = setup_llm_client("Qwen/Qwen-Image-Edit")
        >>> file_path, image_url = get_image_edit_completion(
        ...     "Add a sunset in the background",
        ...     "artifacts/screens/image_1662586800.png",
        ...     client, model, provider
        ... )
        Editing image... This may take a moment.
        ⏳ Editing image...
        ✅ Image edited in 7.45 seconds.
        ✅ Edited image saved to: artifacts/screens/image_edited_1662586800.png
        >>> # Display in Jupyter: display(Image(url=image_url))

    Dependencies:
        - time: For tracking editing duration
        - base64: For encoding images
        - requests: For downloading images from URLs (if needed)
        - PIL (Pillow): For image processing
        - io.BytesIO: For converting image bytes to PIL Images
        - RECOMMENDED_MODELS: For image editing capability validation
    """
    if not client:
        return None, "API client not initialized."

    # Validate using the correct capability flag for editing
    if not RECOMMENDED_MODELS.get(model_name, {}).get("image_modification"):
        return None, f"Error: Model '{model_name}' does not support image editing."

    if not os.path.exists(image_path):
        return None, f"Error: Local image file not found at {image_path}"

    # Display a loading indicator
    print("Editing image... This may take a moment.")
    display(Markdown("⏳ Editing image..."))
    start_time = time.time()

    try:
        image_data_base64 = None

        if api_provider == "huggingface":
            # image_to_image expects a PIL image
            pil_image = Image.open(image_path)

            # Call the image-to-image endpoint
            try:
                # Pass through model-specific params when supported
                edited_image = client.image_to_image(
                    image=pil_image,
                    prompt=prompt,
                    **(edit_params or {})
                )
            except TypeError:
                # Some backends may not accept extra params; retry minimal
                edited_image = client.image_to_image(image=pil_image, prompt=prompt)

            # Convert the returned PIL image to base64
            buffered = BytesIO()
            edited_image.save(buffered, format="PNG")
            image_data_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            return None, f"Provider '{api_provider}' is not supported for image editing yet."

        if not image_data_base64:
            return None, "Image editing failed or returned no data."

        # Save and display the image
        duration = time.time() - start_time
        print(f"✅ Image edited in {duration:.2f} seconds.")

        image_bytes = base64.b64decode(image_data_base64)

        # Create a unique filename
        timestamp = int(time.time() * 1000)
        rel_path = f"artifacts/screens/image_edited_{timestamp}.png"
        full_path = _resolve_artifact_path(rel_path, ensure_dir=True)

        with open(full_path, "wb") as f:
            f.write(image_bytes)
        print(f"✅ Edited image saved to: {os.path.relpath(full_path, _find_project_root())}")

        # Return the data URL
        return full_path, f"data:image/png;base64,{image_data_base64}"

    except Exception as e:
        if "Task 'image-to-image' not supported" in str(e):
            return None, f"The model '{model_name}' does not support the image-to-image task through the current API provider. This is a limitation of the backend service, not the model itself."
        return None, f"An API error occurred during image editing: {e}"

def transcribe_audio(audio_path, client, model_name, api_provider, language_code="en-US"):
    """
    Transcribes an audio file to text using a speech-to-text model.
    
    This function provides a unified interface for audio transcription across
    different providers. It handles the API differences and returns the
    transcribed text.
    
    Args:
        audio_path (str): The local file path to the audio file to transcribe.
        client: The initialized API client object from setup_llm_client().
        model_name (str): The identifier of the audio transcription model to use.
        api_provider (str): The provider name ("openai" or "google").
        language_code (str, optional): The language of the audio. Defaults to "en-US".
    
    Returns:
        str: The transcribed text. Returns an error message string if the
            API call fails.
    
    Raises:
        None: This function catches all exceptions and returns error messages
            as strings instead of raising exceptions.
    
    Notes:
        - Validates that the model supports audio transcription
        - Handles different API structures for each provider
        - OpenAI: Uses the audio.transcriptions.create() method
        - Google: Uses the recognize() method with a RecognitionConfig
    
    Example:
        >>> client, model, provider = setup_llm_client("whisper-1")
        >>> text = transcribe_audio("path/to/audio.wav", client, model, provider)
        >>> print(text)
        "This is the transcribed text from the audio file."
    
    Dependencies:
        - Provider-specific client libraries
        - RECOMMENDED_MODELS: For audio transcription capability validation
    """
    if not client:
        return "API client not initialized."
        
    if not RECOMMENDED_MODELS.get(model_name, {}).get("audio_transcription"):
        return f"Error: Model '{model_name}' does not support audio transcription."

    if not os.path.exists(audio_path):
        return f"Error: Audio file not found at {audio_path}"

    try:
        if api_provider == "openai":
            with open(audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model=model_name,
                    file=audio_file
                )
            return transcription.text
        elif api_provider == "google":
            with open(audio_path, "rb") as audio_file:
                content = audio_file.read()

            audio = {"content": content}
            config = {
                "language_code": language_code,
            }
            response = client.recognize(config=config, audio=audio)
            
            if response.results:
                return response.results[0].alternatives[0].transcript
            else:
                return "No transcription result from Google Speech-to-Text."

    except Exception as e:
        return f"An API error occurred during audio transcription: {e}"

def clean_llm_output(output_str: str, language: str = 'json') -> str:
    """
    Cleans markdown code fences from LLM output.
    Supports various languages.
    """
    if output_str is None:
        return ""
    if not isinstance(output_str, str):
        try:
            return json.dumps(output_str, indent=2)
        except Exception:
            return str(output_str)
    if '```' in output_str:
        # Regex to find code blocks with optional language specifier
        # It looks for ```[language_optional]\n[content]\n```
        pattern = re.compile(r'```(?:' + re.escape(language) + r')?\s*\n(.*?)\n```', re.DOTALL | re.IGNORECASE)
        match = pattern.search(output_str)
        if match:
            return match.group(1).strip()
        else:
            # Fallback if regex doesn't match perfectly (e.g., no language specified, or extra text)
            parts = output_str.split('```')
            if len(parts) >= 3:
                # Take the content between the first and second ```
                return parts[1].strip()
            else:
                # If only one ``` or malformed, return original string
                return output_str.strip()
    return output_str.strip()


def prompt_enhancer(user_input, model_name="o3", client=None, api_provider=None):
    """
    Enhances a raw user prompt using a meta-prompt optimization system.
    
    This function takes a user's raw input and passes it through an elite 
    Prompt Optimization Engine that applies advanced prompt engineering 
    techniques to create a high-quality, optimized prompt suitable for 
    use with other utils.py functions.
    
    The enhancement process includes:
    - Role assignment and persona definition
    - Context provision and grounding
    - Task definition with clarity
    - Output format specification
    - Chain-of-Thought integration for complex reasoning
    - In-Context Learning examples when needed
    - Structural integrity with clear delimiters
    
    Args:
        user_input (str): The raw user prompt or request to be enhanced.
        model_name (str, optional): The model to use for enhancement. 
            Defaults to "gpt-4o". Should be a key from RECOMMENDED_MODELS.
        client (optional): Pre-initialized LLM client. If provided, this client
            will be used instead of creating a new one.
        api_provider (str, optional): The API provider name. Required if client
            is provided.
    
    Returns:
        str: An enhanced, optimized prompt ready for use with other LLM functions.
             Returns the original input with an error message if enhancement fails.
    
    Example:
        >>> enhanced = prompt_enhancer("Write code for a login system")
        >>> client, model, provider = setup_llm_client("gpt-4o")
        >>> result = get_completion(enhanced, client, model, provider)
        
        >>> # Using existing client to avoid duplicate setup
        >>> enhanced = prompt_enhancer("Write code", "gpt-4o", client, provider)
    
    Dependencies:
        - setup_llm_client(): For initializing the LLM client (if not provided)
        - get_completion(): For getting the enhanced prompt from the LLM
        - RECOMMENDED_MODELS: For model validation
    """
    if not user_input or not user_input.strip():
        return "Error: No user input provided for enhancement."
    
    if model_name not in RECOMMENDED_MODELS:
        return f"Error: Model '{model_name}' not found in RECOMMENDED_MODELS. Original input: {user_input}"
    
    # The meta-prompt for prompt optimization
    optimization_prompt = f"""You are an elite Prompt Optimization Engine. Your design is based on the understanding that prompt engineering is a rigorous technical discipline, essential for maximizing LLM efficacy and reliability. Your function is to analyze raw user inputs and systematically compile them into optimized, high-quality prompts.

**User Input:**
<user_input>
{user_input}
</user_input>

**Optimization Protocol:**
Follow this systematic protocol to analyze the user input and construct the optimized prompt.

### Phase 1: Analysis and Strategy Determination
1.  **Analyze Intent and Complexity:** Deconstruct the user's input to identify the core objective. Assess the complexity: Does it require simple retrieval, creative generation, or complex, multi-step reasoning?
2.  **Determine Strategic Enhancements:**
    *   **Chain-of-Thought (CoT):** If the task involves complex reasoning, analysis, or multi-step problem-solving, you must incorporate CoT prompting (e.g., instructing the model to "think step by step").
    *   **In-Context Learning (ICL):** If the task requires a highly specific output format (e.g., structured data) or involves nuanced pattern recognition, generate 1-2 relevant input/output examples (Few-Shot prompting) to guide the model.

### Phase 2: Prompt Construction and Enhancement
Construct the optimized prompt by ensuring the following components are explicitly defined and integrated:

1.  **Role Assignment (Persona):**
    *   Define the most authoritative expert persona for the LLM to adopt (e.g., "You are a Senior Cybersecurity Analyst," "Act as an expert Python developer"). This constrains the knowledge space for improved accuracy and focus.

2.  **Context Provision and Grounding:**
    *   Provide comprehensive background information, define key terms unambiguously, and state all constraints or rules. Ensure the model has sufficient information to ground its response in a relevant factual basis.

3.  **Task Definition and Clarity:**
    *   Use precise, unambiguous instructions and assertive action verbs (e.g., "Analyze," "Synthesize," "Generate").
    *   Decompose the main objective into a clear sequence of steps if necessary.

4.  **Expectation Setting (Output Specification):**
    *   Explicitly define the desired output format (e.g., Markdown report, JSON object, bulleted list), length constraints, style, and target audience.

### Phase 3: Structural Integrity
Organize the entire prompt using clear structural delimiters to ensure optimal parsing by the target LLM. Clearly differentiate between instructions, context, examples, and the core task (e.g., using XML tags such as `<persona>`, `<context>`, `<instructions>`, `<examples>`, `<output_format>`).

### Output
Generate only the final, optimized prompt."""

    try:
        # Use provided client or set up a new one
        if client and api_provider:
            actual_model = model_name
            provider = api_provider
        else:
            # Set up the LLM client for enhancement
            client, actual_model, provider = setup_llm_client(model_name)
            
            if not client:
                return f"Error: Failed to initialize LLM client for model '{model_name}'. Original input: {user_input}"
        
        # Get the enhanced prompt with low temperature for consistency
        enhanced_prompt = get_completion(
            optimization_prompt, 
            client, 
            actual_model, 
            provider, 
            temperature=0.3
        )
        
        if not enhanced_prompt or enhanced_prompt.startswith("Error:") or enhanced_prompt.startswith("API client not initialized"):
            return f"Error: Failed to enhance prompt. Original input: {user_input}"
        
        return enhanced_prompt.strip()
        
    except Exception as e:
        return f"Error during prompt enhancement: {str(e)}. Original input: {user_input}"


# --- Artifact Management & Display ---

def _find_project_root():
    """
    Finds the project root by searching upwards for a known directory marker
    (like '.git' or 'artifacts'). This is more reliable than just using os.getcwd().
    """
    path = os.getcwd()
    while path != os.path.dirname(path):  # Stop at the filesystem root
        # Check for multiple common markers to increase reliability
        if any(os.path.exists(os.path.join(path, marker)) for marker in ['.git', 'artifacts', 'README.md']):
            return path
        path = os.path.dirname(path)
    # Fallback if no markers are found (e.g., in a bare directory)
    print("Warning: Project root marker not found. Defaulting to current directory.")
    return os.getcwd()


def _resolve_artifact_path(file_path: str, *, ensure_dir: bool = False) -> str:
    """
    Resolve a given file path to an absolute path under the project `artifacts/` directory.

    Accepts either paths beginning with `artifacts/` or paths relative to the
    artifacts directory (e.g., `screens/img.png`). Ensures that the resolved path
    cannot escape the artifacts directory (guards against path traversal).

    Args:
        file_path: A relative path like `artifacts/foo.txt` or `logs/run.txt`.
        ensure_dir: If True, creates the parent directory of the resolved path.

    Returns:
        The absolute, normalized path within the artifacts directory.

    Raises:
        ValueError: If the resolved path is outside the artifacts directory or if
            an absolute path is provided outside of artifacts.
    """
    project_root = _find_project_root()
    artifacts_root = os.path.realpath(os.path.join(project_root, 'artifacts'))

    # Normalize incoming path: allow with or without leading 'artifacts/'
    if os.path.isabs(file_path):
        candidate = os.path.realpath(file_path)
    else:
        if file_path.startswith('artifacts' + os.sep) or file_path == 'artifacts':
            candidate = os.path.realpath(os.path.join(project_root, file_path))
        else:
            candidate = os.path.realpath(os.path.join(artifacts_root, file_path))

    # Ensure candidate is within artifacts_root
    if candidate != artifacts_root and not candidate.startswith(artifacts_root + os.sep):
        raise ValueError(f"Path '{file_path}' is outside the artifacts directory.")

    if ensure_dir:
        os.makedirs(os.path.dirname(candidate), exist_ok=True)
    return candidate


def save_artifact(content, file_path):
    """
    Save content under the project `artifacts/` folder only.

    Accepts paths with or without the `artifacts/` prefix. Any attempt to save
    outside `artifacts/` raises ValueError. Creates parent directories as needed.
    """
    try:
        full_path = _resolve_artifact_path(file_path, ensure_dir=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        # Print relative to project root for readability
        rel = os.path.relpath(full_path, _find_project_root())
        print(f"✅ Successfully saved artifact to: {rel}")
    except Exception as e:
        print(f"❌ Error saving artifact to {file_path}: {e}")
        raise


def load_artifact(file_path):
    """
    Load content strictly from the project `artifacts/` folder.

    Accepts paths with or without the `artifacts/` prefix. Any attempt to read
    outside `artifacts/` raises ValueError.
    """
    try:
        full_path = _resolve_artifact_path(file_path, ensure_dir=False)
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ Error: Artifact file not found at {file_path}.")
        return None

def render_plantuml_diagram(puml_code, output_path="artifacts/diagram.png"):
    """
    Renders PlantUML code into a PNG image and displays it in Jupyter environments.
    
    This function takes PlantUML markup code and converts it into a visual diagram
    using the PlantUML web service. The generated image is saved to the specified
    path within the project and automatically displayed in Jupyter notebooks.
    
    Args:
        puml_code (str): The PlantUML markup code to render. Should be valid PlantUML
            syntax (e.g., "@startuml\\nclass Example\\n@enduml").
        output_path (str, optional): Relative path from project root where the PNG
            image will be saved. Defaults to "artifacts/diagram.png". Directory
            structure will be created automatically if it doesn't exist.
    
    Returns:
        None: This function doesn't return a value but produces side effects:
            - Saves PNG image to the specified file path
            - Prints status messages to console
            - Displays the image in Jupyter environments
    
    Raises:
        Exception: Catches and reports any errors during the rendering process,
            including network errors, file system errors, or PlantUML syntax errors.
    
    Notes:
        - Uses the public PlantUML web service (http://www.plantuml.com/plantuml/img/)
        - Handles different versions of the plantuml library automatically
        - Creates output directories as needed using os.makedirs
        - Falls back gracefully if image display fails in non-Jupyter environments
        - Supports both direct file writing and URL-based image fetching
    
    Example:
        >>> puml_code = '''
        ... @startuml
        ... class User {
        ...     +name: string
        ...     +email: string
        ...     +login()
        ... }
        ... @enduml
        ... '''
        >>> render_plantuml_diagram(puml_code, "diagrams/user_class.png")
        ✅ Diagram rendered and saved to: diagrams/user_class.png
    
    Dependencies:
        - plantuml: Python library for PlantUML integration
        - requests: For HTTP requests when fetching images from URLs
        - PIL (Pillow): Used internally by plantuml library
        - IPython.display: For displaying images in Jupyter notebooks (optional)
    """
    try:
        # Route output through strict artifacts resolver
        full_path = _resolve_artifact_path(output_path, ensure_dir=True)

        # FIX: Corrected the PlantUML URL
        pl = PlantUML(url='http://www.plantuml.com/plantuml/img/')
        # plantuml library versions differ in their `processes` signature.
        # Some accept an `outfile` kwarg, others return the image data or a URL.
        result = None
        try:
            # Preferred: try calling with outfile (some versions support this)
            result = pl.processes(puml_code, outfile=full_path)
        except TypeError:
            # Fallback: call without outfile and handle returned data/result.
            result = pl.processes(puml_code)

        # If the library returned raw bytes, save them to the file.
        if isinstance(result, (bytes, bytearray)):
            with open(full_path, 'wb') as f:
                f.write(result)
        # If the library returned a URL string, fetch it and save the image bytes.
        elif isinstance(result, str) and result.startswith('http'):
            try:
                resp = requests.get(result)
                resp.raise_for_status()
                with open(full_path, 'wb') as f:
                    f.write(resp.content)
            except Exception:
                # If fetching the URL fails, still continue to let callers know result.
                pass

        # At this point, the plantuml lib may have already written the file
        # or we wrote it above. Check for file existence before displaying.
        if os.path.exists(full_path):
            rel = os.path.relpath(full_path, _find_project_root())
            print(f"✅ Diagram rendered and saved to: {rel}")
            try:
                # IPython Image accepts filename= or url=. Use filename for local file.
                display(IPyImage(filename=full_path))
            except Exception:
                # Best-effort fallback to markdown link if display fails.
                display(Markdown(f"![diagram]({full_path})"))
        else:
            print(f"⚠️ Diagram rendering returned no file. Result: {result}")
    except Exception as e:
        print(f"❌ Error rendering PlantUML diagram: {e}")
        raise

def _encode_image_to_base64(image_path):
    """Encodes a local image file to a base64 data URL."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image'):
        raise ValueError(f"Cannot determine image type for {image_path}")
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"

def get_edit_completion(prompt, image_path_or_url, client, model_name, api_provider, width=1024, height=1024):
    """
    Sends an image and a text prompt to an image editing model and returns the edited image(s).
    
    This function enables image editing by processing a text prompt describing the desired edits
    and an input image. It handles the APIFREE image editing workflow including task submission,
    polling for results, and image download.
    
    Args:
        prompt (str): Text description of the edits to apply to the image.
        image_path_or_url (str): URL of the image to edit or a local file path.
            If a URL, it must be publicly accessible.
        client: The initialized API client object (currently unused for image editing).
        model_name (str): The identifier of the image editing model to use (e.g., "qwen/qwen-image-edit-2511").
        api_provider (str): The provider name (currently supports "apifree").
        width (int): Output image width in pixels (default: 1024).
        height (int): Output image height in pixels (default: 1024).
    
    Returns:
        list or dict: The edited image result(s). Returns error message string if the API call fails.
            For successful operations, returns the resp_data containing image_list and other metadata.
    
    Notes:
        - Currently supports APIFREE provider with qwen/qwen-image-edit-2511 model
        - Uses the /v1/image/submit and /v1/image/{request_id}/result endpoints
        - Polls for completion with configurable timeout
        - Downloads resulting images to artifacts/screens/ directory
        - Requires APIFREE_API_KEY in environment or .env file
    
    Example:
        >>> client, model, provider = setup_llm_client("qwen/qwen-image-edit-2511")
        >>> result = get_edit_completion(
        ...     prompt="Add a red hat to the person",
        ...     image_path_or_url="artifacts/screens/photo.jpg",
        ...     client=client,
        ...     model_name="qwen/qwen-image-edit-2511",
        ...     api_provider="apifree",
        ...     width=1024,
        ...     height=1024
        ... )
    """
    if api_provider != "apifree":
        return f"Error: Image editing currently only supports 'apifree' provider, got '{api_provider}'"
    
    api_key = os.getenv("APIFREE_API_KEY")
    if not api_key:
        return "Error: APIFREE_API_KEY not found in .env file."
    
    if not image_path_or_url:
        return "Error: image_path_or_url is required for image editing requests."
    
    if not prompt:
        return "Error: prompt is required for image editing requests."
    
    # Handle URLs and local file paths
    is_url = image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://")
    
    if is_url:
        image_urls = [image_path_or_url]
    else:
        project_root = _find_project_root()
        resolved_path = os.path.join(project_root, image_path_or_url)
        if not os.path.exists(resolved_path):
            return f"Error: Local image file not found at {image_path_or_url}"
        
        # For local files, we need to provide them as URLs or data URLs
        # Read and convert to data URL
        with open(resolved_path, "rb") as f:
            img_content = f.read()
        mime_type = mimetypes.guess_type(resolved_path)[0] or "image/png"
        data_url = f"data:{mime_type};base64,{base64.b64encode(img_content).decode('utf-8')}"
        image_urls = [data_url]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    base_url = os.getenv("APIFREE_API_BASE", "https://api.apifree.ai")
    
    try:
        # 1. Submit the image editing request
        payload = {
            "model": model_name,
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": 1,
            "num_inference_steps": 28,
            "width": width,
            "height": height,
        }
        
        submit_url = f"{base_url}/v1/image/submit"
        resp = requests.post(submit_url, headers=headers, json=payload, timeout=120)
        
        if not resp.ok:
            return f"APIFREE error: {resp.status_code} {resp.text}"
        
        data = resp.json()
        if data.get("code") != 200:
            return f"APIFREE error: {data.get('error') or data}"
        
        request_id = data.get("resp_data", {}).get("request_id")
        if not request_id:
            return f"APIFREE error: Missing request_id. Response: {data}"
        
        print(f"Task submitted. Request ID: {request_id}")
        
        # 2. Poll for completion
        result_url = f"{base_url}/v1/image/{request_id}/result"
        max_polls = 60
        poll_count = 0
        
        while poll_count < max_polls:
            time.sleep(2)
            poll_count += 1
            
            check_resp = requests.get(result_url, headers=headers, timeout=60)
            if not check_resp.ok:
                return f"APIFREE error: {check_resp.status_code} {check_resp.text}"
            
            check_data = check_resp.json()
            if check_data.get("code") != 200:
                return f"APIFREE error: {check_data.get('code_msg') or check_data}"
            
            status = check_data.get("resp_data", {}).get("status")
            print(f"Status: {status}. Waiting...")
            
            if status == "success":
                resp_data = check_data.get("resp_data", {})
                image_list = resp_data.get("image_list", [])
                
                if image_list:
                    # Download and save the images
                    for i, img_url in enumerate(image_list):
                        try:
                            img_resp = requests.get(img_url, timeout=60)
                            img_resp.raise_for_status()
                            
                            # Save to artifacts directory
                            timestamp = int(time.time() * 1000)
                            filename = f"image_edited_{request_id}_{i+1}.png"
                            rel_path = f"artifacts/screens/{filename}"
                            full_path = _resolve_artifact_path(rel_path, ensure_dir=True)
                            
                            with open(full_path, "wb") as f:
                                f.write(img_resp.content)
                            
                            print(f"✅ Image saved to: {os.path.relpath(full_path, _find_project_root())}")
                        except Exception as dl_error:
                            print(f"Warning: Could not download image {i+1}: {dl_error}")
                
                print("✅ Generation completed!")
                return resp_data
            
            elif status in {"error", "failed"}:
                error_msg = check_data.get("resp_data", {}).get("error", "Unknown error")
                return f"APIFREE error: Task failed with status '{status}': {error_msg}"
        
        return f"APIFREE error: Timed out waiting for result after {max_polls} polls"
    
    except Exception as e:
        return f"APIFREE request failed: {e}"
