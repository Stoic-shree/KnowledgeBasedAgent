import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("⚠️ OPENROUTER_API_KEY not found in .env file!")
else:
    # Placeholder: OpenRouter does not provide a direct list_models via the OpenAI client.
    # Users can refer to https://openrouter.ai/models for available models.
    print("OpenRouter API key is set. To view available models, visit https://openrouter.ai/models")
