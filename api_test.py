from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()
token = os.environ.get("HF_TOKEN")

api = HfApi()
print(api.whoami(token=token))
