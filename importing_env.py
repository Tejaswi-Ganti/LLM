import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_project = os.getenv("LANGCHAIN_PROJECT")

# Check if the environment variables are loaded correctly
if langchain_api_key is None:
    raise ValueError("LANGCHAIN_API_KEY environment variable is not set or could not be found.")

if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set or could not be found.")

if langchain_project is None:
    raise ValueError("LANGCHAIN_PROJECT environment variable is not set or could not be found.")

# Print to verify (optional)
print(f"LANGCHAIN_API_KEY: {langchain_api_key}")
print(f"OPENAI_API_KEY: {openai_api_key}")
print(f"LANGCHAIN_PROJECT: {langchain_project}")

# Set environment variables
os.environ['LANGCHAIN_API_KEY'] = langchain_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ['LANGCHAIN_PROJECT'] = langchain_project
