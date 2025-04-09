import os
from dotenv import load_dotenv

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path of the .env file in the same directory as the script
env_path = os.path.join(script_dir, '.env')
print(f"Path to .env file: {env_path}")

# Load the environment variables from the .env file
load_dotenv(env_path)

# Also print the current working directory for reference
print(f"Current working directory: {os.getcwd()}")

# Now try to get the environment variable after loading the .env file
x = os.environ.get("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY: {x}")