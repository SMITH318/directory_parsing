from google import genai
import os

# --- Gemini API Configuration --- 
API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
if API_KEY == 'YOUR_API_KEY' or not API_KEY:
    print("ERROR: Gemini API key is not set.")
    exit(1)

client = genai.Client(api_key=API_KEY)

i = 0
# get all the files
for i, file in enumerate(client.files.list()):
    # delete each file
    client.files.delete(name=file.name)

print(f"deleted {i} files")

client.close()