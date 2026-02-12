from google import genai

# --- Gemini API Configuration --- Don't share #########################################################
API_KEY = 'AIzaSyDn9pDuQrolbxHrK28gR9qLxnPzPb8yc7I' #os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
client = genai.Client(api_key=API_KEY)

i = 0
# get all the files
for i, file in enumerate(client.files.list()):
    # delete each file
    client.files.delete(name=file.name)

print(f"deleted {i} files")
