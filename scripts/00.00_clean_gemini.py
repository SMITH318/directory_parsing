from google import genai
import os

# --- Gemini API Configuration --- 
API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
if API_KEY == 'YOUR_API_KEY' or not API_KEY:
    print("ERROR: Gemini API key is not set.")
    exit(1)

client = None

def remove_all_uploaded_files(close_created_client=True):
    created_client = False
    if not client:
        created_client = True
        client = genai.Client(api_key=API_KEY)

    i = 0
    # get all the files
    for i, file in enumerate(client.files.list()):
        # delete each file
        client.files.delete(name=file.name)
    print(f"deleted {i} files")

    if close_created_client and created_client:
        client.close()

def remove_all_batches(close_created_client=True):
    created_client = False
    if not client:
        created_client = True
        client = genai.Client(api_key=API_KEY)

    i = 0
    # get all the jobs
    for i, job in enumerate(client.batches.list()):
        # delete each job
        client.batches.delete(name=job.name)
    print(f"deleted {i} batches")

    if close_created_client and created_client:
        client.close()

def remove_all_caches(close_created_client=True):
    created_client = False
    if not client:
        created_client = True
        client = genai.Client(api_key=API_KEY)

    i = 0
    # get all the caches
    for i, cache in enumerate(client.caches.list()):
        # delete each cache
        client.caches.delete(name=cache.name)
    print(f"deleted {i} batches")

    if close_created_client and created_client:
        client.close()

if __name__ == "__main__":
    remove_all_uploaded_files(False)
    remove_all_batches()
    remove_all_caches()
    client.close()