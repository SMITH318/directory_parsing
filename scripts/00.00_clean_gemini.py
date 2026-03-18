from google import genai
import os

def remove_all_uploaded_files(client):
    i = 0
    # get all the files
    for i, file in enumerate(client.files.list()):
        # delete each file
        client.files.delete(name=file.name)
    print(f"deleted {i} files")

def remove_all_batches(client):
    i = 0
    # get all the jobs
    for i, job in enumerate(client.batches.list()):
        # delete each job
        client.batches.delete(name=job.name)
    print(f"deleted {i} batches")

def remove_all_caches(client):
    i = 0
    # get all the caches
    for i, cache in enumerate(client.caches.list()):
        # delete each cache
        client.caches.delete(name=cache.name)
    print(f"deleted {i} batches")


if __name__ == "__main__":
    # --- Gemini API Configuration --- 
    API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
    if API_KEY == 'YOUR_API_KEY' or not API_KEY:
        print("ERROR: Gemini API key is not set.")
        exit(1)

    client = genai.Client(api_key=API_KEY)
    remove_all_uploaded_files(client)
    remove_all_batches(client)
    remove_all_caches(client)
    client.close()