import json
import csv

# Load the JSON file
with open('data/01_preprocessed/all_metadata.json', 'r') as f:
    data = json.load(f)

# Open CSV file for writing
with open('data/01_preprocessed/all_metadata.csv', 'w', newline='') as csvfile:
    fieldnames = ['pub_id', 'date', 'page_num', 'column', 'path', 'x_offset', 'y_offset']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    # Iterate through each publication
    for pub in data:
        pub_id = pub['pub_id']
        date = pub['date']
        
        # Iterate through each page
        for page in pub['pages']:
            page_num = page['page_num']
            
            # Iterate through each snippet (column)
            for snippet in page['snippets']:
                writer.writerow({
                    'pub_id': pub_id,
                    'date': date,
                    'page_num': page_num,
                    'column': snippet['column'],
                    'path': snippet['path'],
                    'x_offset': snippet['x_offset'],
                    'y_offset': snippet['y_offset']
                })

print("CSV file created successfully at data/01_preprocessed/all_metadata.csv")
