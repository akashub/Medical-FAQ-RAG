import pandas as pd
import json

def convert_csv_to_json(csv_file, json_file):
    """
    Converts a CSV file with 'qtype', 'Question', and 'Answer' columns
    into a JSON file with a specific format.
    
    Args:
        csv_file (str): The path to the input CSV file.
        json_file (str): The path to the output JSON file.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file)

        # Check for the required columns
        required_cols = ['qtype', 'Question', 'Answer']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: The CSV file must contain the following columns: {required_cols}")
            return

        # Create a list of dictionaries in the desired JSON format
        json_data = []
        for index, row in df.iterrows():
            json_data.append({
                "question": row["Question"],
                "answer": row["Answer"],
                "qtype": row["qtype"]
            })

        # Write the list of dictionaries to a JSON file
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        print(f"Successfully converted '{csv_file}' to '{json_file}'")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Call the function with your CSV and desired JSON file names
convert_csv_to_json('train.csv', 'medical_faqs.json')