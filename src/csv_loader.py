import csv
import os 

def fix_data_types(data, set_str=[],set_float=[],set_int=[]):
    print(f"Attempting to fix data types:")
    print(f"To STRING: {set_str}")
    print(f"To FLOAT: {set_float}")
    print(f"To INT: {set_int}")
    for point in data:
        for label in set_str:
            point[label] = str(point[label])
        for label in set_float:
            point[label] = float(point[label])
        for label in set_int:
            point[label] = int(point[label])
    
    return data

def read_csv_points(file_path, headers=None, skip_rows=None, verbose=True):
    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        print(f"Opened CSV: \'{file_path}\'")

        # Skip the specified number of rows
        if skip_rows is not None:
            print(f"Skipping {skip_rows} rows:")
            for i in range(skip_rows):
                row = next(csv_reader)
                if verbose: print(f"[{i+1}]: {row}")
            
        if headers is None:
            # Read the next row as headers
            headers = next(csv_reader)
            print(f"Headers from file: {headers}")
        else:
            # Assume the passed in headers align with the csv data
            print(f"Headers passed in: {headers}")

        # Use DictReader with the headers
        data = []
        i = 0
        for row in csv_reader:
            data.append(dict(zip(headers, row)))
            i += 1
        print(f"Data Rows: {i}")
        
        if verbose: 
            for row in data: 
                print(row)

    return headers, data

# COMBINES FIELDS: Return a data dictionary with an additional field {new_header}, where the data of
# {new_header} is the values of {source_headers} concatenated together separated by underscores
def append_concatenated_header(data, new_header, source_headers, verbose=False):
    for row in data:
        concatenated_value = '_'.join(row[header] for header in source_headers if header in row)
        if verbose: print(f"Concat: {concatenated_value}")
        row[new_header] = concatenated_value
    return data

# FILTERS FIELDS: Return a data dictionary containing only the specified fields
def extract_fields(data, fields):
    extracted_data = [{field: row[field] for field in fields} for row in data if all(field in row for field in fields)]
    return extracted_data

# RENAME FIELD
def rename_field(data, old_field_name, new_field_name):
    """
    Rename a field in a list of dictionaries.

    :param data: List of dictionaries representing the data.
    :param old_field_name: The field (key) you want to rename.
    :param new_field_name: The new name for the field.
    :return: A new list of dictionaries with the renamed field.
    """
    renamed_data = []
    for entry in data:
        # Copy the original dictionary to a new one
        new_entry = entry.copy()
        
        # Rename the field if it exists
        if old_field_name in new_entry:
            new_entry[new_field_name] = new_entry.pop(old_field_name)
        
        renamed_data.append(new_entry)
    print(f"Renamed field \'{old_field_name}\' to \'{new_field_name}\'")
    return renamed_data

def get_headers(data):
    """
    Get the headers (field names) from a list of dictionaries.

    :param data: List of dictionaries representing the data.
    :return: A list of headers (field names).
    """
    if not data:
        return []
    
    # Get the keys from the first dictionary as headers
    headers = list(data[0].keys())
    
    return headers

# Export CSV file
def write_csv(file_path, headers, data):
    # Extract the directory from the file path
    directory = os.path.dirname(file_path)
    
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        print(f"Creating Directory: {directory}")
        os.makedirs(directory)
    
    # Write the data to the specified file path
    with open(file_path, mode='w', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=headers)
        csv_writer.writeheader()
        csv_writer.writerows(data)

    print(f"Finished writing {file_path}")


if __name__ == "__main__":
    # Example usage
    base_dir = "./data/example"
    file_name = 'example.csv'
    file_path = os.path.join(base_dir,file_name)
    skip_rows = 8  # Number of rows to skip before headers and/or data
    headers = None # Define headers if they are not defined in the file

    # Read in points file
    headers, data = read_csv_points(file_path, headers=headers, skip_rows=skip_rows)

    # Create a new field combining several other fields
    new_header = 'label'
    data = append_concatenated_header(data,new_header,['id','type','text'])
    headers.append(new_header)

    # Fix a misnamed field
    data = rename_field(data, 'x-coord', 'x')

    # Extract the desired fields from the imported data
    new_headers = ['label','x','y','z']
    data = extract_fields(data, new_headers)

    # Update your headers based on the data object
    headers = get_headers(data)

    # Show new data
    print("\nNew Data:\n")
    for row in data:
        print(row)

    # Write new data to a csv file with headers
    file_name = 'example_out.csv'
    file_path = os.path.join(base_dir,file_name)
    write_csv(file_path, headers, data)
