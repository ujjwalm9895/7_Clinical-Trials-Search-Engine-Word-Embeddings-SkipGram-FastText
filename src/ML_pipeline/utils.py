# Import necessary libraries and modules
import pandas as pd

# Function to read data from a CSV file
def read_data(file_path, **kwargs):
    # Read the CSV file into a DataFrame for preprocessing
    df = pd.read_csv(file_path, **kwargs)
    
    # Create a separate copy of the DataFrame for returning results
    df1 = pd.read_csv(file_path, **kwargs)
    
    # Return the first 100 rows of the DataFrame for preprocessing
    return df.iloc[:100, :]
