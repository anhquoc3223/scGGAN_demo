import pandas as pd
import numpy as np


#load file CSV

def load_data(file_path:str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path, header=None)
        data = data.drop(index=0)  # Drop the first row
        data =data.drop(columns=0)  # Drop the first column
        data = data.apply(pd.to_numeric, errors='coerce')  # Convert all data to numeric
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
# Save file CSV
def save_data(data: pd.DataFrame, file_path: str) -> bool:
    """
    Save a pandas DataFrame to a CSV file.

    Parameters:
    data (pd.DataFrame): The DataFrame to save.
    file_path (str): The path to the output CSV file.
    """
    try:
        data.to_csv(file_path, index=False, header=False)
        print(f"Data saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

# filter out the genes expressed in fewer than 10 cells and the cells with fewer than 200 expressed genes and normalize data into range [0;1] by dividing with the maximum count of each column (cell)
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by filtering and normalizing.

    Parameters:
    data (pd.DataFrame): The input data.

    Returns:
    pd.DataFrame: The preprocessed data.
    """
    # Filter out genes expressed in fewer than 10 cells
    data = data.loc[:, (data > 0).sum(axis=0) >= 10]
    
    # Filter out cells with fewer than 200 expressed genes
    data = data[(data > 0).sum(axis=1) >= 200]
    
    # Normalize data into range [0, 1] by dividing with the maximum count of each column (cell)
    data = data.div(data.max(axis=0).replace(0, 1), axis=1)
    
    return data
if __name__ == "__main__":
    # Example usage
    file_path = './data/sim2.counts.csv'  # Replace with your file path
    data = load_data(file_path)
    data_normalized = preprocess_data(data)
    print('Processing.....')
    save_data(data_normalized, './data/sim2.counts.normalized.csv')
    print('Done')