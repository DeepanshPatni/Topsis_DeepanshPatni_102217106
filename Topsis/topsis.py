import numpy as np  # Importing numpy for mathematical calculations
import pandas as pd  # Importing pandas to work with data in a table-like format
import sys  # Importing sys to handle system tasks like reading command-line arguments
import os  # Importing os to handle file paths

# Function to read data from a file (either Excel or CSV)
def read_data(input_file):
    """
    This function reads data from a file (CSV or Excel).
    If the file is an Excel file (.xlsx), it reads it with pandas.
    If the file is a CSV file (.csv), it reads it as a CSV.
    """
    file_extension = os.path.splitext(input_file)[-1].lower()  # Get the file extension
    
    if file_extension == '.xlsx':  # If the file is an Excel file
        try:
            data = pd.read_excel(input_file)  # Read the Excel file
            return data
        except FileNotFoundError:
            print(f"Error: The file '{input_file}' does not exist.")
            sys.exit(1)  # Stop the program if the file is not found
    elif file_extension == '.csv':  # If the file is a CSV file
        data = pd.read_csv(input_file)  # Read the CSV file
        return data
    else:
        print("Error: Unsupported file type. Only .xlsx and .csv are supported.")
        sys.exit(1)  # Stop the program if the file type is not supported

# Function to normalize the data (convert it to a common scale)
def normalize_data(data):
    """
    This function normalizes the data by dividing each value by the square root of the sum of its squares.
    This makes the data easier to compare.
    """
    matrix = data.iloc[:, 1:].values  # Get all data except the first column (assumed to be labels)
    norm_matrix = matrix / np.sqrt(np.sum(matrix ** 2, axis=0))  # Normalize the data
    return pd.DataFrame(norm_matrix, columns=data.columns[1:], index=data.index)  # Return the normalized data

# Function to apply weights to the normalized data
def apply_weights(normalized_matrix, weights):
    """
    This function multiplies the normalized data by the weights (importance) of each criterion.
    """
    weighted_matrix = normalized_matrix * weights  # Multiply each column by the corresponding weight
    return weighted_matrix

# Function to calculate the ideal (best) and anti-ideal (worst) solutions
def calculate_ideal_solutions(weighted_matrix, impacts):
    """
    This function calculates the ideal and anti-ideal solutions for each criterion.
    """
    ideal_positive = []  # List to store the ideal solutions (best values)
    ideal_negative = []  # List to store the anti-ideal solutions (worst values)

    # Convert weighted_matrix to a NumPy array to allow NumPy-style indexing
    weighted_matrix = weighted_matrix.values

    # Loop through each column (criterion)
    for i in range(weighted_matrix.shape[1]):
        if impacts[i] == '+':  # Benefit criterion
            ideal_positive.append(np.max(weighted_matrix[:, i]))  # Maximum value for ideal
            ideal_negative.append(np.min(weighted_matrix[:, i]))  # Minimum value for anti-ideal
        else:  # Cost criterion
            ideal_positive.append(np.min(weighted_matrix[:, i]))  # Minimum value for ideal
            ideal_negative.append(np.max(weighted_matrix[:, i]))  # Maximum value for anti-ideal

    return np.array(ideal_positive), np.array(ideal_negative)  # Return ideal and anti-ideal solutions


# Function to calculate the Euclidean distance to the ideal and anti-ideal solutions
def compute_distances(weighted_matrix, ideal_positive, ideal_negative):
    """
    This function calculates the distance of each alternative from the ideal and anti-ideal solutions.
    The distance is calculated using the Euclidean distance formula.
    """
    distance_positive = np.sqrt(np.sum((weighted_matrix - ideal_positive) ** 2, axis=1))  # Distance to the ideal solution
    distance_negative = np.sqrt(np.sum((weighted_matrix - ideal_negative) ** 2, axis=1))  # Distance to the anti-ideal solution
    return distance_positive, distance_negative  # Return both distances

# Function to calculate the TOPSIS score
def calculate_scores(distance_positive, distance_negative):
    """
    This function calculates the TOPSIS score for each alternative.
    The score is calculated as the distance to the anti-ideal solution divided by the sum of the distances to both the ideal and anti-ideal solutions.
    """
    scores = distance_negative / (distance_positive + distance_negative)  # TOPSIS formula
    return scores

# Function to rank the alternatives based on their TOPSIS scores
def rank_alternatives(scores):
    """
    This function ranks the alternatives based on their TOPSIS scores.
    The alternative with the highest score gets rank 1.
    """
    return np.argsort(scores)[::-1] + 1  # Sort scores in descending order and assign ranks

# Function to validate the inputs (weights and impacts)
def validate_inputs(data, weights, impacts):
    """
    This function checks if the inputs are valid.
    It ensures that the number of weights and impacts match the number of criteria,
    and checks that all values in the data are numeric.
    """
    if len(weights) != data.shape[1] - 1:  # Check if weights match the number of criteria
        print("Error: The number of weights must match the number of criteria.")
        sys.exit(1)
    
    if len(impacts) != data.shape[1] - 1:  # Check if impacts match the number of criteria
        print("Error: The number of impacts must match the number of criteria.")
        sys.exit(1)

    if not all(impact in ['+', '-'] for impact in impacts):  # Check if impacts are either '+' or '-'
        print("Error: Impacts must be either '+' or '-'.")
        sys.exit(1)

    for col in data.columns[1:]:  # Check if all columns contain numeric values
        if not pd.api.types.is_numeric_dtype(data[col]):
            print(f"Error: Column '{col}' contains non-numeric values.")
            sys.exit(1)

# Main function that ties everything together
def main(input_file, weights, impacts, result_file):
    """
    This is the main function where the entire TOPSIS method is executed.
    It reads the data, processes it, and saves the result.
    """
    data = read_data(input_file)  # Read the input data
    weights = np.array([float(w) for w in weights.split(',')])  # Convert weights from string to float
    impacts = impacts.split(',')  # Convert impacts from string to list

    validate_inputs(data, weights, impacts)  # Validate the inputs

    normalized_data = normalize_data(data)  # Normalize the data
    weighted_data = apply_weights(normalized_data, weights)  # Apply weights to the data

    ideal_positive, ideal_negative = calculate_ideal_solutions(weighted_data, impacts)  # Calculate the ideal and anti-ideal solutions

    distance_positive, distance_negative = compute_distances(weighted_data, ideal_positive, ideal_negative)  # Calculate the distances

    scores = calculate_scores(distance_positive, distance_negative)  # Calculate the TOPSIS scores

    data['TOPSIS Score'] = scores  # Add the TOPSIS scores to the data
    data['Rank'] = rank_alternatives(scores)  # Rank the alternatives based on their scores

    data.to_csv(result_file, index=False)  # Save the result to a CSV file
    print(f"Results saved to {result_file}")  # Print a message indicating the result file

# Check if the script is being run directly
if __name__ == '__main__':
    if len(sys.argv) != 5:  # Check if the correct number of arguments are provided
        print("Usage: python topsis.py <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)
        

    input_file = sys.argv[1]  # Get the input file
    weights = sys.argv[2]  # Get the weights
    impacts = sys.argv[3]  # Get the impacts
    result_file = sys.argv[4]  # Get the result file name

    main(input_file, weights, impacts, result_file)  # Call the main function to run the program
