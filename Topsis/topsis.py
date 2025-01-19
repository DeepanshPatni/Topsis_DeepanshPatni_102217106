import pandas as pd
import numpy as np
import sys

def validate_inputs(weights, impacts, num_criteria):
    try:
        weights = list(map(float, weights.split(',')))
        impacts = impacts.split(',')
        if len(weights) != num_criteria or len(impacts) != num_criteria:
            raise ValueError("Weights and impacts must match the number of criteria.")
        if not all(i in ['+', '-'] for i in impacts):
            raise ValueError("Impacts must be '+' or '-'.")
    except Exception as e:
        raise ValueError(f"Invalid weights or impacts: {e}")
    
    return weights, impacts

def calculate_topsis(data, weights, impacts):
    # Step 1: Normalization
    normalized_matrix = data.iloc[:, 1:].div(np.sqrt((data.iloc[:, 1:] ** 2).sum()), axis=1)

    # Step 2: Weighted normalized decision matrix
    weighted_matrix = normalized_matrix.mul(weights, axis=1)

    # Step 3: Determine ideal best and worst
    ideal_best = np.where(impacts == '+', weighted_matrix.max(), weighted_matrix.min())
    ideal_worst = np.where(impacts == '+', weighted_matrix.min(), weighted_matrix.max())

    # Step 4: Calculate distances from ideal best and worst
    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

    # Step 5: Calculate TOPSIS score
    topsis_score = dist_worst / (dist_best + dist_worst)
    return topsis_score

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python <RollNumber>.py <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)
    
    input_file, weights, impacts, output_file = sys.argv[1:]

    try:
        # Read and validate data
        data = pd.read_csv(input_file)
        if data.shape[1] < 3:
            raise ValueError("Input file must contain at least 3 columns.")
        if not data.iloc[:, 1:].applymap(lambda x: isinstance(x, (int, float))).all().all():
            raise ValueError("All criteria columns must contain numeric values.")

        # Validate weights and impacts
        weights, impacts = validate_inputs(weights, impacts, data.shape[1] - 1)
        impacts = np.array(impacts)

        # Calculate TOPSIS
        data['Topsis Score'] = calculate_topsis(data, weights, impacts)
        data['Rank'] = data['Topsis Score'].rank(ascending=False).astype(int)

        # Save results
        data.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    except FileNotFoundError:
        print("Input file not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")
