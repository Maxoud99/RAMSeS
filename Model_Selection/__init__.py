import numpy as np

# Example data: a 3x3 matrix, each column could represent a different algorithm's output
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Column headers, assume each corresponds to an algorithm instance
headers = ['LOF_1', 'LOF_2', 'LOF_3']

# Define the algorithms you want to include in your ensemble
ensemble = ['LOF_1', 'LOF_3']

# Convert the headers to a NumPy array for vectorized operations
header_array = np.array(headers)

# Determine which headers are in the ensemble
desired_mask = np.isin(header_array, ensemble)

# Filter the columns of data array based on the desired headers
filtered_data = data[:, desired_mask]

print(filtered_data)
