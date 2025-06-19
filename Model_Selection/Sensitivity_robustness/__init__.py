# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def intersperse_borderline_normal_points(data, labels, min_scale=1.05, max_scale=1.1, contextual_length=5):
#     n_samples, n_features = data.shape
#     augmented_data = []
#     augmented_labels = []
#     new_points_indices = []  # Store indices of new points for later plotting
#     num_new_points = int(n_samples * .1)
#     insert_every = n_samples // num_new_points
#     new_point_counter = 0
#     for i in range(n_samples):
#         augmented_data.append(data[i])
#         augmented_labels.append(labels[i])
#
#         if new_point_counter < num_new_points and (i % insert_every == 0 or i == n_samples - 1):
#             new_data = np.zeros(n_features)
#             for j in range(n_features):
#                 start_idx = max(0, i - contextual_length)
#                 end_idx = min(n_samples, i + contextual_length + 1)
#                 local_std = np.std(data[start_idx:end_idx, j])
#                 scale_factor = np.random.uniform(min_scale, max_scale)
#                 noise = np.random.normal(0, local_std * scale_factor)
#                 new_data[j] = noise
#
#             augmented_data.append(new_data)
#             augmented_labels.append(0)  # Mark as normal
#             new_points_indices.append(len(augmented_data) - 1)  # Save index of the new point
#             new_point_counter += 1
#
#     return np.array(augmented_data), np.array(augmented_labels), new_points_indices
#
#
# # Example usage
# np.random.seed(0)
# data = np.random.randn(100, 10)  # Simulated data with 100 samples and 10 features
# labels = np.zeros(100)  # Assuming all initial points are normal
# augmented_data, augmented_labels, new_points_indices = intersperse_borderline_normal_points(data, labels)
# print(f'labels: {labels}')
# print(f'augmented_labels: {augmented_labels}')
# # Visualization
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Original Data")
# plt.plot(data[:, 0], 'b-', label='Feature 0')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.title("Augmented Data with Borderline Points")
# plt.plot(augmented_data[:, 0], 'b-', label='Feature 0 Augmented')
# plt.scatter(new_points_indices, augmented_data[new_points_indices, 0], color='red', label='New Borderline Points')
# plt.legend()
#
# plt.show()
