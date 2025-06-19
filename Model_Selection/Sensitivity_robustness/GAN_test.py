import copy
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.keras import layers, models

from Metrics.metrics import range_based_precision_recall_f1_auc, prauc, f1_score
from Utils.model_selection_utils import evaluate_model


# Define the generator and discriminator models for GAN
def make_generator_model(input_dim):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(input_dim, activation='tanh')
    ])
    return model


def make_discriminator_model(input_dim):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


# GAN training function with label smoothing and noise
def train_gan(generator, discriminator, data, epochs=100, batch_size=32, noise_dim=10):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    generator_optimizer = tf.keras.optimizers.Adam(0.0001)
    discriminator_optimizer = tf.keras.optimizers.Adam(0.0001)

    for epoch in range(epochs):
        for _ in range(data.shape[0] // batch_size):
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_data = data[idx]

            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            fake_data = generator.predict(noise)

            # Add Gaussian noise to discriminator input
            real_data += 0.1 * np.random.normal(size=real_data.shape)
            fake_data += 0.1 * np.random.normal(size=fake_data.shape)

            with tf.GradientTape() as disc_tape:
                real_output = discriminator(real_data)
                fake_output = discriminator(fake_data)

                # Apply label smoothing
                real_labels = tf.ones_like(real_output) * 0.9
                fake_labels = tf.zeros_like(fake_output) + 0.1

                disc_loss_real = cross_entropy(real_labels, real_output)
                disc_loss_fake = cross_entropy(fake_labels, fake_output)
                disc_loss = disc_loss_real + disc_loss_fake

            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            with tf.GradientTape() as gen_tape:
                fake_data = generator(noise, training=True)
                fake_output = discriminator(fake_data)
                gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        if epoch % 50 == 0:
            logger.info(f"Epoch {epoch}: Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}")


# Function to generate new borderline points
def generate_borderline_points(generator, num_samples=100, noise_dim=10):
    noise = np.random.normal(0, 1, (num_samples, noise_dim))
    generated_data = generator.predict(noise)
    return generated_data


def integrate_gan_with_dataset(data, labels):
    """
    Integrates generated GAN data into the existing dataset with dynamic labeling,
    and returns the updated dataset, labels, indices of normal and anomaly injected points,
    and the total count of labels after integration.

    Args:
        data (np.ndarray): Original data array of shape (n_features, n_samples).
        labels (np.ndarray): Original label array.

    Returns:
        tuple: Updated data, labels, indices of normal and anomaly injected points,
               and the total number of labels.
    """
    input_dim = data.shape[0]  # Assuming data is of shape (n_features, n_samples)
    generator = make_generator_model(input_dim)
    discriminator = make_discriminator_model(input_dim)

    # Train the GAN
    train_gan(generator, discriminator, data.T, epochs=100, batch_size=32, noise_dim=input_dim)

    # Generate borderline points
    num_samples = int(0.1 * len(labels))  # % of the total number of data points
    borderline_points = generate_borderline_points(generator, num_samples=num_samples, noise_dim=input_dim)

    # Use the discriminator to dynamically label the generated points
    discriminator_outputs = discriminator.predict(borderline_points).flatten()
    print(f"Discriminator outputs: {discriminator_outputs}")  # Debug print
    new_labels = np.where(discriminator_outputs > np.mean(discriminator_outputs), 1, 0)
    print(f"Generated labels: {new_labels}")  # Debug print

    # Adjust labels for near-anomaly classification
    normal_threshold = np.mean(discriminator_outputs)
    new_labels[discriminator_outputs > normal_threshold] = 1

    print(f"Adjusted labels after anomaly threshold: {new_labels}")  # Debug print

    # Integrate the generated points into the original dataset using windows
    num_windows = len(data[0]) // 10  # Divide the data into windows
    indices_to_insert = np.array_split(np.arange(len(data[0])), num_windows)

    integrated_data = []
    integrated_labels = []
    injected_normal_indices = []
    injected_anomaly_indices = []

    current_index = 0
    for window in indices_to_insert:
        # Add the original data points for this window
        integrated_data.append(data[:, window])
        integrated_labels.append(labels[window])
        current_index += len(window)

        # Add a generated point in the middle of the window
        if len(borderline_points) > 0:
            integrated_data.append(borderline_points[:1].T)
            integrated_labels.append(new_labels[:1])
            if new_labels[0] == 0:
                injected_normal_indices.append(current_index)
            else:
                injected_anomaly_indices.append(current_index)
            current_index += 1
            borderline_points = borderline_points[1:]
            new_labels = new_labels[1:]

    integrated_data = np.concatenate(integrated_data, axis=1)
    integrated_labels = np.concatenate(integrated_labels, axis=0)

    # Count the total number of labels after integration
    total_labels_count = len(integrated_labels)

    return integrated_data, integrated_labels, np.array(injected_normal_indices), np.array(
        injected_anomaly_indices), total_labels_count


def plot_data_with_injected_points(original_data, augmented_data, injected_normal_indices, injected_anomaly_indices,
                                   dataset, entity, feature_index=0):
    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    # Plot the original data on the first subplot
    axes[0].plot(original_data[feature_index, :], color='darkblue', linestyle='--', label='Original Data')
    axes[0].set_title(f'Original Data for Feature {feature_index}')
    axes[0].set_ylabel('Feature Value')
    axes[0].legend()
    axes[0].grid(True)

    # Plot the augmented data on the second subplot
    axes[1].plot(augmented_data[feature_index, :], color='lightblue', label='Augmented Data')

    # Highlight injected normal points in green
    if injected_normal_indices.size > 0:
        axes[1].scatter(injected_normal_indices,
                        [augmented_data[feature_index, idx] for idx in injected_normal_indices],
                        color='green', label='Injected Normal Points', marker='o', s=50)

    # Highlight injected anomaly points in red
    if injected_anomaly_indices.size > 0:
        axes[1].scatter(injected_anomaly_indices,
                        [augmented_data[feature_index, idx] for idx in injected_anomaly_indices],
                        color='red', label='Injected Anomaly Points', marker='x', s=50)

    axes[1].set_title(f'Augmented Data with Injected Points for Feature {feature_index}')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Feature Value')
    axes[1].legend()
    axes[1].grid(True)
    # Specify the directory
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory = f'myresults/robustness/GAN/{dataset}/{entity}/'
    filename = f'{dataset}_{entity}_Data_vs_DataWithAnomalies_{date_time_string}.png'
    full_path = os.path.join(directory, filename)

    # Check if the directory exists, and if not, create it
    os.makedirs(directory, exist_ok=True)  # Add exist_ok=True to avoid FileExistsError

    # Save the figure
    plt.savefig(full_path, dpi=300)  # Save as PNG file with high resolution

    plt.show()


def run_Gan(test_data, trained_models, model_names, dataset, entity):
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataSet_before = copy.deepcopy(test_data)
    data = test_data.entities[0].Y
    labels = test_data.entities[0].labels
    factor = .1
    augmented_data, augmented_labels, injected_normal_indices, injected_anomaly_indices, total_labels_count = integrate_gan_with_dataset(
        data, labels)
    test_data.entities[0].Y = np.array(augmented_data)
    test_data.entities[0].labels = np.array(augmented_labels)
    n_times = test_data.entities[0].n_time
    test_data.total_time = total_labels_count
    test_data.entities[0].n_time = total_labels_count
    test_data.entities[0].mask = np.ones(test_data.entities[0].Y.shape)
    # original_data, augmented_data, injected_normal_indices, injected_anomaly_indices
    plot_data_with_injected_points(dataSet_before.entities[0].Y, test_data.entities[0].Y, injected_normal_indices,
                                   injected_anomaly_indices, dataset, entity)
    results = {}
    adjusted_y_pred_dict = {}
    for model_name in model_names:
        model = trained_models.get(model_name)
        results[model_name] = []
        adjusted_y_pred_dict[model_name] = []
        if model:
            evaluation = evaluate_model(test_data, model, model_name)  # Assume this function returns a dict
            y_true = evaluation['anomaly_labels'].flatten()
            y_scores = evaluation['entity_scores'].flatten()
            _, _, best_f1, pr_auc, adjusted_y_pred = range_based_precision_recall_f1_auc(y_true, y_scores)
            adjusted_y_pred_dict[model_name].append(adjusted_y_pred)
            results[model_name].append({'f1': best_f1, 'pr_auc': pr_auc})
            logger.info(f"Evaluated {model_name}: F1={best_f1}, PR_AUC={pr_auc}")

    ranked_by_f1 = sorted(results.items(), key=lambda x: x[1][0]['f1'], reverse=True)
    ranked_by_f1_names = [item[0] for item in ranked_by_f1]
    ranked_by_pr_auc = sorted(results.items(), key=lambda x: x[1][0]['pr_auc'], reverse=True)
    ranked_by_pr_auc_names = [item[0] for item in ranked_by_pr_auc]

    true_values = np.array(test_data.entities[0].labels)  # 1 for anomaly, 0 for normal
    print(10 * '=')
    predicted_values = np.array(adjusted_y_pred_dict[ranked_by_f1_names[0]]).flatten()  # Flatten the list of arrays

    # Converting boolean predictions to integer for easy plotting (True to 1, False to 0)
    predicted_int = predicted_values.astype(int)

    # Identifying incorrect predictions
    incorrect_predictions = predicted_int != true_values
    misclassified_count = np.sum(incorrect_predictions)  # Number of misclassifications
    total_anomalies = np.sum(true_values)  # Total number of real anomalies
    total_data = len(true_values)  # Total number of data points
    print(incorrect_predictions)
    print(misclassified_count)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, '.', label='True Values (Anomalies)', color='blue')  # Plot true values
    plt.plot(predicted_int, 'x', label='Predicted Values (Anomalies)', color='red')  # Plot predicted values
    plt.scatter(np.where(incorrect_predictions)[0], predicted_int[incorrect_predictions], facecolors='none',
                edgecolors='purple', s=100, label='Incorrect Predictions', linewidth=2)
    plt.title(
        f'True vs. Predicted Anomalies \n Misclassified Labels: {misclassified_count}\n Total Anomalies: {total_anomalies} \n Total Data: {total_data}')
    plt.xlabel('Index')
    plt.ylabel('Anomaly Presence')
    plt.yticks([0, 1], ['No Anomaly', 'Anomaly'])  # Set y-ticks to be explicit about what 0 and 1 represent
    plt.legend()
    plt.grid(True)

    # Specify the directory
    directory = f'myresults/robustness/GAN/{dataset}/{entity}/'
    filename = f'{dataset}_{entity}_Misclassified_Anomalies_{date_time_string}_.png'
    full_path = os.path.join(directory, filename)

    # Check if the directory exists, and if not, create it
    os.makedirs(directory, exist_ok=True)  # Add exist_ok=True to avoid FileExistsError

    # Save the figure
    plt.savefig(full_path, dpi=300)  # Save as PNG file with high resolution

    plt.show()

    return ranked_by_f1, ranked_by_pr_auc, ranked_by_f1_names, ranked_by_pr_auc_names
