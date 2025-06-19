import copy

import tensorflow as tf

from Metrics.metrics import range_based_precision_recall_f1_auc
from Utils.model_selection_utils import evaluate_model


class AdversarialTimeSeriesEvaluator:
    def __init__(self, train_data, test_data, models, algorithm_list, epsilon: float = 0.01):
        """
        Initializes the evaluator with data and models.

        :param train_data: Training data
        :param test_data: Testing data
        :param models: Dictionary of model names and their TensorFlow/Keras instances
        :param epsilon: Perturbation magnitude for adversarial example generation
        """
        self.train_data = train_data
        self.test_data = test_data
        self.trained_models = models
        self.algorithm_list = algorithm_list
        self.epsilon = epsilon
        self.adversarial_data = None
        self.scores = {}

    def create_adversarial_examples(self):
        """Generate adversarial examples from the test set using the FGSM approach."""
        x_test, y_test = self.test_data.entities[0].Y, self.test_data.entities[0].labels
        x_test_adv = x_test.copy()

        for model_name in self.algorithm_list:
            model = self.trained_models[model_name]
            signed_grads = self._compute_gradients(model, x_test, y_test)
            x_test_adv += self.epsilon * signed_grads  # Apply perturbation

        self.adversarial_data = (x_test_adv, y_test)

    def _compute_gradients(self, model, x, y):
        """Compute gradients for adversarial example generation using TensorFlow/Keras models."""
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)

        # Ensure correct input shape
        if len(x_tensor.shape) == len(model.input_shape) - 1:
            x_tensor = tf.expand_dims(x_tensor, axis=0)

        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            predictions = model(x_tensor, training=False)  # Inference mode
            loss = tf.keras.losses.BinaryCrossentropy()(tf.convert_to_tensor(y, dtype=tf.float32), predictions)

        # Compute gradients with respect to the input data
        grads = tape.gradient(loss, x_tensor)
        return tf.sign(grads).numpy()

    def evaluate_models(self, algorithm_list, trained_models):
        """Evaluate all models on both clean and adversarial data using a consistent evaluation methodology."""
        x_test_clean, y_test_clean = self.test_data.entities[0].Y, self.test_data.entities[0].labels
        x_test_adv, y_test_adv = self.adversarial_data

        for model_name in algorithm_list:
            model = trained_models.get(model_name)
            # Evaluate on clean data
            clean_evaluation = evaluate_model(self.test_data, model, model_name)
            y_true_clean = clean_evaluation['anomaly_labels'].flatten()
            y_scores_clean = clean_evaluation['entity_scores'].flatten()
            adv_test_data = copy.deepcopy(self.test_data)
            adv_test_data.entities[0].Y = x_test_adv
            adv_test_data.entities[0].labels = y_test_adv
            # Evaluate on adversarial data
            adv_evaluation = evaluate_model(adv_test_data, model, model_name)
            y_true_adv = adv_evaluation['anomaly_labels'].flatten()
            y_scores_adv = adv_evaluation['entity_scores'].flatten()
            _, _, best_f1_clean, pr_auc_clean, adjusted_y_pred_clean = range_based_precision_recall_f1_auc(y_true_clean,
                                                                                                           y_scores_clean.round())
            _, _, best_f1_adv, pr_auc_adv, adjusted_y_pred_adv = range_based_precision_recall_f1_auc(y_true_adv,
                                                                                                     y_scores_adv.round())
            # Calculate metrics for clean and adversarial data
            self.scores[model_name] = {
                'clean_f1': best_f1_clean,
                'adv_f1': best_f1_adv,
                'clean_pr_auc': pr_auc_clean,
                'adv_pr_auc': pr_auc_adv
            }

    def rank_models(self):
        """Rank models based on their performance on clean and adversarial data."""
        ranked_by_f1 = sorted(self.scores.items(), key=lambda item: item[1]['adv_f1'], reverse=True)
        ranked_by_pr_auc = sorted(self.scores.items(), key=lambda item: item[1]['adv_pr_auc'], reverse=True)
        return ranked_by_f1, ranked_by_pr_auc

    def run_test(self):
        """Conducts the full adversarial test and returns ranked models based on adversarial robustness."""
        self.create_adversarial_examples()
        self.evaluate_models(self.algorithm_list, self.trained_models)
        return self.rank_models()
