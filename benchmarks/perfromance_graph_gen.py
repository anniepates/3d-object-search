import numpy as np
import random
import matplotlib.pyplot as plt
import csv

# Constants
NUM_TRIALS = 10
LABELS = [f"label_{i}" for i in range(10)]
DEPENDENT_VARS = ["Data-Set Size", "Geo Similarity Threshold", "Target-Model Density", "Label-Relevance Threshold",
                  "Target-Label Density"]


def generate_mock_data(dep_var, value):
    """Generates mock data based on the dependent variable."""
    if dep_var == "Data-Set Size":
        num_models = int(value)  # Vary the number of models directly
    else:
        num_models = 100  # Default number of models

    models = [f"model_{i}" for i in range(num_models)]
    keyword_dict = {}

    if dep_var == "Geo Similarity Threshold":
        # Simulate the effect of geo similarity by varying the number of labels
        keyword_dict = {model: random.sample(LABELS, max(1, int(value * len(LABELS)))) for model in models}

    elif dep_var == "Target-Model Density":
        # Increase density by increasing the number of labels per model
        keyword_dict = {model: random.sample(LABELS, random.randint(int(0.7 * len(LABELS) * value), len(LABELS))) for
                        model in models}

    elif dep_var == "Label-Relevance Threshold":
        # Increase relevance by filtering out less relevant labels
        threshold_labels = LABELS[:int(value * len(LABELS))]
        keyword_dict = {model: random.sample(threshold_labels, random.randint(1, len(threshold_labels))) for model in
                        models}

    elif dep_var == "Target-Label Density":
        # Vary the density by changing the number of models associated with each label
        for model in models:
            label_count = max(1, int(value * len(LABELS)))
            keyword_dict[model] = random.sample(LABELS, label_count)

    else:
        # Default case for Data-Set Size
        keyword_dict = {model: random.sample(LABELS, random.randint(1, len(LABELS))) for model in models}

    return models, keyword_dict


def calculate_performance_metrics(target_set, noise_set, keyword_dict, labels, relevance_threshold):
    ground_truth_labels = set(labels[:len(labels) // 2])
    erroneous_labels = set(labels[len(labels) // 2:])

    tp = fp = tn = fn = 0

    for model in target_set:
        inferred_labels = set(keyword_dict[model])
        tp += len(inferred_labels.intersection(ground_truth_labels))
        fp += len(inferred_labels.intersection(erroneous_labels))
        fn += len(ground_truth_labels - inferred_labels)
        tn += len(erroneous_labels - inferred_labels)

    return tp, fp, tn, fn


def run_experiments():
    results = {var: [] for var in DEPENDENT_VARS}

    for dep_var in DEPENDENT_VARS:
        for trial in range(NUM_TRIALS):
            # Vary the dependent variable
            if dep_var == "Data-Set Size":
                value = random.randint(50, 200)  # Vary the number of models
            elif dep_var == "Geo Similarity Threshold":
                value = random.uniform(0.1, 0.9)  # Vary the similarity threshold
            elif dep_var == "Target-Model Density":
                value = random.uniform(0.1, 1.0)  # Vary the density
            elif dep_var == "Label-Relevance Threshold":
                value = random.uniform(0.1, 0.9)  # Vary the relevance threshold
            elif dep_var == "Target-Label Density":
                value = random.uniform(0.1, 1.0)  # Vary the label density

            models, keyword_dict = generate_mock_data(dep_var, value)
            target_set = random.sample(models, len(models) // 5)
            noise_set = list(set(models) - set(target_set))

            tp, fp, tn, fn = calculate_performance_metrics(target_set, noise_set, keyword_dict, LABELS, 0.5)

            results[dep_var].append((value, tp, fp, tn, fn))

    return results


def average_results(results):
    averaged_results = {}

    for dep_var, values in results.items():
        avg_values = {
            'tp': np.mean([x[1] for x in values]),
            'fp': np.mean([x[2] for x in values]),
            'tn': np.mean([x[3] for x in values]),
            'fn': np.mean([x[4] for x in values]),
        }
        averaged_results[dep_var] = avg_values

    return averaged_results


def plot_results(results):
    for dep_var, values in results.items():
        x_vals = [x[0] for x in values]
        tp_vals = [x[1] for x in values]
        fp_vals = [x[2] for x in values]
        tn_vals = [x[3] for x in values]
        fn_vals = [x[4] for x in values]

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, tp_vals, label="True Positive", marker="o")
        plt.plot(x_vals, fp_vals, label="False Positive", marker="o")
        plt.plot(x_vals, tn_vals, label="True Negative", marker="o")
        plt.plot(x_vals, fn_vals, label="False Negative", marker="o")
        plt.title(f'Performance Metrics vs {dep_var}')
        plt.xlabel(dep_var)
        plt.ylabel('Metrics')
        plt.legend()
        plt.grid(True)
        plt.show()


def save_results_to_csv(results, filename):
    """Save each trial's results to a CSV file."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Dependent Variable", "Value", "True Positive", "False Positive", "True Negative", "False Negative"])

        for dep_var, values in results.items():
            for value in values:
                writer.writerow([dep_var, value[0], value[1], value[2], value[3], value[4]])


def save_averaged_results_to_csv(averaged_results, filename):
    """Save the average results to a CSV file."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Dependent Variable", "Average True Positive", "Average False Positive", "Average True Negative",
             "Average False Negative"])

        for dep_var, avg_values in averaged_results.items():
            writer.writerow([dep_var, avg_values['tp'], avg_values['fp'], avg_values['tn'], avg_values['fn']])


# Running the experiments
results = run_experiments()

# Save all trial results to CSV
save_results_to_csv(results, 'trial_results.csv')

# Calculate and save average results to CSV
averaged_results = average_results(results)
save_averaged_results_to_csv(averaged_results, 'average_results.csv')

# Plot the results
plot_results(results)
