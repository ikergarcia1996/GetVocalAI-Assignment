import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class ConversationAccuracyScorer:
    def __init__(self, gold_data_path: str, predicted_labels: List[float]):
        """
        Initialize Scorer
        Args:
            gold_data_path: str - path to the gold data
            predicted_labels: List[float] - Probability of each candidate being the correct answer
        """

        self.data = []
        with open(gold_data_path, "r", encoding="utf8") as f:
            for line in f:
                example = json.loads(line)
                self.data.append(example)

        i = 0
        for example in self.data:
            num_candidates = len(example["candidates"])
            example["predicted_label"] = predicted_labels[i : i + num_candidates]
            i += num_candidates

        # Ensure that for each example, we have the same number of predicted labels as candidates
        assert i == len(predicted_labels)

    def compute(self):
        """
        Compute the accuracy of the predicted labels
        Args:
            None
        Returns:
            float: accuracy
        """
        correct = 0
        total = 0
        for example in self.data:
            predicted_labels = example["predicted_label"]
            gold_label = example["label"]
            # Test if no candidate is more than 0.5
            if all([label < 0.5 for label in predicted_labels]):
                predicted_label = -1
            else:
                predicted_label = np.argmax(predicted_labels)

            if predicted_label == gold_label:
                correct += 1
            total += 1

        return correct / total

    def print_predictions(self, output_path: str):
        """
        Print the predictions to a json file
        Args:
            output_path: str
        Returns:
            None
        """

        with open(output_path, "w", encoding="utf8") as f:
            print(json.dump(self.data, f, indent=4, ensure_ascii=False))


class SentenceScorer:
    def __init__(self):
        self.tp: int = 0
        self.fp: int = 0
        self.tn: int = 0
        self.fn: int = 0

        self.tn_examples: List[str] = []
        self.fp_examples: List[str] = []

    def add_batch(
        self,
        input_texts: List[str],
        predicted_labels: List[float],
        gold_labels: List[float],
    ):
        """
        Add a batch of model inputs and labels to the scorer
        Args:
            input_texts: List[str]
            labels: torch.tensor
        Returns:
            None
        """

        if len(gold_labels) != len(predicted_labels):
            raise ValueError(
                f"Number of labels does not match number of inputs. Gold labels: {len(gold_labels)}, Predicted labels: {len(predicted_labels)}"
            )

        for gold_label, predicted_label, text in zip(
            gold_labels, predicted_labels, input_texts
        ):
            if gold_label == 1 and predicted_label == 1:
                self.tp += 1
            elif gold_label == 1 and predicted_label == 0:
                self.fn += 1
                self.tn_examples.append(text)
            elif gold_label == 0 and predicted_label == 1:
                self.fp += 1
                self.fp_examples.append(text)
            elif gold_label == 0 and predicted_label == 0:
                self.tn += 1
            else:
                raise ValueError(
                    f"Invalid label: gold_label={gold_label}, predicted_label={predicted_label}"
                )

    def compute(self) -> Tuple[float, float, float]:
        """
        Compute the precision, recall, and f1 score of the predicted labels
        Args:
            None
        Returns:
            Tuple[float, float, float]: precision, recall, f1
        """
        if self.tp == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = self.tp / (self.tp + self.fp)
            recall = self.tp / (self.tp + self.fn)
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1

    def print_summary(self, output_dir: str) -> Tuple[float, float, float]:
        """
        Print a summary of the scorer a output directory

        Args:
            output_dir: str

        Returns:
            Tuple[float, float, float]: precision, recall, f1
        """
        os.makedirs(output_dir, exist_ok=True)
        precision, recall, f1 = self.compute()
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(
                {
                    "tp": self.tp,
                    "fp": self.fp,
                    "tn": self.tn,
                    "fn": self.fn,
                    "total_examples": self.tp + self.fp + self.tn + self.fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                },
                f,
            )

        self.print_confusion_matrix(os.path.join(output_dir, "confusion_matrix.png"))
        self.print_ablation_examples(output_dir)

    def print_confusion_matrix(self, output_path: str):
        """
        Prints the confusion matrix using matplotlib

        Args:
            None

        Returns:
            None
        """

        # Create a confusion matrix
        confusion_matrix = [[self.tp, self.fp], [self.fn, self.tn]]
        confusion_matrix = np.array(confusion_matrix)

        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(confusion_matrix, cmap="Blues")

        # Add a grid
        ax.grid(False)

        # Set the labels and ticks
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Positive", "Negative"], fontsize=12)
        ax.set_yticklabels(["Positive", "Negative"], fontsize=12)

        # Loop over data dimensions and create text annotations
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{confusion_matrix[i, j]}",
                    ha="center",
                    va="center",
                    color="white"
                    if confusion_matrix[i, j] > confusion_matrix.max() / 2
                    else "black",
                    fontsize=14,
                    fontweight="bold",
                )

        # ax.set_title("Confusion Matrix", fontsize=16, pad=20)
        ax.set_xlabel("True Labels", fontsize=14)
        ax.set_ylabel("Predicted Labels", fontsize=14)

        # Add annotations for the axes to clarify predictions and actuals
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()

        plt.savefig(output_path)

    def print_ablation_examples(self, output_dir: str):
        """
        Print examples of false negatives and false positives
        Args:
            output_dir: str
        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "false_negatives.txt"), "w") as f:
            for example in self.tn_examples:
                f.write(f"{example}\n")

        with open(os.path.join(output_dir, "false_positives.txt"), "w") as f:
            for example in self.fp_examples:
                f.write(f"{example}\n")
