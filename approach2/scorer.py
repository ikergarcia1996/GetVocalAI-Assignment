from typing import List
import json
import numpy as np
import os

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
            if gold_label == -1:
                continue # We skip this case for now
            # Test if no candidate is more than 0.5
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf8") as f:
            print(json.dump(self.data, f, indent=4, ensure_ascii=False))
