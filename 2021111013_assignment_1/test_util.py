import numpy as np
import pickle
import sys
import time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

possible_distance_metrics = ["euclidean", "manhattan", "cosine"]
possible_weights = ["uniform", "distance"]
possible_encodings = ["resnet", "vit"]

best_model_file_path = "best_model.pkl"

def euclidean_dist(pointA, pointB):
    sum = np.sum((pointA - pointB) ** 2)
    return np.sqrt(sum)

def manhattan_dist(pointA, pointB):
    return np.sum(np.abs(pointA - pointB))

def cosine_dist(pointA, pointB):
    return 1 - np.dot(pointA[0], pointB[0]) / (np.linalg.norm(pointA[0]) * np.linalg.norm(pointB[0]))

class KNN():
    def __init__(self, k: int, distance_metrics: str = "euclidean", weights: str = "uniform", encoder: str = "ResNet"):
        assert k > 0, "k must be greater than 0"
        self.k = k

        assert distance_metrics.lower() in possible_distance_metrics, "invalid distance_metrics"
        self.distance_metrics = distance_metrics.lower()
        if self.distance_metrics == "euclidean":
            self.dist_func = euclidean_dist
        elif self.distance_metrics == "manhattan":
            self.dist_func = manhattan_dist
        elif self.distance_metrics == "cosine":
            self.dist_func = cosine_dist

        assert weights.lower() in possible_weights, "invalid weights"
        self.weights = weights.lower()

        assert encoder.lower() in possible_encodings, "invalid encoder"
        self.encoder = encoder.lower()

        self.train_data = list()
        self.train_label = list()

    def fit(self, dataset):
        if self.encoder == "resnet":
            self.train_data = dataset[:,1]
        elif self.encoder == "vit":
            self.train_data = dataset[:,2]
        
        self.train_label = dataset[:,3]
    
    def predict(self, test_data):
        assert len(self.train_data) > 0, "train_data is empty"
        assert len(self.train_label) > 0, "train_label is empty"

        pred = list()
        encoder_test_data = list()
        if self.encoder == "resnet":
            encoder_test_data = test_data[:,1]
        elif self.encoder == "vit":
            encoder_test_data = test_data[:,2]
        for test in encoder_test_data:
            dist = np.array([self.dist_func(test, train) for train in self.train_data])
            idx = np.argsort(dist) # Get sorting index list

            k_labels = self.train_label[idx][:self.k]

            # Written with help from GitHub Copilot
            if self.weights == "uniform":
                unique_labels, label_counts = np.unique(k_labels, return_counts=True)
                pred.append(unique_labels[np.argmax(label_counts)])
            elif self.weights == "distance":
                k_dist = dist[idx][:self.k]
                weights = 1 / k_dist
                unique_labels, label_weights = np.unique(k_labels, return_inverse=True)
                weighted_counts = np.bincount(label_weights, weights=weights)
                pred.append(unique_labels[np.argmax(weighted_counts)])
                # label_weights = np.bincount(k_labels, weights=weights, minlength=np.max(k_labels) + 1)
                # pred.append(np.argmax(label_weights))
        return pred
    
    def scoring(self, actual_labels, pred_labels):
        f1 = f1_score(actual_labels, pred_labels, zero_division=0, average='weighted')
        accuracy = accuracy_score(actual_labels, pred_labels)
        precision = precision_score(actual_labels, pred_labels, zero_division=0, average='weighted')
        recall = recall_score(actual_labels, pred_labels, zero_division=0, average='weighted')

        # Return a dictionary of scores rounded off to 4 decimal places
        return {'f1': round(f1, 4), 'accuracy': round(accuracy, 4), 'precision': round(precision, 4), 'recall': round(recall, 4)}