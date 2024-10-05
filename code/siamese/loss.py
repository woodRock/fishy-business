import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
    
class ContrastiveLossWithBalancedAccuracy(nn.Module):
    def __init__(self, margin=1.0, alpha=0.5):
        super(ContrastiveLossWithBalancedAccuracy, self).__init__()
        self.margin = margin
        self.alpha = alpha  # Weight for balancing contrastive loss and accuracy

    def forward(self, output1, output2, label):
        # Contrastive Loss
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        # Predicted labels (0 if distance > margin/2, else 1)
        pred_labels = (euclidean_distance < self.margin/2).float()

        # Balanced Accuracy
        true_positives = torch.sum((pred_labels == 1) & (label == 1))
        true_negatives = torch.sum((pred_labels == 0) & (label == 0))
        false_positives = torch.sum((pred_labels == 1) & (label == 0))
        false_negatives = torch.sum((pred_labels == 0) & (label == 1))

        sensitivity = true_positives / (true_positives + false_negatives + 1e-8)
        specificity = true_negatives / (true_negatives + false_positives + 1e-8)
        balanced_accuracy = (sensitivity + specificity) / 2

        # Combine contrastive loss and balanced accuracy
        total_loss = self.alpha * loss_contrastive - (1 - self.alpha) * balanced_accuracy

        return total_loss
    

class WeightedContrastiveLossWithBalancedAccuracy(nn.Module):
    def __init__(self, margin=1.0, alpha=0.5, class_weights=None):
        super(WeightedContrastiveLossWithBalancedAccuracy, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.class_weights = class_weights if class_weights is not None else torch.tensor([1.0, 1.0])

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Ensure label is long tensor for indexing
        label = label.long()
        
        # Weighted Contrastive Loss
        class_weight = self.class_weights[label]
        loss_contrastive = torch.mean(
            class_weight * ((1 - label) * torch.pow(euclidean_distance, 2) +
                            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        )

        # Predicted labels (0 if distance > margin/2, else 1)
        pred_labels = (euclidean_distance < self.margin/2).float()

        # Balanced Accuracy
        true_positives = torch.sum((pred_labels == 1) & (label == 1))
        true_negatives = torch.sum((pred_labels == 0) & (label == 0))
        false_positives = torch.sum((pred_labels == 1) & (label == 0))
        false_negatives = torch.sum((pred_labels == 0) & (label == 1))

        sensitivity = true_positives / (true_positives + false_negatives + 1e-8)
        specificity = true_negatives / (true_negatives + false_positives + 1e-8)
        balanced_accuracy = (sensitivity + specificity) / 2

        # Combine weighted contrastive loss and balanced accuracy
        total_loss = self.alpha * loss_contrastive - (1 - self.alpha) * balanced_accuracy

        return total_loss