import numpy as np
import torch
import os

class EarlyStopping:
    """
    Implements early stopping based on validation F1 score with dynamic delta adjustment.
    
    Args:
        patience (int): Number of epochs to wait after the last improvement.
        verbose (bool): If True, prints a message for each improvement.
        delta (float): Initial minimum change in the monitored metric to qualify as an improvement.
        path (str): Path to save the best model.
        dynamic_epochs (int): Number of epochs to use for dynamic delta calculation.
    """
    def __init__(self, patience=3, verbose=True, delta=0.01, path='checkpoint.pt', dynamic_epochs=3):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta  # initial delta value
        self.path = path
        self.dynamic_epochs = dynamic_epochs
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_f1_max = -np.inf
        self.f1_history = []  # Record F1 scores for dynamic delta calculation

    def __call__(self, val_f1, model):
        """
        Check if validation F1 score has improved.
        Dynamically update delta based on early epochs' F1 history.
        If no sufficient improvement is observed within the allowed patience, set early_stop flag to True.
        
        Args:
            val_f1 (float): Current epoch validation F1 score.
            model (nn.Module): Model to save if validation F1 improves.
            
        Returns:
            bool: True if early stopping condition is met, else False.
        """
        # Update history and calculate dynamic delta if applicable
        self.f1_history.append(val_f1)
        if len(self.f1_history) == self.dynamic_epochs:
            range_f1 = max(self.f1_history) - min(self.f1_history)
            new_delta = range_f1 * 0.1  # Use 10% of the range as delta if it is larger
            if new_delta > self.delta:
                if self.verbose:
                    print(f"Dynamic delta updated: {self.delta:.6f} -> {new_delta:.6f}")
                self.delta = new_delta

        # Check for sufficient improvement in a unified branch
        if self.best_score is None or val_f1 >= self.best_score + self.delta:
            if self.verbose:
                if self.best_score is None:
                    print(f"Initial improvement: best_score set to {val_f1:.6f}")
                else:
                    print(f"Validation F1 improved from {self.best_score:.6f} to {val_f1:.6f} (delta: {self.delta:.6f}).")
            self.best_score = val_f1
            self.save_checkpoint(val_f1, model)
            self.counter = 0  # Reset counter on improvement
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, val_f1, model):
        """
        Save the model checkpoint when validation F1 improves.
        
        Args:
            val_f1 (float): Current validation F1 score.
            model (nn.Module): Model to be saved.
        """
        if self.verbose:
            print(f"Saving checkpoint: Validation F1 improved to {val_f1:.6f}.")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_f1_max = val_f1