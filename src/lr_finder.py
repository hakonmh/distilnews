import matplotlib.pyplot as plt
import numpy as np
import copy
from transformers import PreTrainedModel


class LRFinder:
    """Learning Rate Finder class to help find optimal learning rate for a given model
    and dataset using a cyclical learning rate policy.
    """

    def __init__(self, model, optimizer, criterion=None, device="cuda"):
        """Initialize the Learning Rate Finder with the given model, optimizer and criterion.

        Args:
            model: The PyTorch model to be trained.
            optimizer: The optimizer used for training.
            criterion (optional): The loss function used for training.
            device (str, optional): The device to train the model on. Defaults to "cuda".
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {"lr": [], "loss": []}

        self._model_state = copy.deepcopy(self.model.state_dict())
        self._optimizer_state = copy.deepcopy(self.optimizer.state_dict())

    def run(self, train_loader, init_value=1e-5, final_value=1e0, beta=0.05):
        """Train the model for one epoch with different learning rates and record the loss.

        Args:
            train_loader: The DataLoader for the training set.
            init_value (float, optional): The initial learning rate value. Defaults to 1e-5.
            final_value (float, optional): The final learning rate value. Defaults to 1e0.
            beta (float, optional): The smoothing factor for the loss. Defaults to 0.05.
        """
        lr = init_value
        best_loss = None

        while True:
            for batch in train_loader:
                self._update_lr(lr)
                loss = self._train_one_round(batch)
                # Compute the smoothed gradient and logg learning rate and loss
                if self.history["loss"]:
                    loss = beta * loss.item() + (1 - beta) * self.history["loss"][-1]
                else:
                    loss = loss.item()

                self.history["lr"].append(lr)
                self.history["loss"].append(loss)

                if best_loss is None or loss < best_loss:
                    best_loss = loss
                # Stop if the loss is exploding
                if lr > final_value or loss > 2 * best_loss:
                    return
                lr *= 1.1

    def _update_lr(self, new_lr):
        """Update the learning rate of the optimizer.

        Args:
            new_lr (float): The new learning rate value.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _train_one_round(self, batch):
        """Train the model for one batch and return the loss.

        Args:
            batch: A tuple containing features and targets.

        Returns:
            loss (torch.Tensor): The loss value for the current batch.
        """
        self.optimizer.zero_grad()
        features, targets = batch
        features = features.to(device=self.device)
        targets = targets.to(device=self.device)

        if isinstance(self.model, PreTrainedModel):
            output = self.model(**features, labels=targets)
            loss = output.loss
        else:
            output = self.model(features)
            if isinstance(output, tuple):
                output = output[0]
            loss = self.criterion(output.float(), targets.float())
        loss.backward()
        self.optimizer.step()

        return loss

    def plot(self):
        """Plot the learning rate range and corresponding losses, mark where
        the loss gradient is steepest and return its learning rate.

        Returns:
            best_lr (float): The learning rate corresponding to the steepest loss gradient.
        """
        last_idx = self._get_max_lr_idx()
        best_lr_index = (np.gradient(np.array(self.history["loss"][:last_idx]))).argmin()

        # Plot the learning rate range and corresponding losses
        plt.plot(self.history["lr"][:last_idx], self.history["loss"][:last_idx])
        plt.scatter(self.history["lr"][best_lr_index], self.history["loss"][best_lr_index], c='red')
        plt.xscale('log')

        plt.xlabel('Log Learning Rate')
        plt.ylabel('Loss')
        plt.show()

        return self.history["lr"][best_lr_index]

    def _get_max_lr_idx(self):
        """Get the index of the first learning rate where max - mean >= mean - min.

        Returns:
            max_lr_idx (int): The index of the first learning rate meeting the condition.
        """
        mean_lr = sum(self.history["loss"]) / len(self.history["loss"])
        min_lr = min(self.history["loss"])
        max_lr_tresh = 2 * mean_lr - min_lr

        try:
            max_lr_idx = next(i for i, v in enumerate(self.history["loss"]) if v >= max_lr_tresh)
        except StopIteration:
            max_lr_idx = len(self.history["loss"]) - 1
        return max_lr_idx

    def reset(self):
        """
        Reset the model and optimizer to their initial state.
        """
        self.model.load_state_dict(self._model_state)
        self.optimizer.load_state_dict(self._optimizer_state)
        self.model.to(self.device)
