import numpy as np


class AdaptiveHorizonScheduler:
    """
    Adapts temporal horizon T based on local Lyapunov exponent (LLE).

    The idea: Start with a small T (easy predictions) and increase T during training_results
    based on the system's chaotic properties as measured by LLE.
    """

    def __init__(self,
                 initial_T: int = 1,
                 max_T: int = 32,
                 lle_trajectory: np.ndarray = None,
                 update_frequency: int = 10,
                 warmup_epochs: int = 5):
        self.initial_T = initial_T
        self.max_T = max_T
        self.current_T = initial_T
        self.lle_trajectory = lle_trajectory
        self.update_frequency = update_frequency
        self.warmup_epochs = warmup_epochs
        self.history = []

    def should_increase_T(self, epoch: int, val_loss: float, best_val_loss: float) -> bool:
        if epoch < self.warmup_epochs or self.current_T >= self.max_T:
            return False

        if (epoch - self.warmup_epochs) % self.update_frequency != 0:
            return False

        if val_loss <= best_val_loss * 1.1:
            return True

        return False

    def increase_T(self) -> int:
        if self.current_T >= self.max_T:
            return self.current_T

        new_T = min(self.current_T + 1, self.max_T)
        self.current_T = new_T
        self.history.append({
            'T': new_T,
            'increased_at_epoch': len(self.history)
        })

        return new_T
