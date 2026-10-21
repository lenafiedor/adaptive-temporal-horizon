from collections import deque

from adaptive_horizon import config


class HorizonScheduler:
    """Curriculum over the training horizon T, growing it by one per stage.

    A stage (one T) ends after ``max_epochs`` epochs or, when ``patience`` is
    set, once the monitored loss stalls: its mean over the last
    ``smoothing_epochs`` epochs at the current T has not improved by
    ``min_delta`` (relative) for ``patience`` epochs, after at least
    ``min_epochs`` epochs. Once T reaches ``T_max``, new stages keep starting
    at ``T_max``.
    """

    def __init__(
        self,
        T_max: int,
        max_epochs: int,
        patience: int | None = None,
        min_delta: float = 0.0,
        min_epochs: int = 1,
        smoothing_epochs: int = 1,
    ):
        self.T = 1
        self.T_max = T_max
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.smoothing_epochs = smoothing_epochs
        self.recent = deque(maxlen=smoothing_epochs)
        self.stage_score = None
        self._start_stage()

    @classmethod
    def fixed(cls, T_max: int, epochs: int):
        """Equal-length stages: ``epochs // T_max`` epochs per T."""
        return cls(T_max, max_epochs=max(1, epochs // T_max))

    @classmethod
    def stall(cls, T_max: int):
        """Stall-triggered stages configured by ``[cross-validation]``."""
        return cls(
            T_max,
            max_epochs=config.CV_MAX_STAGE_EPOCHS,
            patience=config.CV_STALL_PATIENCE,
            min_delta=config.CV_STALL_MIN_DELTA,
            min_epochs=config.CV_MIN_STAGE_EPOCHS,
            smoothing_epochs=config.CV_SMOOTHING_EPOCHS,
        )

    def _start_stage(self):
        self.stage_epochs = 0
        self.best = float("inf")
        self.wait = 0

    def step(self, loss: float | None = None, force_boundary: bool = False) -> bool:
        """Record one epoch; returns whether it ended a stage.

        At a stage boundary, ``stage_score`` holds the smoothed monitored loss
        (None if no loss was given) and T advances.
        """
        self.stage_epochs += 1
        if loss is not None:
            self.recent.append(float(loss))
            smoothed = sum(self.recent) / len(self.recent)
            if smoothed < self.best * (1.0 - self.min_delta):
                self.best = smoothed
                self.wait = 0
            else:
                self.wait += 1

        stalled = (
            self.patience is not None
            and self.stage_epochs >= self.min_epochs
            and self.wait >= self.patience
        )
        if not (stalled or self.stage_epochs >= self.max_epochs or force_boundary):
            return False

        self.stage_score = sum(self.recent) / len(self.recent) if self.recent else None
        if self.T < self.T_max:
            # Losses from a different T must not leak into the next stage's score.
            self.recent.clear()
            self.T += 1
        self._start_stage()
        return True

    def describe(self) -> dict:
        return {
            "max_stage_epochs": self.max_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_stage_epochs": self.min_epochs,
            "smoothing_epochs": self.smoothing_epochs,
        }
