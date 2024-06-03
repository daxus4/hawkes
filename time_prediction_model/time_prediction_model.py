from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class TimePredictionModel(ABC):
    def __init__(self):
        self._parameters: Dict[str, Any] = dict()

    @abstractmethod
    def train(self, times_in_seconds: np.ndarray, **kwargs) -> None:
        pass

    @abstractmethod
    def predict_next_event_time_from_last_event(
        self, time_in_seconds: np.ndarray, **kwargs
    ) -> float:
        return self.predict_next_event_time_from_current_time(
            time_in_seconds, time_in_seconds[-1], **kwargs
        )

    @abstractmethod
    def predict_next_event_time_from_current_time(
        self, time_in_seconds: np.ndarray, current_time_in_seconds: float, **kwargs
    ) -> float:
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters