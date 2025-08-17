import numpy as np


class Sample:
    def __init__(self, sample_id: int, data: np.ndarray):
        self.sample_id = sample_id
        self.data = data

    def __repr__(self):
        return f"Sample(sample_id={self.sample_id}, data={self.data})"

    def __hash__(self):
        return hash((self.sample_id, self.data.tobytes()))

    def __eq__(self, other):
        if not isinstance(other, Sample):
            return False
        return self.sample_id == other.sample_id and np.array_equal(
            self.data.flatten(), other.data.flatten()
        )

    def __len__(self):
        return 1
