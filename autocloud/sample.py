# Aqui é a “unidade” do fluxo (amostra com ID e vetor de features).

import numpy as np


class Sample:
    def __init__(self, sample_id: int, data: np.ndarray, label: str = None):
        self.sample_id = sample_id
        self.data = data
        self.label = label

    def __repr__(self):
        return (
            f"Sample(sample_id={self.sample_id}, data={self.data}"
            +
            (f", label={self.label}" if self.label is not None else "")
            + 
            ")"
        )

    def __hash__(self):
        return hash((self.sample_id))

    def __eq__(self, other):
        if not isinstance(other, Sample):
            return False
        return self.sample_id == other.sample_id

    def __len__(self):
        return 1
