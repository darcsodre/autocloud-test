import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, ".."))

import pytest

from autocloud.sample import Sample
import numpy as np


class TestSample:
    @pytest.fixture
    def sample(self) -> Sample:
        # Fixture para criar uma instância de Sample
        return Sample(sample_id=0, data=np.array([1.0, 2.0, 3.0]))

    def test_initialization(self, sample: Sample):
        # Testa a inicialização do Sample
        assert sample.sample_id == 0
        assert np.array_equal(sample.data, np.array([1.0, 2.0, 3.0]))

    def test_hash(self, sample: Sample):
        # Testa o método __hash__
        assert isinstance(hash(sample), int)

    def test_equality(self, sample: Sample):
        # Testa a igualdade entre Samples
        same_sample = Sample(sample_id=0, data=np.array([1.0, 2.0, 3.0]))
        different_sample = Sample(sample_id=1, data=np.array([4.0, 5.0, 6.0]))

        assert sample == same_sample
        assert sample != different_sample

    def test_length(self, sample: Sample):
        # Testa o método __len__
        assert len(sample) == 1
