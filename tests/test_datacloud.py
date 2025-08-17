import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, ".."))

import pytest


from autocloud.datacloud import DataCloud
from autocloud.sample import Sample
import numpy as np


class TestDataCloud:
    @pytest.fixture
    def datacloud(self) -> DataCloud:
        # Fixture para criar uma instância de DataCloud
        initial_point = Sample(sample_id=0, data=np.array([1.0, 1.0, 1.0]))
        return DataCloud(initial_point)

    @pytest.fixture
    def datacloud_min_var_modified(self) -> DataCloud:
        # Fixture para criar uma instância de DataCloud
        initial_point = Sample(sample_id=0, data=np.array([1.0, 1.0, 1.0]))
        return DataCloud(initial_point, min_var=1e-12)

    def test_initialization(self, datacloud: DataCloud):
        # Testa a inicialização da nuvem de dados
        assert datacloud.n == 1
        assert np.array_equal(datacloud.mean, np.array([1.0, 1.0, 1.0]))
        assert datacloud.variance == 0.0
        assert datacloud.pertinency == 1.0
        assert datacloud.typicality == 1.0
        assert datacloud._E == 0.0  # Eccentricity should be initialized to 0
        assert isinstance(datacloud.id, int)
        assert datacloud.id == 0
        assert datacloud.points == [Sample(sample_id=0, data=np.array([1.0, 1.0, 1.0]))]
        assert datacloud.N == 1
        assert datacloud.min_var == 1e-3

    def test_initialization_with_min_var(self, datacloud_min_var_modified: DataCloud):
        # Testa a inicialização da nuvem de dados com min_var modificado
        assert datacloud_min_var_modified.min_var == 1e-12

    def test_calculate_new_mean(self, datacloud: DataCloud):
        # Testa o cálculo da nova média
        sample = Sample(sample_id=1, data=np.array([2.0, 2.0, 2.0]))
        new_mean = datacloud.calculate_new_mean(
            x=sample, old_mean=datacloud.mean, num_samples=2
        )
        expected_mean = np.array([1.5, 1.5, 1.5])
        assert np.array_equal(new_mean, expected_mean)

    def test_calculate_new_variance(self, datacloud: DataCloud):
        # Testa o cálculo da nova variância
        sample = Sample(sample_id=1, data=np.array([2.0, 2.0, 2.0]))
        new_mean = datacloud.calculate_new_mean(
            x=sample, old_mean=datacloud.mean, num_samples=2
        )
        new_variance = datacloud.calculate_new_variance(
            x=sample,
            new_mean=new_mean,
            old_mean=datacloud.mean,
            old_variance=datacloud.variance,
            num_samples=2,
        )
        expected_variance = 0.75
        assert np.isclose(new_variance, expected_variance)

    def test_update_data_cloud(self, datacloud: DataCloud):
        # Testa a atualização da nuvem de dados com um novo ponto
        new_point = Sample(sample_id=1, data=np.array([2.0, 2.0, 2.0]))
        # datacloud.update_data_cloud(new_point)
        datacloud = datacloud + new_point

        assert datacloud.n == 2
        assert np.array_equal(datacloud.mean, np.array([1.5, 1.5, 1.5]))
        assert datacloud.variance == 0.75
        assert datacloud.pertinency == 1.0
        assert len(datacloud) == 2
        assert datacloud.points == [
            Sample(sample_id=0, data=np.array([1.0, 1.0, 1.0])),
            Sample(sample_id=1, data=np.array([2.0, 2.0, 2.0])),
        ]
        assert datacloud.id == 0  # ID deve permanecer o mesmo

    def test_merge_dataclouds(self, datacloud: DataCloud):
        # Testa a fusão de duas nuvens de dados
        new_point = Sample(sample_id=1, data=np.array([2.0, 2.0, 2.0]))
        datacloud2 = DataCloud(new_point)

        merged_cloud = datacloud + datacloud2

        assert merged_cloud.n == 2
        assert np.array_equal(merged_cloud.mean, np.array([1.5, 1.5, 1.5]))
        assert merged_cloud.variance == 0.75
        assert len(merged_cloud.points) == 2
        assert merged_cloud.points == [
            Sample(sample_id=0, data=np.array([1.0, 1.0, 1.0])),
            Sample(sample_id=1, data=np.array([2.0, 2.0, 2.0])),
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
