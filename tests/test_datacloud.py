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
        assert np.array_equal(datacloud.variance, np.array([0.0, 0.0, 0.0]))
        assert datacloud.pertinency == 1.0
        assert datacloud.typicality == 1.0
        assert datacloud._E == 0.0  # Eccentricity should be initialized to 0
        assert isinstance(datacloud.id, int)
        assert len(datacloud.points) == 1
        assert datacloud.min_var == 1e-3

    def test_initialization_with_min_var(self, datacloud_min_var_modified: DataCloud):
        # Testa a inicialização da nuvem de dados com min_var modificado
        assert datacloud_min_var_modified.min_var == 1e-12

    def test_calculate_new_mean(self, datacloud: DataCloud):
        """Testa o cálculo da nova média comparando com np.mean"""
        sample = Sample(sample_id=1, data=np.array([2.0, 2.0, 2.0]))
        new_mean = datacloud.calculate_new_mean(
            x=sample, old_mean=datacloud.mean, s_new=2
        )

        # Compara com numpy
        all_points = np.array([datacloud.mean, sample.data])
        expected_mean = np.mean(all_points, axis=0)
        assert np.allclose(new_mean, expected_mean)

    def test_calculate_new_mean_1d(self):
        """Testa o cálculo da nova média com dados 1D comparando com np.mean"""
        initial_point = Sample(sample_id=0, data=np.array([1.0]))
        dc = DataCloud(initial_point)
        sample = Sample(sample_id=1, data=np.array([3.0]))
        new_mean = dc.calculate_new_mean(x=sample, old_mean=dc.mean, s_new=2)

        # Compara com numpy
        all_points = np.array([dc.mean, sample.data])
        expected_mean = np.mean(all_points, axis=0)
        assert np.allclose(new_mean, expected_mean)

    def test_calculate_new_mean_5d(self):
        """Testa o cálculo da nova média com dados 5D comparando com np.mean"""
        initial_point = Sample(sample_id=0, data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        dc = DataCloud(initial_point)
        sample = Sample(sample_id=1, data=np.array([3.0, 4.0, 5.0, 6.0, 7.0]))
        new_mean = dc.calculate_new_mean(x=sample, old_mean=dc.mean, s_new=2)

        # Compara com numpy
        all_points = np.array([dc.mean, sample.data])
        expected_mean = np.mean(all_points, axis=0)
        assert np.allclose(new_mean, expected_mean)

    def test_calculate_new_mean_incremental(self):
        """Testa média incremental com múltiplos pontos comparando com np.mean"""
        initial_point = Sample(sample_id=0, data=np.array([1.0, 1.0]))
        dc = DataCloud(initial_point)

        # Adiciona segundo ponto
        sample2 = Sample(sample_id=1, data=np.array([3.0, 3.0]))
        mean2 = dc.calculate_new_mean(x=sample2, old_mean=dc.mean, s_new=2)
        all_points_2 = np.array([dc.mean, sample2.data])
        expected_mean2 = np.mean(all_points_2, axis=0)
        assert np.allclose(mean2, expected_mean2)

        # Adiciona terceiro ponto
        sample3 = Sample(sample_id=2, data=np.array([6.0, 6.0]))
        mean3 = dc.calculate_new_mean(x=sample3, old_mean=mean2, s_new=3)
        all_points_3 = np.array([dc.mean, sample2.data, sample3.data])
        expected_mean3 = np.mean(all_points_3, axis=0)
        assert np.allclose(mean3, expected_mean3)

    def test_calculate_new_variance(self, datacloud: DataCloud):
        """Testa o cálculo da nova variância por dimensão comparando com np.var"""
        sample = Sample(sample_id=1, data=np.array([2.0, 2.0, 2.0]))
        old_mean = datacloud.mean
        new_mean = datacloud.calculate_new_mean(x=sample, old_mean=old_mean, s_new=2)
        new_variance = datacloud.calculate_new_variance(
            x=sample,
            new_mean=new_mean,
            old_mean=old_mean,
            old_variance=datacloud.variance,
            s_new=2,
        )

        # Compara com numpy var por dimensão
        all_points = np.array([datacloud.mean, sample.data])
        expected_variance = np.var(all_points, axis=0)
        assert np.allclose(new_variance, expected_variance)

    def test_update_data_cloud(self, datacloud: DataCloud):
        """Testa a atualização da nuvem comparando com cálculos numpy"""
        original_id = datacloud.id
        new_point = Sample(sample_id=1, data=np.array([2.0, 2.0, 2.0]))
        datacloud = datacloud + new_point

        # Coleta todos os pontos
        all_data = np.array([p.data for p in datacloud.points])

        # Compara média com numpy
        expected_mean = np.mean(all_data, axis=0)
        assert np.allclose(datacloud.mean, expected_mean)

        # Compara variância por dimensão com numpy
        expected_variance = np.var(all_data, axis=0)
        assert np.allclose(datacloud.variance, expected_variance)

        assert datacloud.n == 2
        assert datacloud.pertinency == 1.0
        assert len(datacloud) == 2
        assert datacloud.id == original_id

    def test_merge_dataclouds(self, datacloud: DataCloud):
        """Testa a fusão de nuvens comparando com cálculos numpy"""
        new_point = Sample(sample_id=1, data=np.array([2.0, 2.0, 2.0]))
        datacloud2 = DataCloud(new_point)

        merged_cloud = datacloud + datacloud2

        # Coleta todos os pontos
        all_data = np.array([p.data for p in merged_cloud.points])

        # Compara média com numpy
        expected_mean = np.mean(all_data, axis=0)
        assert np.allclose(merged_cloud.mean, expected_mean)

        # Compara variância por dimensão com numpy
        expected_variance = np.var(all_data, axis=0)
        assert np.allclose(merged_cloud.variance, expected_variance)

        assert merged_cloud.n == 2
        assert len(merged_cloud.points) == 2

    def test_recursive_mean_variance_multiple_points_3d(self):
        """Testa média e variância recursivas com múltiplos pontos 3D"""
        points = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0]),
            np.array([2.0, 3.0, 4.0]),
            np.array([5.0, 6.0, 7.0]),
        ]

        dc = DataCloud(Sample(sample_id=0, data=points[0]))

        # Adiciona pontos recursivamente
        for i, p in enumerate(points[1:], start=1):
            dc = dc + Sample(sample_id=i, data=p)

        # Compara com numpy
        all_data = np.array(points)
        expected_mean = np.mean(all_data, axis=0)
        expected_variance = np.var(all_data, axis=0)

        assert np.allclose(dc.mean, expected_mean)
        assert np.allclose(dc.variance, expected_variance)

    def test_recursive_mean_variance_multiple_points_1d(self):
        """Testa média e variância recursivas com múltiplos pontos 1D"""
        points = [
            np.array([1.0]),
            np.array([5.0]),
            np.array([3.0]),
            np.array([7.0]),
            np.array([2.0]),
            np.array([9.0]),
        ]

        dc = DataCloud(Sample(sample_id=0, data=points[0]))

        # Adiciona pontos recursivamente
        for i, p in enumerate(points[1:], start=1):
            dc = dc + Sample(sample_id=i, data=p)

        # Compara com numpy
        all_data = np.array(points)
        expected_mean = np.mean(all_data, axis=0)
        expected_variance = np.var(all_data, axis=0)

        assert np.allclose(dc.mean, expected_mean)
        assert np.allclose(dc.variance, expected_variance)

    def test_recursive_mean_variance_random_10d(self):
        """Testa com dados aleatórios de alta dimensão (10D)"""
        np.random.seed(42)
        n_points = 20
        n_dims = 10
        points = [np.random.randn(n_dims) for _ in range(n_points)]

        dc = DataCloud(Sample(sample_id=0, data=points[0]))

        # Adiciona pontos recursivamente
        for i, p in enumerate(points[1:], start=1):
            dc = dc + Sample(sample_id=i, data=p)

        # Compara com numpy
        all_data = np.array(points)
        expected_mean = np.mean(all_data, axis=0)
        expected_variance = np.var(all_data, axis=0)

        assert np.allclose(dc.mean, expected_mean, rtol=1e-10)
        assert np.allclose(dc.variance, expected_variance, rtol=1e-10)

    # -------------------------
    # Testes: média e variância após adicionar sample a datacloud
    # -------------------------
    def test_add_sample_step_by_step_3d(self):
        """Verifica média e variância após cada adição de sample (3D)"""
        np.random.seed(99)
        points = [np.random.randn(3) for _ in range(8)]

        dc = DataCloud(Sample(sample_id=0, data=points[0]))

        for i, p in enumerate(points[1:], start=1):
            dc = dc + Sample(sample_id=i, data=p)

            all_data = np.array(points[: i + 1])
            expected_mean = np.mean(all_data, axis=0)
            expected_var = np.var(all_data, axis=0)

            assert np.allclose(dc.mean, expected_mean), f"Passo {i}: média divergiu"
            assert np.allclose(
                dc.variance, expected_var
            ), f"Passo {i}: variância divergiu"

    def test_add_sample_step_by_step_1d(self):
        """Verifica média e variância após cada adição de sample (1D)"""
        points = [np.array([v]) for v in [2.0, 8.0, 4.0, 10.0, 1.0]]

        dc = DataCloud(Sample(sample_id=0, data=points[0]))

        for i, p in enumerate(points[1:], start=1):
            dc = dc + Sample(sample_id=i, data=p)

            all_data = np.array(points[: i + 1])
            expected_mean = np.mean(all_data, axis=0)
            expected_var = np.var(all_data, axis=0)

            assert np.allclose(dc.mean, expected_mean)
            assert np.allclose(dc.variance, expected_var)

    def test_add_sample_asymmetric_variance(self):
        """Verifica variância por dimensão quando cada eixo tem dispersão diferente"""
        # Dim 0: grande variação, Dim 1: pequena variação
        points = [
            np.array([0.0, 5.0]),
            np.array([10.0, 5.1]),
            np.array([20.0, 4.9]),
            np.array([30.0, 5.0]),
            np.array([40.0, 5.05]),
        ]

        dc = DataCloud(Sample(sample_id=0, data=points[0]))
        for i, p in enumerate(points[1:], start=1):
            dc = dc + Sample(sample_id=i, data=p)

        all_data = np.array(points)
        expected_mean = np.mean(all_data, axis=0)
        expected_var = np.var(all_data, axis=0)

        assert np.allclose(dc.mean, expected_mean)
        assert np.allclose(dc.variance, expected_var)
        # Variância da dim 0 deve ser muito maior que a da dim 1
        assert dc.variance[0] > 100 * dc.variance[1]

    def test_add_sample_random_high_dim(self):
        """Verifica média e variância com 50 pontos aleatórios em 20D"""
        np.random.seed(7)
        n_points = 50
        n_dims = 20
        points = [np.random.randn(n_dims) * (i % 5 + 1) for i in range(n_points)]

        dc = DataCloud(Sample(sample_id=0, data=points[0]))
        for i, p in enumerate(points[1:], start=1):
            dc = dc + Sample(sample_id=i, data=p)

        all_data = np.array(points)
        expected_mean = np.mean(all_data, axis=0)
        expected_var = np.var(all_data, axis=0)

        assert np.allclose(dc.mean, expected_mean, rtol=1e-10)
        assert np.allclose(dc.variance, expected_var, rtol=1e-10)

    # -------------------------
    # Testes: média e variância após somar duas dataclouds
    # -------------------------
    def test_add_two_dataclouds_single_point_each(self):
        """Soma duas dataclouds de 1 ponto — equivale a append_sample"""
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 5.0, 6.0])
        dc1 = DataCloud(Sample(sample_id=0, data=p1))
        dc2 = DataCloud(Sample(sample_id=1, data=p2))

        merged = dc1 + dc2

        all_data = np.array([p1, p2])
        expected_mean = np.mean(all_data, axis=0)
        expected_var = np.var(all_data, axis=0)

        assert np.allclose(merged.mean, expected_mean)
        assert np.allclose(merged.variance, expected_var)
        assert merged.n == 2

    def test_add_two_dataclouds_multi_point(self):
        """Soma duas dataclouds com múltiplos pontos — usa merge com pooled variance"""
        np.random.seed(55)
        points_a = [np.random.randn(4) for _ in range(5)]
        points_b = [np.random.randn(4) for _ in range(5)]

        dc_a = DataCloud(Sample(sample_id=0, data=points_a[0]))
        for i, p in enumerate(points_a[1:], start=1):
            dc_a = dc_a + Sample(sample_id=i, data=p)

        dc_b = DataCloud(Sample(sample_id=10, data=points_b[0]))
        for i, p in enumerate(points_b[1:], start=11):
            dc_b = dc_b + Sample(sample_id=i, data=p)

        n_a, n_b = dc_a.n, dc_b.n
        var_a, var_b = dc_a.variance.copy(), dc_b.variance.copy()

        merged = dc_a + dc_b

        # Média é exata: média ponderada
        all_data = np.array(points_a + points_b)
        expected_mean = np.mean(all_data, axis=0)
        assert np.allclose(merged.mean, expected_mean)

        # Variância usa pooled formula por dimensão
        expected_pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        assert np.allclose(merged.variance, expected_pooled_var)

    def test_add_two_dataclouds_different_sizes(self):
        """Soma duas dataclouds de tamanhos diferentes"""
        np.random.seed(33)
        points_a = [np.random.randn(3) for _ in range(3)]
        points_b = [np.random.randn(3) for _ in range(7)]

        dc_a = DataCloud(Sample(sample_id=0, data=points_a[0]))
        for i, p in enumerate(points_a[1:], start=1):
            dc_a = dc_a + Sample(sample_id=i, data=p)

        dc_b = DataCloud(Sample(sample_id=10, data=points_b[0]))
        for i, p in enumerate(points_b[1:], start=11):
            dc_b = dc_b + Sample(sample_id=i, data=p)

        n_a, n_b = dc_a.n, dc_b.n
        var_a, var_b = dc_a.variance.copy(), dc_b.variance.copy()

        merged = dc_a + dc_b

        # Média ponderada é exata
        all_data = np.array(points_a + points_b)
        expected_mean = np.mean(all_data, axis=0)
        assert np.allclose(merged.mean, expected_mean)

        # Variância pooled por dimensão
        expected_pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        assert np.allclose(merged.variance, expected_pooled_var)

        assert merged.n == n_a + n_b

    def test_normalized_eccentricity_less_than_one(self):
        """ζ = ξ/2 deve ser sempre < 1 para qualquer ponto da cloud."""
        np.random.seed(77)
        dims = [1, 2, 5, 10]
        sizes = [3, 10, 50]

        for d in dims:
            for n in sizes:
                points = [np.random.randn(d) for _ in range(n)]
                dc = DataCloud(Sample(sample_id=0, data=points[0]))
                for i, p in enumerate(points[1:], start=1):
                    dc = dc + Sample(sample_id=i, data=p)

                for p in points:
                    zeta = dc.calculate_normalized_eccentricity(
                        num_points=dc.n,
                        mean=dc.mean,
                        variance=dc.variance,
                        point=p,
                    )
                    assert zeta < 1.0, f"ζ={zeta} >= 1 para d={d}, n={n}, point={p}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
