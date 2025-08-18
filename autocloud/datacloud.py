# import logging
import numpy as np

from typing import Union

from .sample import Sample

# LOGGER = logging.getLogger(__name__)


class DataCloud:
    N = 0  # Contador global de nuvens de dados

    def __init__(self, x: Sample, **kwargs):
        # Inicializa uma nova nuvem de dados com o primeiro ponto x
        self.n: int = 1  # Número de pontos na nuvem
        self.mean: np.ndarray = x.data.reshape(-1, 1).mean(axis=1)  # Média (centroide)
        self.variance: float = 0.0  # Variância inicial (0 para um único ponto)
        self.pertinency: float = 1.0
        self.typicality: float = 1.0
        self._E: float = 0.0  # eccentricidade média acumulada
        self.id: int = DataCloud.N  # NOVO: ID único e fixo para cada cloud
        self.points: list[Sample] = [
            x
        ]  # NOVO2: Armazena pontos para cálculo de intersecção
        DataCloud.N += 1  # Incrementa contador global
        # self._merged_clouds: set[int] = set()  # NOVO: Armazena IDs de nuvens mescladas
        self.set_data_points = set(
            [x]
        )  # NOVO: Armazena pontos para controle de intersecção
        self.min_var: float = kwargs.pop("min_var", 1e-3)

    def __repr__(self):
        return (
            f"DataCloud(\n"
            f"  id={self.id},\n"
            f"  n={self.n},\n"
            f"  mean={self.mean},\n"
            f"  variance={self.variance},\n"
            f"  pertinency={self.pertinency},\n"
            f"  typicality={self.typicality}\n"
            f")"
        )

    def __len__(self):
        return len(self.points)

    def __hash__(self):
        return hash(self.id)

    def _calculate_membership(self, x: Sample) -> float:
        """
        Calculates TEDA membership degree for a new point x.

        μ_k = 1 / (1 + D_k)
        where D_k = ||x - mean||^2 / variance

        Parameters
        ----------
        x : Sample
            The new point.

        Returns
        -------
        float
            Membership degree in [0, 1].
        """
        if self.variance <= 0:
            return 1.0  # First point always has membership 1

        distance_sq = np.linalg.norm(x.data - self.mean) ** 2
        D_k = distance_sq / self.variance
        return 1.0 / (1.0 + D_k)

    def calculate_new_mean(
        self, x: Sample, old_mean: np.ndarray, num_samples: int = None
    ) -> np.ndarray:
        """
        Calculates the updated mean after adding a new data point.

        Uses Equation 4 from the referenced article to update the mean incrementally
        when a new point `x` is added to the dataset.

        Parameters
        ----------
        x : Sample
            The new data point to be included in the mean calculation.
        old_mean : np.ndarray
            The mean before including the new data point.
        num_samples : int, optional
            The number of samples after including the new data point.
            If not provided, it defaults to the current number of points in the cloud.

        Returns
        -------
        np.ndarray
            The updated mean after including the new data point.
        """
        num_samples = num_samples or self.n
        return ((num_samples - 1) * old_mean + x.data) / num_samples

    def _update_typicality(self, x: Sample):
        """
        Eq. 8 - Incremental typicality from eccentricity

        Parameters
        ----------
        x : Sample
            The new data point to update typicality.
        """
        if self.variance <= 0:
            self.typicality = 1.0
            self._E = 0.0
            return

        ecc = np.linalg.norm(x.data - self.mean) ** 2 / self.variance
        self._E = ((self.n - 1) * self._E + ecc) / self.n
        self.typicality = 1.0 / (1.0 + self._E)

    def calculate_new_variance(
        self,
        x: Sample,
        new_mean: np.ndarray,
        old_mean: np.ndarray,
        old_variance: float,
        num_samples: int = None,
    ) -> float | np.ndarray[float]:
        """
        Calculates the updated variance after adding a new data point.

        Uses Equation 5 from the referenced article to update the variance incrementally
        when a new point `x` is added to the dataset. This method ensures that the
        variance does not fall below a specified minimum threshold.

        Parameters
        ----------
        x : Sample
            The new data point to be included in the variance calculation.
        new_mean : np.ndarray
            The updated mean after including the new data point.
        old_mean : np.ndarray
            The mean before including the new data point.
        old_variance : float
            The variance before including the new data point.
        num_samples : int, optional
            The number of samples after including the new data point.
            If not provided, it defaults to the current number of points in the cloud.

        Returns
        -------
        float | np.ndarray[float]
            The updated variance after including the new data point, constrained by min_var.
        """
        num_samples = num_samples or self.n
        delta = x.data - old_mean
        delta2 = x.data - new_mean
        new_variance = ((num_samples + 1) * old_variance + np.dot(delta, delta2)) / (
            num_samples
        )
        return np.maximum(new_variance, self.min_var)

    def append_sample_to_datacloud(self, x: Sample):
        """
        Updates the data cloud with a new point `x` using recursive equations.

        This method updates the pertinency (membership), mean, and variance of
        the data cloud according to recursive formulas (Eqs. 4 and 5 from the
        referenced article). It also updates the typicality and stores the new
        point for intersection control.

        Parameters
        ----------
        x : Sample
            The new data point to be added to the cloud.

        Returns
        -------
        None

        Notes
        -----
        - The mean and variance are updated using custom recursive methods.
        - The method also updates pertinency and typicality, and appends the
        new point to the internal list.
        """
        self.pertinency = self._calculate_membership(x)
        old_mean = self.mean
        old_variance = self.variance
        self.n += 1

        # Atualiza média usando o novo método
        self.mean = self.calculate_new_mean(x, old_mean=old_mean)
        # Atualiza variância usando o novo método, passando old_variance
        self.variance = self.calculate_new_variance(
            x, self.mean, old_mean, old_variance
        )
        self._update_typicality(x)
        self.points.append(x)  # NOVO2: salva ponto para controle de intersecção
        self.set_data_points.add(x)  # NOVO: salva ponto para controle de intersecção

    def calculate_eccentricity(
        self,
        num_points: np.ndarray,
        mean: float,
        variance: float,
        point: np.ndarray,
    ) -> float:
        """
        Calculates the eccentricity of a point relative to a mean and variance of the
        cloud.

        Eccentricity is the normalized squared distance of the point from the mean,
        scaled by the variance and normalized by the number of points.

        Parameters
        ----------
        num_points : np.ndarray
            The number of points (or an array of counts) used for normalization.
        mean : float
            The mean value of the data cloud.
        variance : float
            The variance of the data cloud.
        point : np.ndarray
            The point for which to calculate eccentricity.

        Returns
        -------
        float
            The calculated eccentricity value.
        """
        if variance <= 0:
            return 0.0

        eccentricity = (
            1 + ((np.linalg.norm(point - mean) ** 2) / variance)
        ) / num_points
        return eccentricity

    def calculate_normalized_eccentricity(
        self, num_points: np.ndarray, mean: float, variance: float, point: np.ndarray
    ) -> float:
        """
        Calculates the normalized eccentricity of a point relative to a mean and
        variance of the cloud.

        Normalized eccentricity is the normalized squared distances
        of the point from the mean, scaled by the variance, divided by 2.

        Parameters
        ----------
        num_points : np.ndarray
            The number of points (or an array of counts) used for normalization.
        mean : float
            The mean value of the data cloud.
        variance : float
            The variance of the data cloud.
        point : np.ndarray
            The point for which to calculate eccentricity.

        Returns
        -------
        float
            The calculated normalized eccentricity value.
        """
        return (
            self.calculate_eccentricity(
                num_points=num_points, mean=mean, variance=variance, point=point
            )
            / 2
        )

    def merge_dataclouds(self, other: "DataCloud") -> "DataCloud":
        """
        Merges another DataCloud into this one, updating the mean, variance,
        pertinency, and typicality accordingly.

        Parameters
        ----------
        other : DataCloud
            The DataCloud to be merged into this one.

        Returns
        -------
        DataCloud
            The updated DataCloud after merging.
        """
        s_i = self.set_data_points
        s_j = other.set_data_points
        mean_i = self.mean
        mean_j = other.mean
        variance_i = self.variance
        variance_j = other.variance
        weighted_mean = (len(self) * mean_i + len(other) * mean_j) / (
            len(self) + len(other)
        )
        weighted_variance = (
            (len(self) - 1) * variance_i + (len(other) - 1) * variance_j
        ) / (len(self) + len(other) - 2)
        self.points = list(s_i | s_j)  # Unifica os pontos
        self.mean = weighted_mean
        self.variance = weighted_variance
        self.set_data_points = s_i | s_j  # Atualiza o conjunto de pontos
        return self

    def __add__(self, x: Union[Sample, "DataCloud"]) -> "DataCloud":
        """
        Merges another DataCloud or a new point into this DataCloud.

        If `x` is a DataCloud, merges its points into this cloud, updating
        the mean, variance, pertinency, and typicality accordingly.
        If `x` is a single data point (Sample), adds it to this cloud.

        Parameters
        ----------
        x : Sample or DataCloud
            The new point or DataCloud to be merged into this cloud.

        Returns
        -------
        DataCloud
            The updated DataCloud after merging.
        """
        if isinstance(x, DataCloud):
            if len(x) == 1:
                # If x is a single point, append it to the current cloud
                self.append_sample_to_datacloud(x.points[0])
                return self
            # If x is a DataCloud, merge it with the current cloud
            return self.merge_dataclouds(x)
        # Assume x is a single data point (Sample)
        self.append_sample_to_datacloud(x)
        return self
