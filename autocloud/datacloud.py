# Aqui contem tudo que é estado + equações recursivas de uma nuvem
# (média, variância, excentricidade, typicality, merge).

import numpy as np
from typing import Union

from .sample import Sample


class DataCloud:
    """
    Representa uma nuvem de dados (cloud) do AutoCloud/TEDA.

    Mantém estatísticas recursivas (média e variância) e métricas associadas:
    - pertinency (membership TEDA)
    - typicality (típico vs excêntrico)
    - conjunto de amostras para controle de interseção (merge)
    """

    N = 0  # Contador global de nuvens de dados

    def __init__(self, x: Sample, **kwargs):
        # Inicializa uma nova nuvem de dados com o primeiro ponto x
        self.n: int = 1  # Número de pontos na nuvem
        self.mean: np.ndarray = x.data.reshape(-1, 1).mean(axis=1)  # (d,)
        self.variance: float = 0.0  # Variância escalar inicial (0 com um único ponto)

        # TEDA/AutoCloud métricas
        self.pertinency: float = 1.0
        self.typicality: float = 1.0
        self._E: float = 0.0  # eccentricidade média acumulada (para typicality)

        # Identidade e armazenamento de pontos
        self.id: int = DataCloud.N
        self.points: list[Sample] = [x]
        self.set_data_points: set[Sample] = {x}

        # Estabilidade numérica
        self.min_var: float = float(kwargs.pop("min_var", 1e-3))

        DataCloud.N += 1

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

    # -------------------------
    # TEDA membership (pertinency)
    # -------------------------
    def _calculate_membership(self, x: Sample) -> float:
        """
        Calcula o grau de pertinência (membership) TEDA de um ponto x na cloud.

        μ_k = 1 / (1 + D_k)
        D_k = ||x - mean||² / variance

        Observação:
        - Se variance <= 0, retorna 1.0 (condição típica no início).
        """
        if self.variance <= 0:
            return 1.0

        distance_sq = float(np.dot(x.data - self.mean, x.data - self.mean))
        D_k = distance_sq / float(self.variance)
        return 1.0 / (1.0 + D_k)

    # -------------------------
    # Mean update (Eq. 14 do AutoCloud 2020)
    # -------------------------
    def calculate_new_mean(
        self,
        x: Sample,
        old_mean: np.ndarray,
        s_new: int,
    ) -> np.ndarray:
        """
        Calcula a média atualizada assumindo a inclusão de x na cloud.

        Implementa a Eq. (14) do artigo (forma equivalente):
            μ* = ((s_new - 1)/s_new) * μ_old + (1/s_new) * x

        Parameters
        ----------
        x : Sample
            Nova amostra.
        old_mean : np.ndarray
            Média antes da inclusão (μ_old).
        s_new : int
            Tamanho da cloud APÓS incluir x (s*). Ex.: s_new = n_old + 1.

        Returns
        -------
        np.ndarray
            Nova média (μ*).
        """
        s_new = int(s_new)
        return ((s_new - 1) / s_new) * old_mean + (1 / s_new) * x.data

    # -------------------------
    # Typicality update (forma incremental)
    # -------------------------
    def _update_typicality(self, x: Sample) -> None:
        """
        Atualiza typicality de forma incremental a partir da eccentricidade média.

        Nota: aqui você usa uma forma prática baseada em:
          ecc = ||x - mean||² / variance
          E = média recursiva de ecc
          typicality = 1/(1+E)

        Isso mantém a ideia de typicality inversamente relacionada à eccentricidade.
        """
        if self.variance <= 0:
            self.typicality = 1.0
            self._E = 0.0
            return

        ecc = float(np.dot(x.data - self.mean, x.data - self.mean)) / float(
            self.variance
        )
        self._E = ((self.n - 1) * self._E + ecc) / self.n
        self.typicality = 1.0 / (1.0 + self._E)

    # -------------------------
    # Variance update (Eq. 15 do AutoCloud 2020)
    # -------------------------
    def calculate_new_variance(
        self,
        x: Sample,
        new_mean: np.ndarray,
        old_variance: float,
        s_new: int,
    ) -> float:
        """
        Calcula a variância escalar atualizada assumindo a inclusão de x na cloud.

        Implementa a Eq. (15) do artigo (AutoCloud 2020):
            σ²* = ((s_new - 1)/s_new) * σ²_old + (1/s_new) * ||x - μ*||²

        onde:
        - σ²_old: variância antes da inclusão
        - μ*: média após inclusão (new_mean)
        - s_new: tamanho após incluir x

        Parameters
        ----------
        x : Sample
            Nova amostra.
        new_mean : np.ndarray
            Média após incluir x (μ*).
        old_variance : float
            Variância antes de incluir x (σ²_old).
        s_new : int
            Tamanho após incluir x (s*).

        Returns
        -------
        float
            Variância atualizada (σ²*), limitada por self.min_var.
        """
        s_new = int(s_new)
        old_variance = float(old_variance)

        dist2 = float(np.dot(x.data - new_mean, x.data - new_mean))  # ||x - μ*||²
        new_variance = ((s_new - 1) / s_new) * old_variance + (1 / s_new) * dist2

        return float(np.maximum(new_variance, self.min_var))

    # -------------------------
    # Cloud update with a new sample
    # -------------------------
    def append_sample_to_datacloud(self, x: Sample) -> None:
        """
        Atualiza a cloud incluindo a amostra x.

        Passos:
        1) calcula pertinency (membership TEDA) com estado atual
        2) incrementa n
        3) atualiza mean (Eq. 14) usando s_new = n
        4) atualiza variance (Eq. 15) usando s_new = n
        5) atualiza typicality
        6) guarda o ponto em list e set (para merge/interseção)
        """
        self.pertinency = self._calculate_membership(x)

        old_mean = self.mean
        old_variance = self.variance

        # após incluir x, o novo tamanho é self.n + 1
        self.n += 1
        s_new = self.n

        self.mean = self.calculate_new_mean(x=x, old_mean=old_mean, s_new=s_new)
        self.variance = self.calculate_new_variance(
            x=x,
            new_mean=self.mean,
            old_variance=old_variance,
            s_new=s_new,
        )

        self._update_typicality(x)

        self.points.append(x)
        self.set_data_points.add(x)

    # -------------------------
    # Eccentricity (Eq. 11/12 no paper: ξ e ζ=ξ/2)
    # -------------------------
    def calculate_eccentricity(
        self,
        num_points: int,
        mean: np.ndarray,
        variance: float,
        point: np.ndarray,
    ) -> float:
        """
        Calcula a eccentricidade ξ de um ponto em relação à cloud (forma do paper).

            ξ = ( 1 + (||x - μ||² / σ²) ) / s

        Parameters
        ----------
        num_points : int
            Número de pontos s (normalização).
        mean : np.ndarray
            Média μ.
        variance : float
            Variância escalar σ².
        point : np.ndarray
            Vetor do ponto x.

        Returns
        -------
        float
            Eccentricidade ξ.
        """
        variance = float(variance)
        if variance <= 0:
            return 0.0

        dist2 = float(np.dot(point - mean, point - mean))
        return (1.0 + (dist2 / variance)) / int(num_points)

    def calculate_normalized_eccentricity(
        self,
        num_points: int,
        mean: np.ndarray,
        variance: float,
        point: np.ndarray,
    ) -> float:
        """
        Calcula a eccentricidade normalizada ζ = ξ/2 (como no paper).
        """
        return self.calculate_eccentricity(num_points, mean, variance, point) / 2.0

    # -------------------------
    # Merge
    # -------------------------
    def merge_dataclouds(self, other: "DataCloud") -> "DataCloud":
        """
        Mescla outra cloud nesta cloud, unificando amostras e atualizando estatísticas.

        Observação:
        - Aqui você usa média ponderada e uma forma pooled de variância.
        - Se quiser ficar 100% “paper-faithful”, dá para re-derivar a variância do
          conjunto unido, mas isso pode ser mais caro.

        Returns
        -------
        DataCloud
            A própria cloud (self) atualizada.
        """
        s_i = self.set_data_points
        s_j = other.set_data_points

        mean_i = self.mean
        mean_j = other.mean
        var_i = float(self.variance)
        var_j = float(other.variance)

        n_i = len(self)
        n_j = len(other)
        n_tot = n_i + n_j

        if n_tot == 0:
            return self

        # Média ponderada
        self.mean = (n_i * mean_i + n_j * mean_j) / n_tot

        # Variância pooled (evita divisão por zero)
        if n_i > 1 and n_j > 1:
            self.variance = ((n_i - 1) * var_i + (n_j - 1) * var_j) / (n_tot - 2)
        else:
            # fallback conservador
            self.variance = max(var_i, var_j, self.min_var)

        # Unifica pontos
        self.set_data_points = s_i | s_j
        self.points = list(self.set_data_points)
        self.n = len(self.points)

        return self

    def __add__(self, x: Union[Sample, "DataCloud"]) -> "DataCloud":
        """
        Se x for Sample: adiciona o ponto.
        Se x for DataCloud: mescla as clouds.
        """
        if isinstance(x, DataCloud):
            if len(x) == 1:
                self.append_sample_to_datacloud(x.points[0])
                return self
            return self.merge_dataclouds(x)

        self.append_sample_to_datacloud(x)
        return self
