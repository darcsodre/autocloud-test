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
        self.variance: np.ndarray = np.zeros_like(
            self.mean
        )  # Variância por dimensão (d,)

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
        D_k = ||x - mean||² / sum(variance)

        Observação:
        - Se sum(variance) <= 0, retorna 1.0 (condição típica no início).
        - Usa soma das variâncias por dimensão como métrica escalar.
        """
        total_var = float(np.sum(self.variance))
        if total_var <= 0:
            return 1.0

        distance_sq = float(np.dot(x.data - self.mean, x.data - self.mean))
        D_k = distance_sq / total_var
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

        Funciona para qualquer dimensionalidade (escalar ou vetorial).

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
            Nova média (μ*) com a mesma forma de old_mean.
        """
        s_new = int(s_new)

        # Garante que old_mean e x.data sejam arrays numpy
        old_mean = np.asarray(old_mean)
        x_data = np.asarray(x.data)

        # Calcula a nova média usando broadcasting do numpy
        # Funciona para arrays de qualquer dimensão
        new_mean = ((s_new - 1) / s_new) * old_mean + (1 / s_new) * x_data

        return new_mean

    # -------------------------
    # Typicality update (forma incremental)
    # -------------------------
    def _update_typicality(self, x: Sample) -> None:
        """
        Atualiza typicality de forma incremental a partir da eccentricidade média.

        Nota: aqui você usa uma forma prática baseada em:
          ecc = ||x - mean||² / sum(variance)
          E = média recursiva de ecc
          typicality = 1/(1+E)

        Isso mantém a ideia de typicality inversamente relacionada à eccentricidade.
        Usa soma das variâncias por dimensão como métrica escalar.
        """
        total_var = float(np.sum(self.variance))
        if total_var <= 0:
            self.typicality = 1.0
            self._E = 0.0
            return

        ecc = float(np.dot(x.data - self.mean, x.data - self.mean)) / total_var
        self._E = ((self.n - 1) * self._E + ecc) / self.n
        self.typicality = 1.0 / (1.0 + self._E)

    # -------------------------
    # Variance update (Welford's algorithm adaptado por dimensão)
    # -------------------------
    def calculate_new_variance(
        self,
        x: Sample,
        new_mean: np.ndarray,
        old_mean: np.ndarray,
        old_variance: np.ndarray,
        s_new: int,
    ) -> np.ndarray:
        """
        Calcula a variância por dimensão atualizada assumindo a inclusão de x na cloud.

        A variância por dimensão é definida como:
            σ²_d = (1/n) Σ (x_i,d - μ_d)²

        Usa Welford's algorithm por dimensão para atualização incremental:
            σ²_n,d = ((n-1)/n) * σ²_{n-1,d} + (1/n) * δ_d · δ'_d
        onde δ_d = x_d - μ_{n-1,d} e δ'_d = x_d - μ_n,d

        Esta fórmula considera corretamente a mudança na média.

        Parameters
        ----------
        x : Sample
            Nova amostra.
        new_mean : np.ndarray
            Média após incluir x (μ_n), shape (d,).
        old_mean : np.ndarray
            Média antes de incluir x (μ_{n-1}), shape (d,).
        old_variance : np.ndarray
            Variância antes de incluir x (σ²_{n-1}), shape (d,).
        s_new : int
            Tamanho após incluir x (n).

        Returns
        -------
        np.ndarray
            Variância atualizada por dimensão (σ²_n), shape (d,), limitada por self.min_var.
        """
        s_new = int(s_new)
        old_variance = np.asarray(old_variance)
        old_mean = np.asarray(old_mean)
        new_mean = np.asarray(new_mean)

        n = s_new
        n_old = s_new - 1

        if n_old == 0:
            # Primeiro ponto: variância é zero em todas as dimensões
            return np.zeros_like(new_mean)

        # Welford's algorithm por dimensão: δ = x - old_mean, δ' = x - new_mean
        # M2_n = M2_{n-1} + δ * δ' (elemento a elemento)
        # σ²_n = M2_n / n
        delta = x.data - old_mean
        delta_prime = x.data - new_mean

        # M2_{n-1} = σ²_{n-1} * (n-1)
        M2_old = old_variance * n_old

        # M2_n = M2_{n-1} + δ * δ' (multiplicação elemento a elemento)
        M2_new = M2_old + delta * delta_prime

        # σ²_n = M2_n / n
        new_variance = M2_new / n

        return np.maximum(new_variance, self.min_var)

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
            old_mean=old_mean,
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
        variance: np.ndarray,
        point: np.ndarray,
    ) -> float:
        """
        Calcula a eccentricidade ξ de um ponto em relação à cloud (forma do paper).

            ξ = ( 1 + (||x - μ||² / sum(σ²)) ) / s

        Parameters
        ----------
        num_points : int
            Número de pontos s (normalização).
        mean : np.ndarray
            Média μ, shape (d,).
        variance : np.ndarray
            Variância por dimensão σ², shape (d,).
        point : np.ndarray
            Vetor do ponto x.

        Returns
        -------
        float
            Eccentricidade ξ.
        """
        total_var = float(np.sum(variance))
        if total_var <= 0:
            return 0.0

        dist2 = float(np.dot(point - mean, point - mean))
        return (1.0 + (dist2 / total_var)) / int(num_points)

    def calculate_normalized_eccentricity(
        self,
        num_points: int,
        mean: np.ndarray,
        variance: np.ndarray,
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
        var_i = self.variance
        var_j = other.variance

        n_i = len(self)
        n_j = len(other)
        n_tot = n_i + n_j

        if n_tot == 0:
            return self

        # Média ponderada
        self.mean = (n_i * mean_i + n_j * mean_j) / n_tot

        # Variância pooled por dimensão (evita divisão por zero)
        if n_i > 1 and n_j > 1:
            self.variance = ((n_i - 1) * var_i + (n_j - 1) * var_j) / (n_tot - 2)
        else:
            # fallback conservador - max elemento a elemento
            self.variance = np.maximum(var_i, var_j)
            self.variance = np.maximum(self.variance, self.min_var)

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

    def print_label_distribution(self) -> None:
        """
        Prints the distribution of points by unique label as proportions and counts.

        Example output:
            Label Distribution:
            label_A: ████████░░ 80.0% (8/10)
            label_B: ██░░░░░░░░ 20.0% (2/10)
        """
        if not self.points:
            print("No points in cloud")
            return

        # Count points by label
        label_counts = {}
        for sample in self.points:
            label = sample.label if hasattr(sample, "label") else "unknown"
            label_counts[label] = label_counts.get(label, 0) + 1

        total = len(self.points)
        bar_length = 10

        print("Label Distribution:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            proportion = count / total
            filled = int(bar_length * proportion)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"  {label}: {bar} {proportion*100:.1f}% ({count}/{total})")
