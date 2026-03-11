# Aqui contem a lógica do algoritmo (decidir em qual cloud entra, criar nova cloud,
# disparar merge sob demanda).

import logging
import numpy as np

from .datacloud import DataCloud
from .sample import Sample


LOGGER = logging.getLogger(__name__)


class AutoCloud:

    def __init__(self, chebyshev_parameter: float):
        self.chebyshev_factor = (
            chebyshev_parameter**2 + 1
        )  # parametro de Chebyshev, usado para calcular o threshold de entrada em uma cloud
        self.data_clouds: list[DataCloud] = (
            []
        )  # começa vazio: nenhuma cloud criada ainda.
        self.internal_counter = 0  # conta quantas amostras já passaram pelo algoritmo. E usadoara tratar a segunda amostra.
        self.last_merge_eventos = []

    def verify_merge(self, set_cloud_i: set[Sample], set_cloud_j: set[Sample]) -> bool:
        """
        Verifies if two DataClouds can be merged, given their sets of samples.

        Parameters
        ----------
        set_cloud_i : set[Sample]
            The first DataCloud to check.
        set_cloud_j : set[Sample]
            The second DataCloud to check.

        Returns
        -------
        bool
            True if the clouds can be merged, False otherwise.
        """

        # Esse método implementa o critério de merge por interseção de amostras
        # (igual ao paper: comparar interseção com partes exclusivas).
        intersection_i_j = (
            set_cloud_i & set_cloud_j
        )  # amostras que estão em ambas as clouds.
        s_i_minus_j = set_cloud_i - intersection_i_j  # amostras só da cloud i.
        s_j_minus_i = set_cloud_j - intersection_i_j  # amostras só da cloud j.

        # Faz merge se a interseção é maior do que a parte exclusiva de pelo menos uma das clouds.
        # “Se duas nuvens compartilham mais amostras em comum do que amostras exclusivas (de pelo menos uma delas),
        # então elas estão praticamente representando a mesma coisa → faz sentido unir.”
        if (len(intersection_i_j) > len(s_i_minus_j)) or (
            len(intersection_i_j) > len(s_j_minus_i)
        ):
            return True
        return False

    # Esse método faz uma rodada de merge, mas do jeito “first match”:
    # Ele procura o primeiro par (i,j) que satisfaz verify_merge.
    # Assim que encontra, faz o merge e interrompe os loops (break).
    # Depois ele monta a nova lista:
    # inclui a cloud mergeada
    # inclui as clouds que não participaram do merge
    def iterate_merge(self, data_clouds: list[DataCloud]) -> list[DataCloud]:
        new_clouds: list[DataCloud] = []  # lista final após essa rodada.
        merged_clouds: set[int] = (
            set()
        )  # índices das clouds que foram “consumidas” no merge.
        has_merged = False  # Ela começa falsa. O código entra nos loops e vai testando pares (i, j)
        for i in range(0, len(data_clouds) - 1):
            # iterações (bubble sort-like) para verificar merges entre clouds
            # Quando ele encontra um par que pode ser mergeado.
            # Ele entra no if self.verify_merge(...)e faz: has_merged = True.
            for j in range(i + 1, len(data_clouds)):
                if self.verify_merge(
                    data_clouds[i].set_data_points, data_clouds[j].set_data_points #detecção de merges
                ):
                    self.last_merge_eventos.append((data_clouds[i].id, data_clouds[j].id))
                
                    new_cloud = (
                        data_clouds[i] + data_clouds[j]
                    )  # “cria uma cloud nova que é a fusão das duas clouds i e j”.
                    new_clouds.append(new_cloud)
                    merged_clouds.add(i)
                    merged_clouds.add(j)
                    has_merged = True
                if has_merged:
                    break  # Esse break sai do loop do j assim que o primeiro merge acontece.
            if has_merged:
                break  # Esse segundo break sai do loop do i também.
        for i in range(len(data_clouds)):
            if i not in merged_clouds:
                new_clouds.append(data_clouds[i])
        if len(new_clouds) == 0:
            new_clouds = data_clouds
        return new_clouds

    # merge_clouds(): chama iterate_merge() várias vezes para fazer vários merges, um por iteração, até estabilizar.
    def merge_clouds(self) -> None:
        has_merged = True
        self.last_merge_eventos = []
        data_clouds_candidates = self.data_clouds.copy()
        while has_merged:
            new_clouds = self.iterate_merge(data_clouds_candidates)
            has_merged = len(new_clouds) < len(data_clouds_candidates)
            data_clouds_candidates = new_clouds.copy()
        self.data_clouds = new_clouds

    def run_single_sample(self, sample: Sample) -> dict:
        # Guarda apenas as informações que o main precisa imprimir.
        debug_info = {
            "sample_id": sample.sample_id,
            "label": sample.label,
            "sample_data": sample.data.copy(),
            "clouds_atualizadas_info": [],
            "criou_nova_cloud": False,
            "nova_cloud_id": None,
            "merge_eventos": [],
        }
        
        # Mantém a lógica original: a segunda amostra entra direto na primeira cloud.
        if self.internal_counter == 1:
            cloud = self.data_clouds[0]       
            
            self.data_clouds[0] += sample  # So começa com a segunda amostra

            # Registra somente o necessário para o main.
            debug_info["clouds_atualizadas_info"].append(
                {
                    "cloud_id": cloud.id,
                    "pertinencia": float(cloud.pertinency),
                }
            )
        
        else:
            has_joined = False
            for cloud_index, cloud in enumerate(self.data_clouds):
                s_new = len(cloud) + 1
                
                new_mean = cloud.calculate_new_mean(
                    x=sample, old_mean=cloud.mean, s_new=s_new
                )
                
                new_variance = cloud.calculate_new_variance(
                    x=sample,
                    new_mean=new_mean,
                    old_mean=cloud.mean,
                    old_variance=cloud.variance,
                    s_new=s_new,
                )
                
                # calculo de exentricidade
                eccentricity = cloud.calculate_eccentricity( 
                        num_points=s_new,
                        mean=new_mean,
                        variance=new_variance,
                        point=sample.data,
             )
                
                #guarda a variavel que ja foi calculada.
                threshold = self.chebyshev_factor / s_new

                # Calculo do threshold
                if eccentricity <= (self.chebyshev_factor / s_new):
                    cloud = cloud + sample
                    self.data_clouds[cloud_index] = cloud
                    has_joined = True

                    # Registra só o necessário para o main.
                    debug_info["clouds_atualizadas_info"].append(
                        {
                            "cloud_id": cloud.id,
                            "pertinencia": float(cloud.pertinency),
                            "eccentricity": float(eccentricity),
                            "threshold": float(threshold),
                        }
                    )
            # Mantém a lógica original: se não entrou em nenhuma cloud, cria nova cloud.
            if not has_joined:
                new_candidate_cloud = DataCloud(x=sample)
                self.data_clouds.append(new_candidate_cloud)

                debug_info["criou_nova_cloud"] = True
                debug_info["nova_cloud_id"] = new_candidate_cloud.id

            # Somente verifica merge se a amostra entrou em alguma cloud, para evitar verificações desnecessárias
            else: 
                self.merge_clouds()
                debug_info["merge_eventos"] = self.last_merge_eventos.copy()
        
        # Mantém o incremento original.
        self.internal_counter += 1

        # Agora retorna as informações para o main imprimir.
        return debug_info

    def run(self, samples: list[Sample]) -> None:
        """
        Runs the AutoCloud algorithm on a list of samples.

        Parameters
        ----------
        samples : list[Sample]
            The list of samples to process.
        """
        for sample in samples:
            self.run_single_sample(sample)

    def print_summary(self) -> None:
        """
        Prints the summary of the AutoCloud algorithm,
        including the number of clouds and their details.
        """
        print(f"Total Clouds: {len(self.data_clouds)}")
        for index, cloud in enumerate(self.data_clouds):
            # print(f"Cloud {index + 1}: {cloud}")
            print(f"Cloud {index + 1}: ")
            print(cloud.print_label_distribution())
