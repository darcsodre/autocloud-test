import numpy as np

from .datacloud import DataCloud


class AutoCloud:
    def __init__(self, m=2.4):  # Valor padrão (default) caso o usuário não especifique
        self.m: float = m  # Parâmetro de sensibilidade (Eq. 9)
        self.clouds = []  # Lista de DataClouds
        self.classIndex = []  # Histórico de classificações
        self.k = 0  # Contador de amostras processadas
        self.id_history = []  # guarda o ID original da cloud de cada amostra
        self.merge_log = []  # histórico de fusões: (k, id_i, id_j, id_resultante)
        self.sample_registry = {}  # Dicionário para rastrear amostras processadas
        self.instantes_criacao = []  # NOVO: guarda (k, index) de toda cloud criada
        self.instantes_criacao_idfixo = []
        self.pertinencias = []  # guarda {cloud_index: μij} para cada amostra
        self.pertinencias_fixas = []  # NOVO: pertinência sincronizada com ID fixo

    def calculate_eccentricity(self, x, cloud):
        """Calcula a excentricidade normalizada (Eqs. 3 e 12)"""
        if cloud.n == 0 or cloud.variance == 0:
            return 1.0  # Máxima excentricidade se não houver dados

        term1 = 1 / cloud.n
        diff = cloud.mean - x
        term2 = (diff.T @ diff) / (cloud.n * cloud.variance)

        return term1 + term2  # Sem divisão por 2!

    def merge_clouds(self, cloud_protegida=None):
        """Fusão de clouds baseada na sobreposição das amostras (Eqs. 16-20)"""

        epsilon = 1e-6
        merged_indices = set()

        i = 0
        while i < len(self.clouds):
            if i in merged_indices:
                i += 1
                continue

            j = i + 1
            while j < len(self.clouds):
                if j in merged_indices or i in merged_indices:
                    j += 1
                    continue

                # NOVO2 Cálculo de intersecção
                set_i = set(self.clouds[i].points)
                set_j = set(self.clouds[j].points)

                intersection = set_i.intersection(set_j)
                inter_size = len(intersection)

                unique_i = len(set_i - set_j)
                unique_j = len(set_j - set_i)

                # Proteção para cloud recém-criada
                if (
                    self.clouds[i] is cloud_protegida
                    or self.clouds[j] is cloud_protegida
                ):
                    j += 1
                    continue

                # if inter_size > min(unique_i, unique_j):
                if (
                    inter_size > unique_i or inter_size > unique_j
                ):  # alterei para ficar igual do artigo.Duas clouds são fundidas se há sobreposição significativa de amostras.
                    id_i = self.clouds[i].id
                    id_j = self.clouds[j].id
                    self.merge_log.append((self.k, id_i, id_j, id_i))

                    print(
                        f"[k={self.k}] Fusão: cloud ID {id_j} absorvida por cloud ID {id_i}"
                    )

                    ni, nj = self.clouds[i].n, self.clouds[j].n

                    # Média (Eq. 18)
                    mean_i = self.clouds[i].mean
                    mean_j = self.clouds[j].mean
                    self.clouds[i].mean = (ni * mean_i + nj * mean_j) / (ni + nj)

                    # Variância (Eq. 20)
                    var_i = self.clouds[i].variance + epsilon
                    var_j = self.clouds[j].variance + epsilon
                    mean_diff_squared = np.linalg.norm(mean_i - mean_j) ** 2
                    self.clouds[i].variance = (
                        ni * var_i
                        + nj * var_j
                        + (ni * nj * mean_diff_squared) / (ni + nj)
                    ) / (ni + nj)

                    # Número de pontos
                    self.clouds[i].n = ni + nj

                    # NOVO2:Junta os pontos (fundamental para futuras fusões)
                    self.clouds[i].points = list(set_i.union(set_j))

                    # Atualiza classIndex
                    # Atualiza classIndex no modo fuzzy
                    for k_idx, idx_list in enumerate(self.classIndex):
                        new_idx_list = []
                        for idx in idx_list:
                            if idx == j:
                                new_idx_list.append(i)  # Cloud j foi absorvida por i
                            elif idx > j:
                                new_idx_list.append(
                                    idx - 1
                                )  # Ajuste dos índices após remoção de j
                            else:
                                new_idx_list.append(idx)
                        self.classIndex[k_idx] = new_idx_list

                        # ALTERADO: Atualiza a lista depois da fusao
                        if k_idx < len(self.pertinencias):
                            old_dict = self.pertinencias[k_idx]
                            new_dict = {}
                            for idx in old_dict:
                                if idx == j:
                                    new_dict[i] = max(
                                        new_dict.get(i, 0.0), old_dict[idx]
                                    )
                                elif idx > j:
                                    new_dict[idx - 1] = old_dict[idx]
                                else:
                                    new_dict[idx] = old_dict[idx]
                            self.pertinencias[k_idx] = new_dict

                    print(
                        f"Processadas {self.k} amostras, Fusão: cloud {j} absorvida por cloud {i}."
                    )

                    self.clouds.pop(j)
                    print(
                        f"[DEBUG:k={self.k}] Índices ativos após remoção: {[i for i in range(len(self.clouds))]}"
                    )

                    merged_indices.add(j)
                    continue

                j += 1
            i += 1
        for k_idx, idx_list in enumerate(self.classIndex):
            self.classIndex[k_idx] = sorted(set(idx_list))

    def calculate_quality(self):
        """Calcula a métrica de qualidade das fusões das clouds, baseada na Eq. 21 do artigo."""
        # Casos especiais
        # if len(self.clouds) < 2:
        if len(self.clouds) < 2 or len(self.pertinencias) == 0:
            return 0.0

        total = 0.0
        min_distance = float("inf")
        eps = 1e-10  # Valor pequeno para evitar divisão por zero

        """
        # Calcula a soma ponderada das variâncias
        for cloud in self.clouds:
            xi = self.calculate_eccentricity(cloud.mean, cloud)
            tau = max(0, 1 - xi)  #NOVO2:Correção direta da Eq. 21
            total += (tau**2) * (cloud.variance + eps)
        """
        # ALTERADO: usar pertinências μij registradas por amostra (Eq. 21)
        for i, cloud in enumerate(self.clouds):
            sigma_i = cloud.variance + eps
            soma_muij_quadrado = 0.0

            for pertinencia_dict in self.pertinencias:
                mu_ij = pertinencia_dict.get(i, 0.0)
                soma_muij_quadrado += (
                    mu_ij**2
                )  # Métrica ponderada com o quadrado da pertinência e variância de cada cloud.

            total += soma_muij_quadrado * sigma_i

        # Encontra a menor distância entre centróides de cloud distintos
        for i, c1 in enumerate(self.clouds):
            for j, c2 in enumerate(self.clouds):
                if i < j:
                    distance = np.linalg.norm(c1.mean - c2.mean)
                    if distance > eps:
                        min_distance = min(min_distance, distance)

        # Se todos cloud forem muito próximos (possivelmente ruins)
        if min_distance == float("inf"):
            return 0.0

        # Cálculo final com proteção contra zero no denominador
        N = len(self.pertinencias)
        denominator = N * max(min_distance, eps)
        return total / denominator

    def run(self, x):
        if self.k < 2000 or self.k % 5000 == 0:
            print(f"[k={self.k}] x = {x}")

        self.k += 1  # "estou processando mais uma amostra"
        # if not isinstance(x, np.ndarray):
        # x = np.array(x)  # Garante que x seja um numpy array

        # Passo 1: Primeira amostra → cria a primeira cloud
        if self.k == 1:
            self.clouds.append(DataCloud(x, min_var=1e-12))
            self.classIndex.append([0])  # ALTERADO → lista, pois agora é fuzzy
            self.pertinencias.append(
                {0: 1.0}
            )  # ALTERADO: pertinência máxima para a primeira cloud
            self.pertinencias_fixas.append({self.clouds[0].id: 1.0})  # NOVO
            print(
                f"Processadas {self.k} amostras, cloud {len(self.clouds) - 1} criada."
            )
            return

        # Passo 2: Segunda amostra → sempre atualiza a primeira cloud
        if self.k == 2:
            self.clouds[0].updateDataCloud(x)
            self.classIndex.append([0])  # ALTERADO → lista, pois agora é fuzzy
            self.pertinencias.append(
                {0: 1.0}
            )  # ALTERADO: pertinência máxima para a primeira cloud
            self.pertinencias_fixas.append({self.clouds[0].id: 1.0})  # NOVO
            print(f"Processadas {self.k} amostras, pertence à cloud 0.")
            return

        # Passo 3: A partir da terceira amostra → checa tipicidade com excentricidade

        threshold_factor = (self.m**2 + 1) / 2  # Cálculo movido para evitar redundância
        clouds_associadas = (
            set()
        )  # NOVO2 → Guarda todas as clouds que atendem o critério

        ecc_dict = {}  # Armazena eccentricidades antes da atualização
        for i, cloud in enumerate(self.clouds):
            ecc = self.calculate_eccentricity(x, cloud)
            threshold = (
                threshold_factor / cloud.n
            )  # Movido para dentro do loop para evitar cálculos repetidos

            if ecc <= threshold:
                ecc_dict[i] = ecc  # Salva ecc original
                cloud.updateDataCloud(x)
                clouds_associadas.add(i)

        # Se o a amostra pertence a uma ou mais clouds
        if clouds_associadas:
            lista_associada = sorted(list(clouds_associadas))
            # self.classIndex.append(lista_associada)
            # self.id_history.append([self.clouds[i].id for i in lista_associada])
            # ALTERADO: cálculo do grau de pertinência μij = 1 - ξ
            pertinencia_dict = {}
            for i in lista_associada:
                # ecc = self.calculate_eccentricity(x, self.clouds[i])
                ecc = ecc_dict.get(
                    i, 1.0
                )  # Usa ecc salvo ou valor alto (para segurança)
                mu = float(max(0, 1 - ecc))
                pertinencia_dict[i] = mu

            # Logica Fuzzy
            self.classIndex.append(lista_associada)
            print(
                f"Processadas {self.k} amostra, pertence às clouds {lista_associada}."
            )  # classIndex

            self.id_history.append([self.clouds[i].id for i in lista_associada])
            print(
                f"[k={self.k}] Atribuída às clouds IDs {[self.clouds[i].id for i in lista_associada]}"
            )  # IDfixo

            self.pertinencias.append(
                pertinencia_dict
            )  # ALTERADO: salva pertinências por amostra
            print(f"[k={self.k}] Pertinências (classIndex): {pertinencia_dict}")

            if pertinencia_dict:
                cloud_max = max(pertinencia_dict, key=pertinencia_dict.get)
                mu_max = pertinencia_dict[cloud_max]
                print(
                    f"[k={self.k}] Cloud com maior pertinência (classIndex): {cloud_max} (μ = {mu_max:.6f})"
                )

            # Separadamente, salvar pertinências com base em ID fixo
            pertinencia_fixa_dict = {
                self.clouds[i].id: pertinencia_dict[i] for i in lista_associada
            }
            self.pertinencias_fixas.append(pertinencia_fixa_dict)

            print(f"[k={self.k}] Pertinências (ID fixo): {pertinencia_fixa_dict}")

            if pertinencia_fixa_dict:
                cloud_max_id = max(pertinencia_fixa_dict, key=pertinencia_fixa_dict.get)
                mu_max_id = pertinencia_fixa_dict[cloud_max_id]
                print(
                    f"[k={self.k}] Cloud com maior pertinência (ID fixo): {cloud_max_id} (μ = {mu_max_id:.6f})"
                )

            return

        else:
            # Passo 5: Nenhuma cloud típica ou próxima → cria nova cloud
            self.clouds.append(DataCloud(x))
            new_index = len(self.clouds) - 1
            self.classIndex.append([new_index])
            # self.instantes_criacao.append((self.k, new_index))  # Salva o instante da criação da cloud
            self.pertinencias.append(
                {new_index: 1.0}
            )  # ALTERADO: pertinência máxima para nova cloud
            self.pertinencias_fixas.append(
                {self.clouds[-1].id: 1.0}
            )  # NOVO, grau de pertinencia ID fixo
            self.instantes_criacao.append(
                (self.k, new_index)
            )  # Garante que use o índice final após fusões
            self.instantes_criacao_idfixo.append(
                (self.k, self.clouds[-1].id)
            )  # salva também ID fixo
            self.id_history.append(self.clouds[-1].id)
            print(
                f"Processadas {self.k} amostras, cloud {new_index} criada."
            )  # classIndex
            print(
                f"[k={self.k}] Nova cloud criada com ID fixo {self.clouds[-1].id}"
            )  # lista ID

            # Verificação de pertinência = 1.0
            mu_classindex = self.pertinencias[-1].get(new_index)
            mu_idfixo = self.pertinencias_fixas[-1].get(self.clouds[-1].id)

            print(
                f"[VERIFICAÇÃO] Pertinência (classIndex): μ[{new_index}] = {mu_classindex}"
            )
            print(
                f"[VERIFICAÇÃO] Pertinência (ID fixo):    μ[{self.clouds[-1].id}] = {mu_idfixo}"
            )

            # Salvar referência da cloud recém-criada
            cloud_criada = self.clouds[-1]  # classIndex

            # Fusão sob demanda com proteção
            self.merge_clouds(
                cloud_protegida=cloud_criada
            )  # Para merge_clouds saber quando foi criada uma nova cloud.

            # Após fusão, verificar se a nova cloud mudou de índice usando classIndex
            if cloud_criada in self.clouds:
                updated_index = self.clouds.index(cloud_criada)
                if updated_index != new_index:
                    print(
                        f"Processadas {self.k} amostras, cloud {new_index} agora é cloud {updated_index} após fusões."
                    )  # classIndex
            self.classIndex[-1] = sorted(set(self.classIndex[-1]))
