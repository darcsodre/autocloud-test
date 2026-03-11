# Aqui é quem orquestra experimento (carregar dados, montar Samples, varrer parâmetro,
# executar stream, plotar).

import logging
import time
import numpy as np
import pandas as pd

from collections import Counter
from itertools import chain

from autocloud.autocloud import AutoCloud
from autocloud.datacloud import DataCloud
from autocloud.sample import Sample
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


def update_bar_progress(current, total, bar_length=50):
    """
    Atualiza a barra de progresso no console.

    Parameters
    ----------
    current : int
        O índice atual do progresso.
    total : int
        O total de itens a serem processados.
    bar_length : int, optional
        O comprimento da barra de progresso, por padrão 50.
    """
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    print(f"\r|{bar}| {percent:.2%}", end="\r")


def plot_clouds(data_clouds: list[DataCloud], point: Sample = None, **kwargs):
    max_feature_1 = kwargs.get("max_feature_1", None)
    min_feature_1 = kwargs.get("min_feature_1", None)
    max_feature_2 = kwargs.get("max_feature_2", None)
    min_feature_2 = kwargs.get("min_feature_2", None)
    max_feature_3 = kwargs.get("max_feature_3", None)
    min_feature_3 = kwargs.get("min_feature_3", None)
    colors = plt.cm.get_cmap("hsv", len(data_clouds) + 1)
    fig = plt.figure(figsize=(10, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    for idx, cloud in enumerate(data_clouds):
        points = np.array([sample.data for sample in cloud.points])
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=colors(idx),
            label=f"Cloud {cloud.id}",
        )
        ax.scatter(
            cloud.mean[0],
            cloud.mean[1],
            cloud.mean[2],
            color="black",
            marker="x",
            s=10,
        )  # Mark the mean
    if point is not None:
        ax.scatter(
            point.data[0],
            point.data[1],
            point.data[2],
            color="red",
            marker="^",  # Distinct marker: triangle
            s=70,
            label="New Sample",
        )
    ax.set_title("Data Clouds")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    # Set axes limits if min/max are provided
    if min_feature_1 is not None and max_feature_1 is not None:
        ax.set_xlim([min_feature_1, max_feature_1])
    if min_feature_2 is not None and max_feature_2 is not None:
        ax.set_ylim([min_feature_2, max_feature_2])
    if min_feature_3 is not None and max_feature_3 is not None:
        ax.set_zlim([min_feature_3, max_feature_3])
    ax.legend()
    plt.show()


def plot_2d_clouds(data_clouds: list[DataCloud], point: Sample = None, **kwargs):
    max_feature_1 = kwargs.get("max_feature_1", None)
    min_feature_1 = kwargs.get("min_feature_1", None)
    max_feature_2 = kwargs.get("max_feature_2", None)
    min_feature_2 = kwargs.get("min_feature_2", None)
    title = kwargs.get("title", "Data Clouds")
    colors = plt.cm.get_cmap("hsv", len(data_clouds) + 1)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    for idx, cloud in enumerate(data_clouds):
        points = np.array([sample.data for sample in cloud.points])
        ax.scatter(
            points[:, 0],
            points[:, 1],
            color=colors(idx),
            label=f"Cloud {cloud.id}",
        )
        ax.scatter(
            cloud.mean[0],
            cloud.mean[1],
            color="black",
            marker="x",
            s=10,
        )  # Mark the mean
    if point is not None:
        ax.scatter(
            point.data[0],
            point.data[1],
            color="red",
            marker="^",  # Distinct marker: triangle
            s=70,
            label="New Sample",
        )
    ax.set_title("Data Clouds")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    # Set axes limits if min/max are provided
    if min_feature_1 is not None and max_feature_1 is not None:
        ax.set_xlim([min_feature_1, max_feature_1])
    if min_feature_2 is not None and max_feature_2 is not None:
        ax.set_ylim([min_feature_2, max_feature_2])
    ax.set_title(title)
    ax.legend()
    plt.show()


def main():
    # Carregar dados
    # df = pd.read_csv("dataset_entropia_legitimo_malicioso.csv", sep=",")
    df = pd.read_csv("autocloud_synthetic_dataset_10000.csv", sep=",")

    df = df[
        [
            "feature_1",
            "feature_2",
            "feature_3",
            "label",
            
            # "frame.time_relative",
            ##"tcp.len",  # Usado no artigo, com valores diveros maiores que 0.dos
            #"tcp.flags_ack",
            # "tcp.flags_syn",
            #"tcp.flags_fin",  # Usado no artigo, aparentemente tem muitos 0.
            # "tcp.flags_urg",
            # "tcp.flags_ae",
            # "tcp.flags_cwr",
            # "tcp.flags_push",
            # "tcp.flags_res",
            # "tcp.flags_reset",
            # "tcp.flags_ece",
            ##"tcp.time_delta",  # Usado no artigo, mas tem valores muito baixos.
            ##"mqtt.msgtype",  # Usado no artigo, valores diversos.
            #"mqtt.dupflag",  # Usado no artigo, tem valores: True, False e numerico.
            # "mqtt.hdrflags", #valores binarios 
            ##"mqtt.len",  # Usado no artigo, valores diversos.
            # "mqtt.msg",#literalmente mensagens e strings.
            #"mqtt.qos",  # Usado no artigo, tem muito valores NaN,0 e 1. testar
            ##"mqtt.msgid", #numeros diveross, mas mais da metade é NaN. dos  testar
            # "velocidade", #mais da metade dos valores são NaN, testar
            ##"angulo", #mais da metade dos valores são NaN, testar
            #"attack_label",
        ]
    ]  # Substitui valores NaN por 0
    # ['legitimate', 'dos', 'malformed', 'falsedata']
    
    # Define qual ataque será analisado nesta execução.
    ATTACK_LABEL = "dos"  # Troque para qual ataque quiser analisar.
    #df = df[df["attack_label"].isin(["legitimate", ATTACK_LABEL])]
    #col_label = "attack_label"
    df = df[df["label"].isin(["legitimate", ATTACK_LABEL])]  #dataset sintetico.
    col_label = "label"    #dataset sintetico.
    #df = df[df["attack_label"].isin(["legitimate", "dos"])]
    col_values = df.columns.tolist()
    col_values.remove(col_label)
   
    print("=" * 80)
    
    for col in col_values:
        df[col] = df[col].fillna(0) #Substitui os NaN reais por 0.
        df[col] = df[col].replace("NaN", 0) #Aqui ele pega casos em que o CSV trouxe literalmente a string "nan", e substitui por 0.
    attack_labels_map_index = {i: label for i, label in enumerate(df[col_label])}

    features = df.values
    features = [
        Sample(sample_id=i, data=np.array(x[:-1]), label=attack_labels_map_index[i])
        for i, x in enumerate(features)
    ]
    
    ms = [2.5]
    for m in ms:
        print(f"Processing AutoCloud with Chebyshev parameter m={m}")
        auto_cloud = AutoCloud(chebyshev_parameter=m)
        start_time = time.time()
        bar_length = 10
        
        # Guarda a amostra anterior para comparar com a amostra atual.
        amostra_anterior = None
        
        for i, x in enumerate(features):
            #auto_cloud.run_single_sample(x)
            # Processa a amostra atual e retorna as informações calculadas nesta iteração.
            debug_info = auto_cloud.run_single_sample(x) 

            # Verifica se a amostra atual é um ataque e se a anterior era legítima.
            # Isso identifica exatamente o instante em que houve a mudança para ataque.
            mudou_para_ataque = (
                amostra_anterior is not None
                and amostra_anterior.label == "legitimate"
                and x.label == ATTACK_LABEL
            )

            # Se começou o ataque, imprime um destaque chamativo.
            if mudou_para_ataque:
                print("\n" + "!" * 120)
                print(f"!!! INÍCIO DO ATAQUE: legitimate -> {ATTACK_LABEL} | instante k = {i} !!!")
                print("!" * 120)

            # Imprime todas as amostras do filtro.
            print("\n" + "=" * 100)
            print(f"Instante k: {i}")
            print(f"sample_id: {debug_info['sample_id']}")
            print(f"label da amostra: {debug_info['label']}")
            print(f"valor da amostra: {debug_info['sample_data']}")

            # Mostra se a amostra atualizou uma ou mais clouds e a pertinência em cada uma.
            if debug_info["clouds_atualizadas_info"]:
                infos_clouds = []
                for info in debug_info["clouds_atualizadas_info"]:
                    trecho = f"cloud ID {info['cloud_id']}, grau de pertinência {info['pertinencia']:.6f}"

                    if "eccentricity" in info and "threshold" in info:
                        trecho += (
                            f", eccentricity {info['eccentricity']:.6f}, "
                            f"threshold {info['threshold']:.6f}"
                        )

                    infos_clouds.append(trecho)

                print("Clouds atualizadas: " + " ; ".join(infos_clouds))
            else:
                print("Clouds atualizadas: nenhuma")

            # Se a amostra não atualizou cloud nenhuma, ela criou uma nova.
            print(f"Criou nova cloud? {debug_info['criou_nova_cloud']}")
            if debug_info["criou_nova_cloud"]:
                print(f"ID da nova cloud: {debug_info['nova_cloud_id']}")

            # Se houve merge após atualização das clouds, mostra os IDs mergeados.
            if debug_info["merge_eventos"]:
                for id_a, id_b in debug_info["merge_eventos"]:
                    print(f"Merge entre cloud ID {id_a} e cloud ID {id_b}")
            else:
                print("Merge: não ocorreu")

            print("=" * 100)
            
            # plot_clouds(
            #     auto_cloud.data_clouds,
            #     max_feature_1=max_feature_1,
            #     min_feature_1=min_feature_1,
            #     max_feature_2=max_feature_2,
            #     min_feature_2=min_feature_2,
            #     max_feature_3=max_feature_3,
            #     min_feature_3=min_feature_3,
            # )
            # print()
            
            amostra_anterior = x
            
            if i % 1000 == 0:
                update_bar_progress(i + 1, len(features), bar_length)
            # if i % 1 == 0 and i > 0:  # Evita exibir na iteração 0
            # elapsed = time.time() - start_time
            # estimated_total = (elapsed / i) * 50000
            # remaining_time = estimated_total - elapsed
            # print(f"Processadas {i} amostras, {len(ac.clouds)} clouds criadas")
            # print(f"Qualidade atual: {ac.calculate_quality():.4f}")
            # print(f"Tempo estimado restante: {remaining_time:.2f}s")
        elapsed = time.time() - start_time
        auto_cloud.print_summary()
        # plot_clouds(
        #     auto_cloud.data_clouds,
        #     max_feature_1=max_feature_1,
        #     min_feature_1=min_feature_1,
        #     max_feature_2=max_feature_2,
        #     min_feature_2=min_feature_2,
        #     max_feature_3=max_feature_3,
        #     min_feature_3=min_feature_3,
        # )
        # plot_2d_clouds(
        #     auto_cloud.data_clouds,
        #     max_feature_1=max_feature_1,
        #     min_feature_1=min_feature_1,
        #     max_feature_2=max_feature_2,
        #     min_feature_2=min_feature_2,
        #     title=f"Data Clouds for m={m}",
        # )
        print(f"\nProcessamento concluído em {elapsed:.2f} segundos.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        # format="%(asctime)s %(levelname)s:%(message)s",
        datefmt=r"%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(
        logging.WARNING
    )  # Suppress matplotlib debug
    logging.getLogger("mpl_toolkits.mplot3d").setLevel(
        logging.WARNING
    )  # Suppress 3D toolkit debug
    # logging.getLogger("autocloud").setLevel(logging.DEBUG)  # Set AutoCloud debug level

    main()
