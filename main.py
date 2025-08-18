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
    df = pd.read_csv("dataset_entropia_legitimo_malicioso.csv", sep=",")

    # features = df[["tcp.flags.ack", "tcp.flags.syn", "tcp.flags.fin"]].values
    features = df[["tcp.flags.ack", "tcp.flags.syn"]].values
    features = [
        Sample(sample_id=i, data=np.array(x))
        for i, x in enumerate(features[75000:150000])
        # Sample(sample_id=i, data=np.array(x)) for i, x in enumerate(features[:402513])
    ]
    array_features = np.array([sample.data for sample in features])
    max_feature_1 = np.max(array_features[:, 0])
    min_feature_1 = np.min(array_features[:, 0])
    max_feature_2 = np.max(array_features[:, 1])
    min_feature_2 = np.min(array_features[:, 1])
    # max_feature_3 = np.max(array_features[:, 2])
    # min_feature_3 = np.min(array_features[:, 2])
    # ms = [0.45]
    # ms = np.linspace(0.1, 1.0, 10)
    ms = [0.5]
    for m in ms:
        print(f"Processing AutoCloud with Chebyshev parameter m={m}")
        auto_cloud = AutoCloud(chebyshev_parameter=m)
        start_time = time.time()
        bar_length = 10
        for i, x in enumerate(features):
            # plot_clouds(
            #     auto_cloud.data_clouds,
            #     point=x,
            #     max_feature_1=max_feature_1,
            #     min_feature_1=min_feature_1,
            #     max_feature_2=max_feature_2,
            #     min_feature_2=min_feature_2,
            #     max_feature_3=max_feature_3,
            #     min_feature_3=min_feature_3,
            # )
            auto_cloud.run_single_sample(x)
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
        # plot_clouds(
        #     auto_cloud.data_clouds,
        #     max_feature_1=max_feature_1,
        #     min_feature_1=min_feature_1,
        #     max_feature_2=max_feature_2,
        #     min_feature_2=min_feature_2,
        #     max_feature_3=max_feature_3,
        #     min_feature_3=min_feature_3,
        # )
        plot_2d_clouds(
            auto_cloud.data_clouds,
            max_feature_1=max_feature_1,
            min_feature_1=min_feature_1,
            max_feature_2=max_feature_2,
            min_feature_2=min_feature_2,
            title=f"Data Clouds for m={m}",
        )
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
