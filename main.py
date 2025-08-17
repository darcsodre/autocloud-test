import time
import numpy as np
import pandas as pd

from collections import Counter
from itertools import chain

from autocloud.autocloud import AutoCloud
from autocloud.datacloud import DataCloud
from autocloud.sample import Sample


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


def main():
    # Carregar dados
    df = pd.read_csv("dataset_entropia_legitimo_malicioso.csv", sep=",")

    features = df[["tcp.flags.ack", "tcp.flags.syn", "tcp.flags.fin"]].values
    features = [
        Sample(sample_id=i, data=np.array(x)) for i, x in enumerate(features[:402513])
    ]
    auto_cloud = AutoCloud(chebyshev_parameter=3.2)
    start_time = time.time()
    # bar_length = 50
    for i, x in enumerate(features):
        auto_cloud.run_single_sample(x)
        # update_bar_progress(i + 1, len(features), bar_length)
        # if i % 1 == 0 and i > 0:  # Evita exibir na iteração 0
        # elapsed = time.time() - start_time
        # estimated_total = (elapsed / i) * 50000
        # remaining_time = estimated_total - elapsed
        # print(f"Processadas {i} amostras, {len(ac.clouds)} clouds criadas")
        # print(f"Qualidade atual: {ac.calculate_quality():.4f}")
        # print(f"Tempo estimado restante: {remaining_time:.2f}s")
    elapsed = time.time() - start_time
    print(f"\nProcessamento concluído em {elapsed:.2f} segundos.")


if __name__ == "__main__":
    main()
