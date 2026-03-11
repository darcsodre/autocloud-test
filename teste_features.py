'''import pandas as pd

df = pd.read_csv("dados_robo.csv", sep=",", encoding="latin1")

features_teste = ["tcp.len", "tcp.flags_ack", "tcp.flags_syn", "tcp.flags_fin", "tcp.flags_urg", "tcp.flags_ae",
                 "tcp.flags_cwr","tcp.flags_push"]
            
for feature in features_teste:
    print("\n" + "=" * 60)
    print("Feature:", feature)
    print("Shape:", df[feature].shape)
    print("\nContagem de valores:")
    print(df[feature].value_counts(dropna=False))
    print("\nValores únicos:")
    #print(df[feature].unique())'''

import pandas as pd

df = pd.read_csv("dados_robo.csv", sep=",", encoding="latin1")

print(df["mqtt.msgtype"].head(0))

features_teste = [
    "tcp.len",
    "tcp.flags_ack",
    "tcp.flags_syn",
    "tcp.flags_fin",
    "tcp.flags_urg",
    "tcp.flags_ae",
    "tcp.flags_cwr",
    "tcp.flags_push",
    "tcp.flags_res",
    "tcp.flags_reset",
    "tcp.flags_ece",
    "tcp.time_delta",
    "mqtt.msgtype",
    "mqtt.dupflag",
    "mqtt.hdrflags",
    "mqtt.len",
    "mqtt.msg",
    "mqtt.qos",
    "mqtt.msgid",
    "velocidade",
    "angulo",
    "vbat",
]

for feature in features_teste:
    print("\n" + "=" * 60)
    print("Feature:", feature)
    print("Shape:", df[feature].shape)
    print("\nContagem de valores:")
    print(df[feature].value_counts(dropna=False))

