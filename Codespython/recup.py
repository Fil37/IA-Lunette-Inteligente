# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_csv("runs/detect/yolo11x_high_perf/results.csv")

# Dernière epoch entraînée
print(df.tail(1))

# Courbe loss
df[['epoch', 'train/box_loss', 'val/box_loss']].plot()

# mAP
df[['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']].plot()
