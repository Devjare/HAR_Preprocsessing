import pandas as pd

fullds = pd.read_csv("./dataset/Phones_accelerometer.csv")
new_size = int(len(fullds)/20)
reduced_ds = fullds.head(new_size)
reduced_ds.to_csv("./dataset/reduced.csv")
