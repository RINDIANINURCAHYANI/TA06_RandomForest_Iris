import pandas as pd

df = pd.read_csv("data/iris.csv")
df.columns = df.columns.str.strip()  # hapus spasi di awal/akhir
print("Nama kolom di CSV:", df.columns.tolist())
