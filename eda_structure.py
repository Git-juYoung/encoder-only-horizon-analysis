import pandas as pd


df = pd.read_csv(
    "LD2011_2014.txt",
    sep=";",
    decimal=",",
    low_memory=False
)

df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])

power = df.iloc[:, 1:].astype("float32")
total_nan = power.isna().sum().sum()

df_2011 = df[df["timestamp"] < "2012-01-01"]
zero_clients = (df_2011.iloc[:, 1:] == 0).all()

clients_no_2011 = zero_clients.sum()
total_clients = power.shape[1]
loss_ratio = clients_no_2011 / total_clients

print(
    "Shape:", df.shape, "\n\n",
    "Column count:", len(df.columns), "\n",
    "First 5 columns:", df.columns[:5], "\n\n",
    "Start date:", df.iloc[:, 0].min(), "\n",
    "End date:", df.iloc[:, 0].max(), "\n\n",
    "Time interval check:\n", df.iloc[:, 0].diff().value_counts().head(), "\n\n",
    "Total NaN in power columns:", total_nan, "\n\n",
    "Clients with no data in 2011:", clients_no_2011, "\n\n",
    "Total clients:", total_clients, "\n",
    "Loss ratio:", round(loss_ratio * 100, 2), "%",
    "Head:\n", df.head(1)
)
