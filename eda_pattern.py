import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "LD2011_2014.txt",
    sep=";",
    decimal=",",
    low_memory=False
)

df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])

power = df.iloc[:, 1:].astype("float32")

df_2011 = df[df["timestamp"] < "2012-01-01"]
zero_clients = (df_2011.iloc[:, 1:] == 0).all()
valid_columns = zero_clients[~zero_clients].index
power = power[valid_columns]

df["hour"] = df["timestamp"].dt.hour

print("Shape:", power.shape)

means = power.mean()
stds = power.std()

print("Mean range:", means.min(), "~", means.max())
print("Std range :", stds.min(), "~", stds.max())

plt.figure()
means.hist(bins=30)
plt.title("Household Mean Distribution")
plt.savefig("mean_distribution.png")
plt.close()

plt.figure()
stds.hist(bins=30)
plt.title("Household Std Distribution")
plt.savefig("std_distribution.png")
plt.close()

hourly_pattern = power.groupby(df["hour"]).mean()

plt.figure()
for i in range(5):
    plt.plot(hourly_pattern.iloc[:, i])

plt.title("Daily Pattern (First 5 Households)")
plt.xlabel("Hour")
plt.ylabel("Average Consumption")
plt.savefig("daily_pattern.png")
plt.close()

lag96_vals = power.apply(lambda x: x.autocorr(lag=96))
lag672_vals = power.apply(lambda x: x.autocorr(lag=672))

print("Lag96 mean:", lag96_vals.mean())
print("Lag96 std :", lag96_vals.std())
print()
print("Lag672 mean:", lag672_vals.mean())
print("Lag672 std :", lag672_vals.std())

plt.figure()
lag96_vals.hist(bins=30)
plt.title("Lag96 Distribution")
plt.savefig("lag96_distribution.png")
plt.close()

plt.figure()
lag672_vals.hist(bins=30)
plt.title("Lag672 Distribution")
plt.savefig("lag672_distribution.png")
plt.close()
