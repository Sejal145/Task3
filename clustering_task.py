# clustering_task.py - reproduce clustering analysis
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv("customer_data.csv")
features = ["Age","AnnualIncome_k","SpendingScore"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Fit KMeans with chosen k
km = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = km.fit_predict(X_scaled)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df['PCA1'] = coords[:,0]; df['PCA2'] = coords[:,1]

df.to_csv("customer_data_clustered.csv", index=False)
print("Done.")
