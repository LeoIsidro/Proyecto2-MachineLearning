from GMM import GMM,np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score

import pandas as pd

names = pd.read_csv("./clase.txt")

dataset = pd.read_csv("./dataset_tissue.txt", skiprows=1, header=None).drop(0, axis=1).transpose()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(dataset)

pca = PCA(n_components=66)
df_pca = pca.fit_transform(X_scaled)                   


X = df_pca
Y = names.x.values
n_components = 7


dic_clases={}
cont=0
for i in range(len(Y)):
    if Y[i] not in dic_clases:
        dic_clases[Y[i]] = (cont,[i])
        cont+=1
    else:
        dic_clases[Y[i]][1].append(i)


# Crea el modelo GMM
gmm = GMM(n_clusters=n_components)

# Ajusta el modelo a los datos
gmm.entrenamiento(X)

# Predice las etiquetas de los clusters para cada punto de datos
labels = gmm.prediccion(X)

print(labels)

silhouette_vals = silhouette_samples(X, labels)

# Crea una figura para visualizar la imagen de silueta
fig, ax = plt.subplots()
y_lower = 10

for i in range(n_components):
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()
    y_upper = y_lower + cluster_silhouette_vals.shape[0]

    color = plt.cm.viridis(float(i) / n_components)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * cluster_silhouette_vals.shape[0], str(i))
    y_lower = y_upper + 10


ax.set_xlabel("Coeficiente de Silueta")
ax.set_ylabel("Etiqueta del Clúster")
ax.set_title("Gráfico de Silueta para Clustering GMM",)

# Línea vertical para el valor promedio de silueta en todos los datos
average_silhouette_score = silhouette_score(X, labels)
ax.axvline(x=average_silhouette_score, color="red", linestyle="--")
ax.set_yticks([])  # Borrar etiquetas en el eje y

print("score:", average_silhouette_score)
plt.show()

"""
means = gmm.means
covariances = gmm.covariances

# Plotea los resultados
plt.figure(figsize=(10, 5))

# Datos de entrada
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis')
plt.title("Datos de entrada")

# Resultados del GMM
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("Resultados del GMM")

# Centros de las distribuciones gaussianas
for i, (mean, cov) in enumerate(zip(means, covariances)):
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convertir a grados
    ell = plt.matplotlib.patches.Ellipse(mean, v[0], v[1], 180 + angle, color='black')
    ell.set_clip_box(plt.gca().bbox)
    ell.set_alpha(0.5)
    plt.gca().add_artist(ell)

plt.show()"""

