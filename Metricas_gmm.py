from GMM import GMM,np

import numpy as np
import matplotlib.pyplot as plt


import pandas as pd

names = pd.read_csv("./clase.txt")

dataset = pd.read_csv("./dataset_tissue.txt", skiprows=1, header=None).drop(0, axis=1).transpose()

from sklearn.decomposition import PCA

pca = PCA(n_components=70)
df_pca = pca.fit_transform(dataset)                   


X = df_pca
Y = names.x.values
k = 7
    

# Crea el modelo GMM
gmm = GMM(n_clusters=k)

# Ajusta el modelo a los datos
gmm.entrenamiento(X)

# Predice las etiquetas de los clusters para cada punto de datos
labels = gmm.prediccion(X)

#Metrica Matriz similitud


"""from sklearn.metrics import pairwise_distances
# # Ordenar los datos según su etiqueta de cluster
sorted_data = X[np.argsort(labels)]
# # Calcular la matriz de similitud
similarity_matrix = pairwise_distances(sorted_data, metric='euclidean')

# Crea una figura para visualizar el gráfico de calor
plt.figure(figsize=(8, 6))
plt.imshow(similarity_matrix, cmap='jet', interpolation='none')
plt.colorbar()

# Añade etiquetas a los ejes (opcional)
plt.xlabel('Puntos')
plt.ylabel('Puntos')
plt.title('Matriz de Similitud para GMM')

# Muestra el gráfico de calor
plt.show()"""


# Metrica Rand index

"""def asignar_etiqueta(labels, Y=Y):
    labels_reales = []

    dic_labels={
        'kidney': 1,
        'hippocampus': 5,
        'cerebellum': 2,
        'colon': 4,
        'liver': 0,
        'endometrium': 3,
        'placenta': 6
    }

    for i in range(len(labels)):
        labels_reales.append(dic_labels[Y[i]])

    labels_reales = np.array(labels_reales)     

    return labels_reales


from sklearn.metrics import adjusted_rand_score

labels_etiquetas = asignar_etiqueta(labels)


print(labels)
print(labels_etiquetas)
rand_index = adjusted_rand_score(labels_etiquetas, labels)
print("rand_index:", rand_index)"""


# Metrica silueta

from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score

silhouette_vals = silhouette_samples(X, labels)

# Crea una figura para visualizar la imagen de silueta
fig, ax = plt.subplots()
y_lower = 10

for i in range(k):
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()
    y_upper = y_lower + cluster_silhouette_vals.shape[0]

    color = plt.cm.viridis(float(i) / k)
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



# Visualiza los puntos

"""means = gmm.means
covariances = gmm.covarianza

# Plotea los resultados
plt.figure(figsize=(10, 5))

# Datos de entrada
plt.subplot(1, 2, 1)
plt.scatter(X[:, 7], X[:, 8], c=labels_etiquetas, cmap='viridis')
plt.title("Datos de entrada")

# Resultados del GMM
plt.subplot(1, 2, 2)
plt.scatter(X[:, 7], X[:, 8], c=labels_prediccion, cmap='viridis')
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

plt.show()

"""