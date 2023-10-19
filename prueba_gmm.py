from GMM import GMM,np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from  sklearn.metrics import silhouette_score

# Genera un dataset de ejemplo
n_samples = 189
n_features = 5
n_components = 7
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_components, random_state=42)

# Crea el modelo GMM
gmm = GMM(n_clusters=n_components)

# Ajusta el modelo a los datos
gmm.entrenamiento(X)
print("Y :",y)

# Predice las etiquetas de los clusters para cada punto de datos
labels = gmm.prediccion(X)
print(len(labels))
print("labels :",labels)


# Obtiene los parámetros aprendidos del modelo
means = gmm.means
covariances = gmm.covarianza



# Plotea los resultados
plt.figure(figsize=(10, 5))

# Datos de entrada
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
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

plt.show()

