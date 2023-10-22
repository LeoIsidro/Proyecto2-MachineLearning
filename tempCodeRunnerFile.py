means = gmm.means
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

