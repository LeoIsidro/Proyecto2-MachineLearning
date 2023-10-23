import numpy as np
from sklearn.cluster import KMeans

class GMM:

    def __init__(self, n_clusters, n_iter=100, tolerancia=1e-5):
        self.n_cluster = n_clusters
        self.n_iter = n_iter
        self.tolerancia = tolerancia

    def inicializar_parametros(self,data):

        # Inicializamos  las medias utilizando K-means
        kmeans = KMeans(n_clusters=self.n_cluster, random_state=0,n_init=100).fit(data)
        self.means = kmeans.cluster_centers_

        # Inicializamos las matrices de covarianza basada en la covarianza de los datos reales.
        covarianzas_data = np.cov(data, rowvar=False)
        self.covarianza = [covarianzas_data.copy() for _ in range(self.n_cluster)]


        #Inicializamos las probabilidades de cada cluster
        self.probabilidad = np.ones(self.n_cluster) / self.n_cluster
        

    def funcion_probabilidad(self,X,media,desviacion_estandar):
        d=X.shape[0]
        determinante_covarianza=np.linalg.det(desviacion_estandar)
        inversa_covarianza=np.linalg.inv(desviacion_estandar)

        factor1=1.0/((2*np.pi)**(d/2)*np.sqrt(determinante_covarianza))
        factor2=np.exp(-0.5 * np.dot(np.dot((X-media).T,inversa_covarianza),(X-media)))

        return factor1*factor2
    
    def probabilidad_posteriori(self,X,media,desviacion_estandar,pi):
        Yz=np.zeros((len(X),self.n_cluster))
        for j in range(len(X)):
            for i in range(self.n_cluster):
                prob=self.funcion_probabilidad(X[j],media[i],desviacion_estandar[i])
                Yz[j][i]=pi[i]*prob

            Yz[j]/=np.sum(Yz[j])    


        return Yz

    def EM(self,data):

        self.inicializar_parametros(data)


        # Inicializamos la lista de log-likelihoods
        log_likelihoods = []

        for i in range(self.n_iter):
            Yz=self.probabilidad_posteriori(data,self.means,self.covarianza,self.probabilidad)
            

            # Actualizamos los parametros, utilizamos newaxis para que podamos actualizar cada componente por separado, en lugar de actualizar la matriz completa

            Nk= np.sum(Yz,axis=0)
            for k in range(self.n_cluster):
                self.means[k] = np.sum(data * Yz[:, k][:, np.newaxis], axis=0) / Nk[k]
                cov_actualizar=np.dot((data - self.means[k]).T, (data - self.means[k]) * Yz[:, k][:, np.newaxis]) / Nk[k]
                cov_actualizar+=self.tolerancia * np.eye(data.shape[1])
                self.covarianza[k] = cov_actualizar
                self.probabilidad[k] = Nk[k] / len(data)

            # Calculamos el log-likelihood
            log_likelihood = np.sum(np.log(np.sum([self.probabilidad[k] * self.funcion_probabilidad(data[i], self.means[k], self.covarianza[k]) for k in range(self.n_cluster)], axis=0)))    
            log_likelihoods.append(log_likelihood)

            # Comprobamos si el algoritmo ha convergido
            if i > 0 and abs(log_likelihood - log_likelihoods[-2]) < self.tolerancia:
                break

    def entrenamiento(self,data):
        self.EM(data)

    def prediccion(self,data):
        # Calculamos la probabilidad posteriori para cada punto de datos
        Yz=self.probabilidad_posteriori(data,self.means,self.covarianza,self.probabilidad)
        # Asignamos la etiqueta de cluster a cada punto de datos
        return np.argmax(Yz,axis=1)















