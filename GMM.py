import numpy as np
from scipy.stats import multivariate_normal

class GMM:

    def __init__(self, n_clusters, n_iter=100, tolerancia=1e-6):
        self.n_cluster = n_clusters
        self.n_iter = n_iter
        self.tolerancia = tolerancia

    def inicializar_parametros(self,data):
        n,d =data.shape
        np.random.seed(0)

        #Elegimos un valor aleatorio para la media de cada cluster

        # Inicializar las medias aleatoriamente eligiendo n_components muestras aleatorias del conjunto de datos.
        random_indices = np.random.choice(n, self.n_cluster, replace=False)
        self.means = data[random_indices]

        # Inicializar las matrices de covarianza como matrices de identidad.
        self.covarianza = [np.eye(d) for _ in range(self.n_cluster)]


        #Inicializamos las probabilidades de cada cluster

        self.probabilidad = np.ones(self.n_cluster) / self.n_cluster

    def Ni(self,Yz,j):
        sum=0
        for i in range(len(Yz)):
            sum+=Yz[i][j]

        return sum
    
    def media(self,X, Yz,i):
        media_vector=[]
        for j in range(len(X[0])):
            media_vector.append(1/self.Ni(Yz,i)*np.dot(X[:,j],Yz[:,i]))

        return media_vector


    def desviacion_estandar(self,X,Yz,i):
        matriz_covarianzas = np.zeros((len(X[0]), len(X[0])))  
        for j in range(len(X)):
            matriz_covarianzas += Yz[j][i] * np.dot((X[j] - self.means[i]), (X[j] - self.means[i]))


        matriz_covarianzas=1/self.Ni(Yz,i)*matriz_covarianzas    

        return matriz_covarianzas

    def pi(self,Yz,i):
        return self.Ni(Yz,i)/len(Yz)
            


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
                #prob=multivariate_normal.pdf(X[j],mean=media[i],cov=desviacion_estandar[i])
                Yz[j][i]=pi[i]*prob

            Yz[j]/=np.sum(Yz[j])    


        return Yz

    def EM(self,data):

        self.inicializar_parametros(data)

        log_likelihoods = []

        print("means :",self.means)
        print("covarianza :",self.covarianza)
        print("probabilidad :",self.probabilidad)

        for i in range(self.n_iter):
            Yz=self.probabilidad_posteriori(data,self.means,self.covarianza,self.probabilidad)
            

            # Actualizamos los parametros

            Nk= np.sum(Yz,axis=0)
            for k in range(self.n_cluster):
                #self.means[k]=self.media(data,Yz,k)
                #self.covarianza[k]=self.desviacion_estandar(data,Yz,k)
                #self.probabilidad[k]=self.pi(Yz,k)
                self.means[k] = np.sum(data * Yz[:, k][:, np.newaxis], axis=0) / Nk[k]
                self.covarianza[k] = np.dot((data - self.means[k]).T, (data - self.means[k]) * Yz[:, k][:, np.newaxis]) / Nk[k]
                self.probabilidad[k] = Nk[k] / len(data)

            log_likelihood = np.sum(np.log(np.sum([self.probabilidad[k] * self.funcion_probabilidad(data[i], self.means[k], self.covarianza[k]) for k in range(self.n_cluster)], axis=0)))    
            log_likelihoods.append(log_likelihood)

            if i > 0 and abs(log_likelihood - log_likelihoods[-2]) < self.tolerancia:
                break

    def entrenamiento(self,data):
        self.EM(data)

    def prediccion(self,data):
        Yz=self.probabilidad_posteriori(data,self.means,self.covarianza,self.probabilidad)
        return np.argmax(Yz,axis=1)
    
    def obtener_parametros(self):
        return self.means,self.covarianza,self.probabilidad















