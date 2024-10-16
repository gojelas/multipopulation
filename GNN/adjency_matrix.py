import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt



class AgeReduction():
    def __init__(self) -> None:
        pass

    def age_reduction(self, X: torch.Tensor, n_components = 1):
        """
        ACP in age dimension
        Args: 
            X: np.array 3D matrix [i,t,x]
        Returns:
            X_pca: np.array 3D matrix [i,t,x'] represents matrix of x'-st ACP components
        """
        X_pca = np.ndarray((X.shape[0], X.shape[1], n_components))

        for i in range(X.shape[0]):
            pca = PCA(n_components= n_components)
            X_pca[i,:,:] = pca.fit_transform(X[i,:,:])
            print("The variance explained by principal components: ", pca.explained_variance_ratio_)

        X_pca  = X_pca.reshape((X_pca.shape[0],X_pca.shape[1]))
        return X_pca 
    

class DTW():
    def __init__(self) -> None:
        pass

    def DTW(self, A: list ,B: list):
        n = len(A)
        m = len(B)
        
        # Matrice d'initiation des distances
        distance = np.zeros((n+1, m+1))
        distance[0,:] = 10**10
        distance[:,0] = 10**10
        distance[0,0] = 0
        
        # Calcul du chemin optimal
        for i in range(1,n+1):
            for j in range(1,m+1):
                cost = (A[i-1] - B[j-1])**2
                distance[i,j] = cost + min(distance[i,j-1], distance[i-1,j], distance[i-1,j-1])

        #print(distance)
        return distance[n,m]
    
    def DTW_Matrix(self, X: torch.Tensor):
        """
        X is n_countries x n_times x n_ages tensors which contains mortality rates
        """

        # Age dimension reduction
        AR = AgeReduction()
        K = AR.age_reduction(X)

        # DTW_matrix 
        n_countries = X.shape[0]
        n_times = X.shape[1]
        dtw_matrix = torch.Tensor(n_countries,n_countries)
        for i in range(n_countries):
            for j in range(n_countries):
                dtw_matrix[i,j] = self.DTW(K[i,],K[j,])

        return dtw_matrix


class AdaptativeMatrix():
    def __init__(self) -> None:
        pass

    def adaptative_matrix(self, X):
        """
        X is n_countries x n_times x n_ages tensors which contains mortality rates
        """
        AR = AgeReduction()
        X_red = AR.age_reduction(X) # K is n_countries x n_times tensor

        # Application of KMeans for clustering
        KM = KMeans(n_clusters=2,random_state=0)
        KM.fit(X_red)
        clusters = KM.predict(X_red)


        # Méthode du coude : tester différents k et calculer l'inertie
        inertia = []
        score = []
        K = range(2, 10)
        for k in K:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X_red)
            inertia.append(kmeans.inertia_)

        # Calcul du silhouette score
            y_kmeans = kmeans.fit_predict(X_red)
            score.append(silhouette_score(X_red, y_kmeans))

        # Tracer l'inertie en fonction de k pour repérer le coude
        plt.figure(figsize=(8, 5))
        plt.plot(K, inertia, 'bx-')
        plt.plot(K, score, 'bx-')
        plt.xlabel('Nombre de clusters (k)')
        plt.ylabel('Inertie')
        plt.title('Méthode du coude pour déterminer le k optimal')
        plt.show()

        return clusters
