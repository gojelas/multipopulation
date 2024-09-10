import numpy as np
import torch

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
    
    def DTW_Matrix(self, K: torch.Tensor):
        """
        K is n_countries x n_times matrix which contains first principal component of age
        """
        n_countries = K.shape[0]
        n_times = K.shape[1]
        dtw_matrix = torch.Tensor(n_countries,n_countries)
        for i in range(n_countries):
            for j in range(n_countries):
                dtw_matrix[i,j] = self.DTW(K[i,],K[j,])

        return dtw_matrix


class Adaptative_Matrix():
    def __init__(self) -> None:
        pass

    def


