import torch 
import torch.nn as nn
import torch.nn.functional as F 

# La méthode implémentée ici pour le réseau graphique est la méthode classique de message passing
# H^(l+1) = s(^D ^(-1/2) Â ^D ^(-1/2) H^(l) W^(l) + b^l)
# où ^D représente la matrice angulaire par paires de la matrice Â. 
# Remarquons qu'ici c'est pas le rôle de matrice de degré qui est donné à ^D mais un rôle plus pointu permettant 
# de capter des relations plus complexes.
# Mais dans un premier temps on va faire simple et utiliser la matrice de degré pour ^D
# et Â = A + I où A est la matrice d'adjacence et I est la matrice identité 
# A in R^(mxm) où m représente le nombre de noeuds
# ^D in R^(mxm)
# H^(l) in R^(m x T x d) d est considéré comme la dimension des caractéristiques, T est considéré comme la dimension des observations
# De ce faire, on applique une matrice de poids W^(l) in R^(d x d) pour la dimension des caractéristiques.
# b^(l) in R^(d x d)


# L'idée derrière la matrice de degré étant une normalisation de la matrice d'adjacence donc ça ne dérange pas 
# que la matrice d'adjacence soit valuée.

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias= True):
        super(GCNLayer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features,out_features))
        # Il peut y avoir des cas où le biais n'est pas considéré et pour cela:
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # pour initialiser les paramètres, la fonction d'initialisation étant définie par la suite
        self.init_parameters()

    def init_parameters(self):
        # Une initialisation avec la distribution normale de moyenne 0 et de d'écart-type 0.01 est fait
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.uniform_(self.bias, a=-0.1, b=0.1)


    def tensor_2D_3D_product(self, A, B):
        """
        Returns product of matrix A and B, where A represents adjacency matrix which is squared matrix and B represents
        features matrix which is 3D matrix.
        """
        #a_size = A.shape
        #b_size = B.shape
        #C = torch.FloatTensor(b_size)
        #for i in range(a_size[0]):
        #        for j in range(a_size[1]):
        #                C[i] += (A[i,j] * B[j])

        #return C.float()

        a_size = A.shape
        b_size = B.shape

        # S'assurer que les deux matrices soient du même format
        A = A.to(torch.float64)
        B = B.to(torch.float64)

        # Création d'une matrice de la même taille que B pour pouvoir enrégistrer le produit tensoriel là dans.
        C = torch.FloatTensor(b_size)
        for i in range(b_size[2]):
                C[:,:,i] = torch.mm(A,B[:,:,i])

        return C

    def tensor_3D_2D_product(self, X, W):
        """
        X is features matrix I x T x A
        W is weight matrix A x A' where W' is the expected dimension of features for output

        Returns:
        prod: matrix of I x T x A
        """
        x_shape = X.shape
        w_shape = W.shape
        new_features_shape = w_shape[1]
        prod = torch.Tensor(X.shape[0],X.shape[1],new_features_shape)

        #for j in range(w_shape[1]):
        #        for i in range(w_shape[0]):
        #                prod[:,:,j] += (X[:,:,i] * W[i,j])
        for i in range(prod.shape[0]):
                prod[i] = torch.mm(X[i], W)
        return prod
                

    # Définition de la fonction forward avec la matrice des caractéristiques des noeuds et d'djacence en argument
    def forward(self, node_feats, adj_matrix):
        """Forward.
        
        Args: 
            node_feats: represents the tensor of nodes features of graph. Shape = [n_nodes, n_times, n_ages]. Age dimension 
            represents features dimension
            adj_matrix: represents the adjency matrix. In this case (GT-A), it is the Hadamard product of A_{long-lat},
            A_{ada} and A_{DTW}. Shape = [n_nodes, n_nodes]

        """

        # Â
        n_countries = adj_matrix.shape[0]
        adj_matrix = adj_matrix + torch.eye(n_countries)
        
        # ^D
        nodes_degrees = adj_matrix.sum(dim=-1, keepdims=True)

        # Ã = ^D ^(-1/2) Â ^D ^(-1/2) 
        adj_hat = adj_matrix / nodes_degrees

        # out = ÃH^(l)W^(l) + b
        out = self.tensor_2D_3D_product(adj_hat,node_feats)
        out = self.tensor_3D_2D_product(out,self.weight)
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
class GCNMultiLayer(nn.Module):
    def __init__(self, in_features, hid_features, out_features,bias=True):
        super(GCNMultiLayer,self).__init__()
        self.gcn1 = GCNLayer(in_features, hid_features)
        self.gcn2 = GCNLayer(hid_features, out_features)

    
    def forward(self, nodes_features, adj_matrix):
        H = self.gcn1(nodes_features, adj_matrix)
        H = F.relu(H)
        H = self.gcn2(H, adj_matrix)

        return H
    

class EncoderLayer(nn.Module):
    def __init__(self, n_features=100, head=5, feed_forward_dim = 64):
        # d_model représente le nombre de features dans X et dans notre cas c'est la taille de la dimension de l'âge
        super(EncoderLayer,self).__init__()
        # Ici on suppose que Q, K, V in R^(T x d_model)
        self.d_model = n_features
        self.d_K = n_features
        self.d_V = n_features
        self.head = head
        
        #Définition des poids
        self.weight_Q = nn.ParameterList([nn.Parameter(torch.FloatTensor(n_features, self.d_K)) for _ in range(head)])
        self.weight_K = nn.ParameterList([nn.Parameter(torch.FloatTensor(n_features, self.d_K)) for _ in range(head)])
        self.weight_V = nn.ParameterList([nn.Parameter(torch.FloatTensor(n_features, self.d_V)) for _ in range(head)])
        self.weight_multi = nn.Parameter(torch.FloatTensor(n_features * head, n_features))

        self.ffn = nn.Sequential(
            nn.Linear(n_features, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, n_features)
        )
        self.layer_norm1 = nn.LayerNorm(n_features)
        self.layer_norm2 = nn.LayerNorm(n_features)
    
        # Initialisation des poids d'attention
        self.initiate_parameters()
    
    def initiate_parameters(self):
        # Initialisation avec Xavier pour chaque tête
        for i in range(self.head):
            nn.init.xavier_normal_(self.weight_K[i])
            nn.init.xavier_normal_(self.weight_Q[i])
            nn.init.xavier_normal_(self.weight_V[i])
        nn.init.xavier_normal_(self.weight_multi)


    def forward(self,X):
        """Forward.
        Args:
            X: represents output of GCN but the case for each is dealt with here. X in R^(n_times x n_ages)

        """
        self.n_times = X.shape[0]
        

        # enregistrement de la première attention calculée
        multi_head_Attention = []

        #####multi_head_Attention = torch.FloatTensor(X.shape)
        for i in range(self.head):
            Q_i = torch.mm(X, self.weight_Q[i])
            K_i = torch.mm(X, self.weight_K[i])
            V_i = torch.mm(X, self.weight_V[i])
            Attention_i = torch.mm(F.softmax(torch.mm(Q_i, K_i.T)/(self.d_K ** 0.5)), V_i)
            multi_head_Attention.append(Attention_i)
        
        multi_head_Attention = torch.concat(multi_head_Attention, dim=1)


        # multi-head = concat(head_0,...,head_h) W^(0) où W^(0) in R^(head*n_features, n_features)
        result = torch.matmul(multi_head_Attention, self.weight_multi)

        # X = LayerNorm(X + self_Multi_head(X))
        # La normalisation pour chaque exemple x(pour chaque t dans notre cas) est faite sur la dimension des features, d'où la 
        # présence de d_model en argument. Pour être plus clair, on considère la moyenne et l'écart-type sur toutes les
        # features pour chaque exemple pris: ^x = (x-u)/s où u est la moyenne sur toutes les features et s l'écart-type
        X = X + result
        X = self.layer_norm1(X)
        ffn_x = self.ffn(X)
        X = X + ffn_x
        X = self.layer_norm2(X)

        return X 

class Encoder(nn.Module):
    def __init__(self, n_features= 100, n_predictions= 9, hid_linear_part_dim = 200):
        super(Encoder, self).__init__()
        self.n_predictions = n_predictions
        self.n_features = n_features
        self.d_model = n_features
        self.encod1 = EncoderLayer(n_features=self.n_features)
        self.encod2 = EncoderLayer(n_features=self.n_features)
        self.encod3 = EncoderLayer(n_features=self.n_features)

        self.linear_part = nn.Sequential(
            nn.Linear(self.n_features, hid_linear_part_dim),
            nn.Linear(hid_linear_part_dim, self.n_features)
        )

        self.n_countries = None
        self.n_times = None


    def forward(self, X):
        """Forward_Encoder.
        Args:
            X: Tensor de taille n_countries x n_times x n_ages. which represents the mortality matrix by country.
        """

        if self.n_countries is None:
            self.n_countries = X.shape[0]
        if self.n_times is None:
            self.n_times = X.shape[1]
        

        # Initialisation de la matrice de prévisions
        y = torch.FloatTensor(self.n_countries, self.n_predictions, self.n_features)
        #y = torch.zeros(self.n_countries, self.n_times, self.n_features)


        # Tel que l'encoder est défini c'est à appliquer pour chaque pays. Donc en partant de la matrice X, sortie du réseau
        # de neurone graphique, X.shape = (n_countries, n_times, n_ages). Donc on applique l'encoder pour chaque pays de X 
        # pour évaluer uniquement la dépendance spatiale.
        for country in range(self.n_countries):
            x = self.encod1(X[country]) # x.shape = [n_times, n_features]
            x = self.encod2(x)
            x = self.encod3(x)

            
            # La partie linéaire pour faire les prévisions pour chaque pays. La prévision étant faite jusqu'à l'horizon 
            # de 10 années.
            current_input = x
            predictions = []
            for _ in range(self.n_predictions):
                next_pred = self.linear_part(current_input[-1])
                predictions.append(next_pred.squeeze(0)) # enregistrer la prediction

                current_input = torch.cat([current_input, next_pred.unsqueeze(0)],dim=0)

            y[country] = torch.stack(predictions, dim=0)

            #y[country] = x.clone()
        
        return y






class GTA_Model(nn.Module):
    def __init__(self, in_features=100, hid_features=60,n_predictions=10, out_features=100):
        super(GTA_Model, self).__init__()
        self.gcn = GCNMultiLayer(in_features, hid_features, out_features)
        self.encoder = Encoder(n_predictions=n_predictions)
    
    def forward(self, X, A):
        X = self.gcn(X ,A)
        y = self.encoder(X)

        return y
        











