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
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features,out_features))
        
        # Il peut y avoir des cas où le biais n'est pas considéré et pour cela:
        if bias:
            self.weight = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # pour initialiser les paramètres, la fonction d'initialisation étant définie par la suite
        self.init_parameters()

    def init_parameters(self):
        # Une initialisation avec la distribution normale de moyenne 0 et de d'écart-type 0.01 est fait
        nn.init.normal_(self.weight, mean=0.0, std=0.01)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0.0, std=0.01)

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
        out = torch.mm(adj_hat, torch.mm(node_feats,self.weight))
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
class GCNMultiLayer(nn.Module):
    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hid_features)
        self.gcn2 = GCNLayer(hid_features, out_features)

    
    def forward(self, nodes_features, adj_matrix):
        H = self.gcn1(nodes_features, adj_matrix)
        H = F.relu(H)
        H = self.gcn2(H, adj_matrix)

        return H
    

class EncoderLayer(nn.Module):
    def __init__(self,n_features, head=5, feed_forward_dim = 64, hid_linear_part_dim = 64,n_predictions = 10):
        # d_model représente le nombre de features dans X et dans notre cas c'est la taille de la dimension de l'âge
        super(EncoderLayer).__init__()
        # Ici on suppose que Q, K, V in R^(T x d_model)
        self.d_model = n_features
        self.d_K = n_features
        self.d_V = n_features
        self.head = head
        self.n_predictions = n_predictions
        self.weight_Q = nn.Parameter(torch.FloatTensor(n_features, self.d_K))
        self.weight_K = nn.Parameter(torch.FloatTensor(n_features, self.d_K))
        self.weight_V = nn.Parameter(torch.FloatTensor(n_features, self.d_V))
        self.weight_multi = nn.Parameter(torch.FloatTensor(self.head * n_features, n_features))

        self.ffn = nn.Sequential(
            nn.Linear(n_features, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, n_features)
        )

        self.linear_part = nn.Sequential(
            nn.Linear(n_features, hid_linear_part_dim),
            nn.Linear(hid_linear_part_dim, n_predictions * n_features)
        )
    
    def initiate_parameters(self):
        nn.init.normal_(self.weight_K, mean=0.0, std=0.01)
        nn.init.normal_(self.weight_Q, mean=0.0, std=0.01)
        nn.init.normal_(self.weight_V, mean=0.0, std=0.01)
    def forward(self,X):
        """Forward.
        Args:
            X: represents output of GCN. X in R^(n_countries x n_times x n_ages)

        """

        # initialisation des points d'attention 
        self.initiate_parameters()

        Q_0 = torch.mm(X, self.weight_Q)
        K_0 = torch.mm(X, self.weight_K)
        V_0 = torch.mm(X, self.weight_V)
        Attention_0 = torch.mm(F.softmax(torch.mm(Q_0, K_0.T)/((self.d_K)**(0.5))), V_0)

        # enregistrement de la première attention calculée
        multi_head_Attention = Attention_0

        multi_head_Attention = torch.FloatTensor(X.shape)
        for i in range(1,self.head):
            self.initiate_parameters()

            Q_i = torch.mm(X, self.weight_Q)
            K_i = torch.mm(X, self.weight_K)
            V_i = torch.mm(X, self.weight_V)
            Attention_i = torch.mm(F.softmax(torch.mm(Q_i, K_i.T)/((self.d_K)**(0.5))), V_i)
            torch.concat((multi_head_Attention, Attention_i),dim=1)
        

        # initialisation de la matrice de poids du multi-head attention
        torch.init.normal_(self.weight_multi, mean=0.0, std=0.01)

        # multi-head = concat(head_0,...,head_h) W^(0) où W^(0) in R^(head*n_features, n_features)
        multi_head_Attention = torch.mm(multi_head_Attention, self.weight_multi)

        # X = LayerNorm(X + self_Multi_head(X))
        # La normalisation pour chaque exemple x(pour chaque t dans notre cas) est faite sur la dimension des features, d'où la 
        # présence de d_model en argument. Pour être plus clair, on considère la moyenne et l'écart-type sur toutes les
        # features pour chaque exemple pris: ^x = (x-u)/s où u est la moyenne sur toutes les features et s l'écart-type
        X = F.layer_norm(X + multi_head_Attention, self.d_model)
        ffn_x = self.ffn(X)
        X = F.layer_norm(X + ffn_x)

        # La partie linéaire pour faire les prévisions
        X = self.linear_part(X)
        X = X.view(*X.shape[:-1], self.n_predictions, self.d_model)

        return X
    

class Encoder(nn.Module):
    def __init__(self, ):
        super(Encoder).__init__()
    
    def forward(self, X):
        """Forward_Encoder.
        Args:
            X: Tensor de taille n_countries x n_times x n_ages. which represents the 
        """









