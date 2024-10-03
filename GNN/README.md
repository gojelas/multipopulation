# Le modèle GT-A 

## Description
 Ce tuto décrit le modèle GT-A avec une matrice d'adjence construite à partir de la matrice de dégrés au lieu de la matrice angulaire. Dans le cygle GT-A, G est mis pour graphique, T pour Transformer, et A pour Adaptative. Dans ce problème, on considère l'ensemble des pays comme un graphe dont les noeuds sont les pays, représentés par leur matrice de taux de mortalité. De ce faire, on arrive à mettre à jour l'information(matrice de taux de mortalité) de chaque pays en considérant l'information de leur voisinage. Le voisinage est défini dans ce cas à partir d'une matrice d'adjacence construite à partir des coordonnées géographiques et des taux de mortalité de chaque pays.

## Base de données
La base Human Mortality Database est celle qui a été utilisée pour ce travail. Dans cette base, on a considéré au total 22 pays et pour chaque pays, on considère c'est juste le taux de mortalité qui est considéré.

## Construction de la matrice d'adjacence
L'algorithme de construction de la matrice d'adjence s'organise comme suit:
### Etape 1: Matrice $A_{long-lat}$
    1. Récupérer des coordonnées géographiques de chacun des 22 pays sur Wikipédia. Sur Wikipédia, la longitude par exemple pour chaque pays, représente la moyenne des longitudes extrêmes des pays. C'est le cas également pour la latitude.

    2. Calcul de la matrice $A_{long-lat}$, considérée comme la distance euclidienne entre les pays.

### Etape 2: Matrice $A_{DTW}$
- **Entrée :**
    - $X$: Matrice du taux de mortalité de taille $[n_{pays}, n_{times}, n_{ages}]$

- **Calculs :**
    1. Appliquer une ACP sur la dimension de l'âge et récupérer la première composante principale pour chaque pays. Cela transforme donc la matrice X en une matrice X' de taille $[n_{pays}, n_{times}]$
    2. Calculer la matrice de distance Dynamic Time Warping (DTW) entre les différentes séries temporelles de composantes principales de chaque pays.

- **Sortie :**
    - $A_{DTW}$: Matrice de distance DTW de taille $[n_{pays}, n_{pays}]$

### Etape 3: Matrice $A_{ada}$
- **Entrée :** 
    - $X'$: Matrice de composante principale par pays de taille $[n_{pays}, n_{times}]$ obtenue à l'étape 2.

- **Calculs :**
    1. Appliquer la méthode des K-means sur la matrice X' pour classifier les composantes en n clusters.
    2. Utiliser la méthode de coude et de silhouette pour déterminer le nombre n de clusters.
    3. Choisir une liste de paramètres $(\alpha_1, \dots, \alpha_n, \beta)$; où $\alpha_i$ représente la correlation entre les pays d'une même classe et $\beta$ celle des pays de classes différentes. Cela sous-entend que $\beta$ est très proche de 0 (supposé =0.05 ici). Cela évite d'avoir des 0 dans la matrice $A_{ada}$
    4. Construire $A_{ada}$ à partir de ces paramètres.

_ **Sortie :**
    - $A_{ada}$: Matrice de taille  contenant le clustering des pays $[n_{pays}, n_{pays}]$

### Etape 4: Matrice d'adjacence:
    - Faire le produit de Hadamard de chacune des trois matrices précédentes:
    $A = A_{long-lat} * A_{DTW} * A_{ada}$


## Propapagation de message (Réseau de neurone graphique):
Une fois la matrice d'adjacence calculée, il faut mettre à jour la matrice de taux de mortalité au niveau de chaque pays en considérant le voisinage de ces derniers. Cela se fait comme suit:
$H^{l+1} = \sigma(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^l W^{l} + b^l) $
où:
- $\hat{A} = A + I$ cela permet de prendre en compte l'information au niveau du noeud cible lors de la mise à jour; $I$ étant la matrice identité
- $\hat{D}$ réprésente la matrice de degré de $\hat{A}$ dans ce cas. Mais dans l'étude de base, cela représente la matrice angulaire entre les lignes ou colonnes de $\hat{A}$.
- $H^l$ représente l'information au niveau des noeuds à l'itération $l$. $H^0 = X$. Rappelons que la dimension de l'âge est considérée comme celle des features et la dimension du temps comme celle des observations.
- $W^l \in \mathbb{R}^{d_l \times d_{l+1}}$ représente la matrice de poids; où $d_l$ est la dimension des features pour $H^l$.
- $b^l \in \mathbb{R}^{d_l}$ représente la matrice de biais.