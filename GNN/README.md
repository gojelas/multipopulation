# Le modèle GT-A 

## Description
 Ce tuto décrit le modèle GT-A avec une matrice d'adjence construite à partir de la matrice de dégrés au lieu de la matrice angulaire.

## Base de données
La base Human Mortality Database est celle qui a été utilisée pour ce travail. Dans cette base, on a considéré au total 22 pays et pour chaque pays, on considère c'est juste le taux de mortalité qui est considéré.

## Construction de la matrice d'adjacence
L'algorithme de construction de la matrice d'adjence s'organise comme suit:
### Etape 1: Matrice $A_{long-lat}$
1. Récupérer des coordonnées géographiques de chacun des 22 pays sur Wikipédia. Sur Wikipédia, la longitude par exemple pour chaque pays, représente la moyenne des longitudes extrêmes des pays. C'est le cas également pour la latitude.

2. Calcul de la matrice $A_{long-lat}$, considérée comme la distance euclidienne entre les pays.

### Etape 2: Matrice $A_{DTW}$
- **Entrée :**
    - \( X \): Matrice du taux de mortalité de dimension [n_{pays}, n_{times}, n_{ages}]

- **Calculs :**
    1. Appliquer une ACP sur la dimension de l'âge et récupérer la première composante principale pour chaque pays. Cela transforme donc la matrice X en une matrice X' de taille [n_{pays}, n_{times}]
    2. 