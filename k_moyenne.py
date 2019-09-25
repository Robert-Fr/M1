import numpy as np
from random import choice
from sklearn.datasets import load_iris
import copy as c
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
dataset = load_iris()
# x est notre liste de vecteur, chaque vecteur représantant une iris à l'aide de 4 attributs
x = dataset.data


# fonction permettant de renvoyer k centroides de maniere aleatoire dans x
def init_centroide_alea(x, k):
    l = []
    for i in range(0, k):
        l.append(x[choice(range(0, len(x)))])
        #print(i+1)
        #print(l[i])
    return l



# définition de la fonction qui:
#  pour une précision epsilon, un nombre de cluster k, et un ensemble de donnée x
#  renvoie l'appartenance des éléments de x à un certains cluster ( indice entre 1 et k ) sous forme d'une liste
#   et renvoie la liste des centroides à la fin de l'algorithme ( sous forme de liste de vecteur a 4 attributs )
def k_moy(epsilon, k, x):
    # init des centroide aleatoirement
    liste_centroide = init_centroide_alea(x, k)

    # ces deux liste contiennent l'information sur l'appartenance de chaque élément de x a un cluster, et ce pour le rang l et l+1 (besoin pour faire la boucle while)
    # on les initialise à 0
    liste_app_l = len(x)*[0]
    liste_app_l_suivant = len(x)*[0]
    # notre indice qui indique combien d'itérations nous avons fait dans l'algorithme
    l = 0
    # on initialise la condition de boucle a une valeur supérieur a epsilon pour rentrer au moins une fois dans la boucle
    condition_boucle = epsilon + 1.

    while condition_boucle > epsilon:
        #on met à jour notre liste d'appartenance 
        liste_app_l=c.deepcopy(liste_app_l_suivant)
        # mise à jour de l'appartenance de chaque donnée à un cluster
        for i in range(0, len(x)): # pour tout les éléments de x
            dist_min = np.linalg.norm(x[i]-liste_centroide[0])
            ind_min = 0
            for j in range(0, k): #pour tout les élément de la liste de cluster
                if np.linalg.norm(x[i]-liste_centroide[j]) < dist_min :# si le point appartient au cluster numéro j
                    dist_min = np.linalg.norm(x[i]-liste_centroide[j])
                    ind_min = j
            liste_app_l_suivant[i]=ind_min # on stocke le centroide le plus proche
            #print(liste_app_l_suivant[i])

        # mise à jour des centroides par la moyenne des données des clusters
        for i in range(0,k) :
             compt=0
             som=[0.,0.,0.,0.]
             for j in range(0,len(x)) :
                if liste_app_l_suivant[j] == i :
                    som[0] += x[j][0]# le premier atribut du jème vecteur de x
                    som[1] += x[j][1]# le deuxième
                    som[2] += x[j][2]# le troisième
                    som[3] += x[j][3]# le quatrième 
                    compt+=1
                    #print(som)
             liste_centroide[i]=[som[0]/compt,som[1]/compt, som[2]/compt,som[3]/compt]
             #print(liste_centroide[i])


        #
        l += 1
        # on met à jour la condition de boucle pour savoir si on fait une itération supplémentaire
        condition_boucle = 0.
        for i in range(0, len(x)):
            condition_boucle += abs(liste_app_l_suivant[i] - liste_app_l[i])
        condition_boucle = condition_boucle / len(x)
    
    print("Algorithme terminé avec " + str(l) + " itérations.")
    return liste_app_l_suivant

assignements=k_moy(0.05, 5, x)
print(assignements)
pca = PCA(n_components=2)
x_r = pca.fit(x).transform(x)
k=len(set(assignements))
print(k)
target_names=range(k)
plt.figure()
for i,target_names in zip(range(k),target_names):
    plt.scatter(x_r[assignements == i, 0 ], x_r[assignements == i, 1], label=target_names)
plt.legend(loc='best')
plt.title('partition_obtenue_par_k-means_sur_la_collection_Iris')
plt.show()