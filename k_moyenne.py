from sklearn.datasets import load_iris
dataset= load_iris()
import numpy as np
from random import choice
# x est notre liste de vecteur, chaque vecteur représantant une iris à l'aide de 4 attributs
x=dataset.data
print(type(x))
print(type(x[1]))
print(x[1])
print(x.dtype)
#fonction permettant de renvoyer k centroides de maniere aleatoire dans x
def init_centroide_alea(x,k):
    l=np.zeros((1,k))
    for i in range(1,k):
        l.put(i,choice(range(1,k)))
    return l
	
def dist_euclidienne (x,y):
    return 0

# définition de la fonction qui:
#  pour une précision epsilon, un nombre de cluster k, et un ensemble de donnée x
#  renvoie l'appartenance des éléments de x à un certains cluster ( indice entre 1 et k ) sous forme d'une liste
#   et renvoie la liste des centroides à la fin de l'algorithme ( sous forme de liste de vecteur a 4 attributs )
def k_moy(epsilon,k,x):
    # init des centroide aleatoirement
    liste_centroide= init_centroide_alea(x,k)
    for i in liste_centroide :
        print(i)
    print("i")
    #ces deux liste contiennent l'information sur l'appartenance de chaque élément de x a un cluster, et ce pour le rang l et l+1 (besoin pour faire la boucle while)
    #on les initialise à 0
    liste_app_l= len(x)*[0]
    liste_app_l_suivant= len(x)*[0]
    #notre indice qui indique combien d'itérations nous avons fait dans l'algorithme
    l=0
    #on initialise la condition de boucle a une valeur supérieur a epsilon pour rentrer au moins une foi dans la boucle
    condition_boucle=epsilon+1

    while condition_boucle> epsilon :
        #mise à jour de l'appartenance de chaque donnée à un cluster
        for i in range(1,len(x)) :
            dist_min=dist_euclidienne(x[i],liste_centroide)
            ind_min=1
            for j in range(1,k) :
                if dist_euclidienne(x[i],liste_centroide)<dist_min:
                    dist_min = dist_euclidienne(x[i],liste_centroide[j])
                    ind_min=j
            liste_app_l_suivant[i]=ind_min

        #mise à jour des centroides par la moyenne des données des clusters
        #for i in range(1,k) :
			
			#for j in range(1,len(x)) :
			#	if liste_app_l_suivant[j]=i :
					#cumul+=x[j]
					
        #
        l+=1
        #on met à jour la condition de boucle pour savoir si on fait une itération supplémentaire
        condition_boucle = 0
        for i in range(1, len(x)) :
            condition_boucle += abs(liste_app_l_suivant[i] - liste_app_l[i])
        condition_boucle = condition_boucle / len(x)

k_moy(0.5,5,x)