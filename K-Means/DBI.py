import math
import numpy as np

# nc is number of clusters
# to be implemented without the use of any libraries (from the scratch)
 
def vectorDistance(v1, v2):
    """
    this function calculates de euclidean distance between two
    vectors.
    """
    return np.sqrt(np.sum(np.square(v1-v2)))
 
def compute_Si(i, x, clusters,nc):
    norm_c = nc
    s = 0
    for t in x[i]:
        s += vectorDistance(t,clusters)
    return s/norm_c
 
def compute_Rij(i, j, x, clusters, nc):
    Mij = vectorDistance(clusters[i],clusters[j])
    Rij = (compute_Si(i,x,clusters[i],nc) + compute_Si(j,x,clusters[j],nc))/Mij
    return Rij
 
def compute_Di(i, x, clusters, nc):
    list_r = []
    for j in range(nc):
        if i != j:
            temp = compute_Rij(i, j, x, clusters, nc)
            list_r.append(temp)
    return max(list_r)
 
def compute_DB_index(x, clusters, nc):
    sigma_R = 0.0
    for i in range(nc):
        sigma_R = sigma_R + compute_Di(i, x, clusters, nc)
    DB_index = float(sigma_R)/float(nc)
    return DB_index