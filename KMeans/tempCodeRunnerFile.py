def kmeans(data, K, maxIter):
    m, dim = np.shape(data)
    k = K
    GbestScore, GbestPositon, Curve = BOAK(pop, k, data)
    # GbestPositon = [[4.3, 2, 3.10221011, 0.1,4.3, 2,1,0.1,4.3,2,1,0.1]]
    U = GbestPositon[0]
    def _distance(p1,p2):
        """
        Return Eclud distance between two points.
        p1 = np.array([0,0]), p2 = np.array([1,1]) => 1.414
        """
        return np.sqrt(np.sum(np.square(np.array(p1)-np.array(p2))))

    def _rand_center(data,k):
        """Generate k center within the range of data set."""
        n = data.shape[1] # features
        centroids = np.zeros((k,n)) # init with (0,0)....
        for i in range(n):
            dmin, dmax = np.min(data[:,i]), np.max(data[:,i])
            centroids[:,i] = dmin + (dmax - dmin) * np.random.rand(k)
        return centroids
    def _converged(centroids1, centroids2):
        
        # if centroids not changed, we say 'converged'
         set1 = set([tuple(c) for c in centroids1])
         set2 = set([tuple(c) for c in centroids2])
         return (set1 == set2)
        
    dbsList = [float('inf')]
    n = data.shape[0] # number of entries
    centroids = np.zeros([k, dim])
    for i in range(k):
        centroids[i] = U[i *  dim: (i +1)* dim]
    label = np.zeros(n,dtype=np.int) # track the nearest centroid
    assement = np.zeros(n) # for the assement of our model
    converged = False
    old_centroids = np.copy(centroids)
    for i in range(n):
        # determine the nearest centroid and track it with label
        min_dist, min_index = np.inf, -1
        for j in range(k):
            dist = _distance(data[i],centroids[j])
            if dist < min_dist:
                min_dist, min_index = dist, j
                label[i] = j
        assement[i] = _distance(data[i],centroids[label[i]])**2
    
    # update centroid
    dbsList.append(dbs(data, label))
    new_centroids = []
    for m in range(k):
        if len(data[label==m]) == 0:
            k -= 1
        else:
            centroids[m] = np.mean(data[label==m],axis=0)
            new_centroids.append(centroids[m])
    centroids = new_centroids
    converged = _converged(old_centroids,centroids)  
    # while not converged:
    #     old_centroids = np.copy(centroids)
    #     for i in range(n):
    #         # determine the nearest centroid and track it with label
    #         min_dist, min_index = np.inf, -1
    #         for j in range(k):
    #             dist = _distance(data[i],centroids[j])
    #             if dist < min_dist:
    #                 min_dist, min_index = dist, j
    #                 label[i] = j
    #         assement[i] = _distance(data[i],centroids[label[i]])**2
        
    #     # update centroid
    #     dbsList.append(dbs(data, label))
    #     new_centroids = []
    #     for m in range(k):
    #         if len(data[label==m]) == 0:
    #             k -= 1
    #         else:
    #          centroids[m] = np.mean(data[label==m],axis=0)
    #          new_centroids.append(centroids[m])
    #     centroids = new_centroids
    #     converged = _converged(old_centroids,centroids)    
    # dbsList = dbsList + [dbsList[len(dbsList) - 1] for i in range(100 - len(dbsList))]
    print('dbsList', dbsList)
    return centroids, label, dbsList