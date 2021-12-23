def get_clusters_centroids(X_dataset, Y_dataset, n_cluster):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_distances_argmin_min  
    
    km = KMeans(n_clusters=n_cluster,
                   init='k-means++',
                   n_init=10,
                   max_iter=300,
                   random_state=0)
    km.fit(X_dataset)
    cluster_index, dist = pairwise_distances_argmin_min(km.cluster_centers_, X_dataset)
    
    X_train = [X_dataset[c_center] for c_center in cluster_index]
    Y_train = [Y_dataset[c_center] for c_center in cluster_index]
    
    ind_rem = np.delete(np.arange(len(X_dataset)),cluster_index)
    
    X_test = [X_dataset[f] for f in ind_rem]
    Y_test = [Y_dataset[f] for f in ind_rem]

    return X_train,Y_train,X_test,Y_test

def regression(estimator, X_train,Y_train,X_test,Y_test, transfer=False):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    estimator.fit(X_train,Y_train)
    
    Y_train_pred = estimator.predict(X_train)
    Y_test_pred = estimator.predict(X_test)
    
    if transfer == True:
        ind = np.argmin(Y_test)
        Y_test = np.array(Y_test)
        Y_test_pred -= Y_test_pred[ind]
        Y_test -= Y_test[ind]
        
    Y_train_mae = mean_absolute_error(Y_train,Y_train_pred)     
    Y_test_mae = mean_absolute_error(Y_test,Y_test_pred)     
    
    Y_train_RMSE = mean_squared_error(Y_train, Y_train_pred, squared=False)
    Y_test_RMSE = mean_squared_error(Y_test, Y_test_pred, squared=False)

    return Y_train_mae, Y_test_mae, Y_train_RMSE, Y_test_RMSE
