import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():

    rscale = 2
    centers = [[np.random.randn()*rscale, np.random.randn()*rscale], 
               [np.random.randn()*rscale, np.random.randn()*rscale],
               [np.random.randn()*rscale, np.random.randn()*rscale]]
    print(str(centers))

    [X, labels_truth] = make_blobs(n_samples=1000, centers = centers, cluster_std = .3)

    [fig, ax] = plt.subplots(2)
    ax[0].scatter(X[:,0], X[:,1])

    X_scaled = StandardScaler().fit_transform(X)
    ax[1].scatter(X_scaled[:,0], X_scaled[:,1])

    plt.show()

    db = DBSCAN(eps=0.3, min_samples=10).fit(X_scaled)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    
    labels = db.labels_
    noise_points = list(labels).count(-1)
    print("Number of noise points %d" %noise_points)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0,1,len(unique_labels))]
    # core_samples_mask = 

    for k, col in zip(unique_labels, colors):

        if k == -1:
            col = [0,0,0,1]
        
        class_member_mask = (labels == k)

        xy = X_scaled[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)
        #plt.show()
        print("")

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
        #plt.show()
        print("")

    plt.show()
    print("ending main")



if __name__ == "__main__":
    main()
