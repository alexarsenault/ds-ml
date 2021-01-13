import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from random import random

def plot_homemade_decision_function(x1, x2, x3, intercept, ax=None):
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    z = np.linspace(zlim[0], zlim[1], 30)
    X, Y = np.meshgrid(x,y)
    z = lambda x,y: (-intercept - x1*x - x2*y) / x3

    xy = np.vstack([X.ravel(), Y.ravel()]).T
    p = x1*xy[:,0] + x2*xy[:,1]
    P = p.reshape(X.shape)

    # plot decision boundary and margins
    ax.plot_surface(X, Y, z(X,Y).astype(float))

    """
    ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Sepal Length (cm), X1')
    ax.set_ylabel('Petal Length (cm), X2')
    ax.set_zlabel('Sepal Width (cm), X3')
    plt.show()

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()

def main():
    df = pd.read_csv('./data/iris.csv')
    df = df.drop(['Id'],axis=1)
    target = df['Species']
    s = set(target)
    s = list(s)
    rows = list(range(100,150))
    df = df.drop(df.index[rows])

    x = df['SepalLengthCm']
    y = df['PetalLengthCm']
    z = df['SepalWidthCm']

    setosa_x = x[:50]
    setosa_y = y[:50]
    setosa_z = z[:50]

    versicolor_x = x[50:]
    versicolor_y = y[50:]
    versicolor_z = z[50:]

    """
    plt.figure(figsize=(8,6))
    plt.scatter(setosa_x,setosa_y,color='green')
    plt.scatter(versicolor_x,versicolor_y,color='red')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.show()
    """

    ## Drop rest of the features and extract the target values
    #df = df.drop(['SepalWidthCm','PetalWidthCm'],axis=1)
    df = df.drop(['PetalWidthCm'],axis=1)

    df.loc[df['Species']=='Iris-setosa', 'Species'] = -1
    df.loc[df['Species']=='Iris-versicolor','Species'] = 1
    Y = df['Species']
    
    df = df.drop(['Species'],axis=1)
    X = df.values.tolist()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)

    x1_train = x_train[:,0].reshape(x_train[:,0].shape[0],1)     # x1-coordinate training data
    x2_train = x_train[:,1].reshape(x_train[:,1].shape[0],1)     # x2-coordinate training data
    x3_train = x_train[:,2].reshape(x_train[:,2].shape[0],1)     # x3-coordinate training data

    w1 = np.ones((x_train.shape[0],1))*random()     # Initialize weight 1 with random values
    w2 = np.ones((x_train.shape[0],1))*random()     # Initialize weight 2 with random values
    w3 = np.ones((x_train.shape[0],1))*random()     # Initialize weight 3 with random values

    epochs = 1
    num_epochs = 10000
    alpha = 0.0001

    w1_list = []
    w2_list = []
    w3_list = []
    epoch_cost = np.zeros((x1_train.shape[0],1))
    cost_list = []

    
    while(epochs < num_epochs):
        y = w1 * x1_train + w2 * x2_train + w3 * x3_train   # Compute target variable using current weights
        prod = y * y_train

        count = 0
        for val in prod:
            if(val >= 1):       # Correct prediction, update weights
                cost = 0
                w1 = w1 - alpha * (2 * 1/epochs * w1)
                w2 = w2 - alpha * (2 * 1/epochs * w2)
                w3 = w3 - alpha * (2 * 1/epochs * w3)
                
            else:               # Incorrect prediction, update weights
                cost = 1 - val 
                w1 = w1 + alpha * (x1_train[count] * y_train[count] - 2 * 1/epochs * w1)
                w2 = w2 + alpha * (x2_train[count] * y_train[count] - 2 * 1/epochs * w2)
                w3 = w3 + alpha * (x3_train[count] * y_train[count] - 2 * 1/epochs * w3)

            epoch_cost[count] = cost
            count += 1
        
        cost_list.append(epoch_cost.sum())
        w1_list.append(w1[0])
        w2_list.append(w2[0])
        w3_list.append(w3[0])
        epochs += 1

    w_vec = np.array([w1[0], w2[0], w3[0]])
    ind_neg_1 = y_train==-1
    ind_pos_1 = y_train==1
    min_idx = np.argmin(np.dot(w_vec.T , x_train[ind_pos_1.flatten(),:].T))
    min_val = np.dot(w_vec.T,x_train[min_idx,:])

    max_idx = np.argmax(np.dot(w_vec.T, x_train[ind_neg_1.flatten(),:].T))
    max_val = np.dot(w_vec.T,x_train[max_idx,:])
    
    intercept = -(max_val + min_val) / 2

    # Compute predictions based on our weights
    test_predictions = w1[0]*x_test[:,0] + w2[0]*x_test[:,1] + w3[0]*x_test[:,2]
    test_predictions[test_predictions>0] = 1
    test_predictions[test_predictions<0] = -1
    test_predictions = test_predictions.reshape(test_predictions.shape[0],1)

    results = (test_predictions == y_test)
    print(results)

    # Plot scatter with SVM boundaries
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(setosa_x,setosa_y,setosa_z,color='green')
    ax.scatter(versicolor_x,versicolor_y,versicolor_z,color='red')
    plot_homemade_decision_function(w1[0], w2[0], w3[0], intercept)

    # Plot weight values over epochs
    plt.plot(range(len(w1_list)),w1_list,color='b',label='W1')
    plt.plot(range(len(w2_list)),w2_list,color='r',label='W2')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.show()

    # Plot total cost over epochs
    plt.plot(range(len(cost_list)),cost_list)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.show()
    
    # Generate random samples centered around 2 points
    X,y = make_blobs(n_samples=200, centers=2, random_state=11, cluster_std=.7)
    plt.scatter(X[:,0], X[:,1])
    plt.show()

    svc_model = SVC(kernel="linear", C=1E10)
    svc_model.fit(X,y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(svc_model)

if __name__ == "__main__":
    main()
