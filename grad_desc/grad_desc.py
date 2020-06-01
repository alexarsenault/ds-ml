import numpy as np
import matplotlib.pyplot as plt

def calc_cost(theta, x, y):
    
    m = len(y)

    pred = (x.dot(theta))
    cost = (1/2*m) * (np.sum(np.square(pred - y)))

    return cost

def grad_descend(theta, x, y, learningRate, numIterations):

    m = len(y)
    cost_history = np.zeros(numIterations)
    theta_history = np.zeros((numIterations,2))

    for itr in range(numIterations):
        curr_pred = np.dot(x, theta)
        diff_vec = curr_pred - y
        
        theta = theta - ((1/m) * learningRate*(x.T.dot(diff_vec)))
        theta_history[itr,:] = theta.T
        cost_history[itr] = calc_cost(theta,x,y)

    return theta, cost_history, theta_history 


def main():

    lr = 0.01
    num_itr = 100

    x = 10 * np.random.rand(100,1)
    y = 4 + 2 * x+np.random.randn(100,1)

    plt.scatter(x,y)
    plt.show()

    X_b = np.c_[np.ones((len(x),1)),x]

    theta = np.random.randn(2,1)

    theta,cost_history,theta_history = grad_descend(theta,X_b,y,lr,num_itr)


    num_itr_vector = range(0,100)
    #plt.scatter(num_itr_vector,theta_history[:,1])

    y_regression = theta[1]*x + theta[0]*1    
    plt.scatter(x,y)
    plt.plot(x,y_regression)
    plt.show()

    print("End of main function.")



if __name__ == "__main__":
    main()