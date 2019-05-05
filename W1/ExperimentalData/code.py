import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def generate_data(size, add_noise = False):
    '''
        Geneate X and y matrix.
        X = (size,2) (m1 and m2)
        y = (size,1)
        relation: y = m1*m2*6.67 + noise (if add_noise is true)

    '''
    np.random.seed(0)
    m1 = np.random.rand(size)*10
    m2 = np.random.rand(size)*10
    y = 6.67 * m1 * m2
    if add_noise: 
        for index, el in enumerate(y):
            y[index] += np.random.rand()*el
    X = np.ones((size,2))
    X[:,0] = m1
    X[:,1] = m2
    return X,y

def visualize_data(X,y,in_3d=False, print_values=False):
    if in_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(X[:,0],X[:,1],y)
        ax.set_xlabel('Mass 1')
        ax.set_ylabel('Mass 2')
        ax.set_zlabel('Force')
        if print_values:
            for index, value in enumerate(X[:,0]):
                print(value,X[:,1][index],y[index])
        plt.show()
        return
    plt.subplot(121)
    plt.scatter(X[:,0],y,c='red')
    plt.subplot(122)
    plt.scatter(X[:,1],y,c='green')
    plt.show()

def apply_regression(X,y,max_degree):
    X_train_and_cv, X_test, y_train_and_cv, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
    X_train, X_cv, y_train, y_cv = train_test_split(X_train_and_cv,y_train_and_cv,test_size=0.2,random_state=0)
    print(X_train.shape,X_cv.shape,X_test.shape,y_train.shape,y_cv.shape,y_test.shape)

    mse_training = []
    mse_validation = []

    for degree in range(0,max_degree):
        poly_features = PolynomialFeatures(degree=degree)
        X_train_polynomial = poly_features.fit_transform(X_train)
        X_cv_polynomial = poly_features.fit_transform(X_cv)
        lr_model = LinearRegression()
        lr_model.fit(X_train_polynomial,y_train)
        y_train_prediction = lr_model.predict(X_train_polynomial)
        y_cv_prediction = lr_model.predict(X_cv_polynomial)
        mse_training.append(mean_squared_error(y_train,y_train_prediction))
        mse_validation.append(mean_squared_error(y_cv,y_cv_prediction))

    plot_errors(mse_training,mse_validation, print_error=True)
    best_degree = -1 
    min_error = float('inf')
    for index,error in enumerate(mse_validation):
        if error < min_error:
            min_error = error
            best_degree = index
    print('best_degree =', best_degree)
    get_best_model(best_degree=best_degree,X_train=X_train,y_train=y_train)

def get_best_model(best_degree,X_train,y_train):
    best_poly_features = PolynomialFeatures(degree=best_degree)
    X_train_best_polynomial = best_poly_features.fit_transform(X_train)
    print('Feature names', best_poly_features.get_feature_names())
    lr_model_best = LinearRegression()
    lr_model_best.fit(X_train_best_polynomial,y_train)
    print('coefficients', lr_model_best.coef_)
    
def plot_errors(mse_training,mse_validation,print_error=False):
    # print(len(mse_training))
    degrees = [ i for i in range(len(mse_training))]
    plt.scatter(degrees[2:], mse_training[2:], c='green')
    plt.scatter(degrees[2:], mse_validation[2:], c='red')
    plt.xlabel('degree')
    plt.ylabel('Error')
    if print_error:
        for e1,e2 in zip(mse_training,mse_validation):
            print(e1,e2)
    plt.show()


if __name__ == '__main__':
    X,y = generate_data(size=5000,add_noise=False)
    # visualize_data(X,y,in_3d=True,print_values=False)
    apply_regression(X,y,max_degree=10)