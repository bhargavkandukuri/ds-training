import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_data():
    Data = pd.read_csv('train.csv')
    y = Data['SalePrice']
    X = Data.drop('SalePrice',axis=1)
    return X,y

def get_column_names(X):
    return X.columns

def visualize(x,y):
    plt.scatter(x,y)
    plt.xlabel('BaseMent Area', fontsize=20)
    plt.ylabel('Price', fontsize=20)
    plt.show()

def visualize_3d(x1,x2,y):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(x1,x2,y)
    ax.set_xlabel('Basement')
    ax.set_ylabel('1stFlr')
    ax.set_zlabel('price')
    plt.show()

if __name__ == '__main__':
    X,y = read_data()
    column_names = get_column_names(X)
    # visualize(X['TotalBsmtSF'][100:200],y[100:200])
    visualize_3d(X['TotalBsmtSF'],X['1stFlrSF'],y)