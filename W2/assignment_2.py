import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

class VisualizeData:
    def __init__(self,data):
        self.data = data
        self.column_names = list(self.data.columns)
        self.target_variable_name = self.column_names[-1]
        self.feature_names = self.column_names[:-1]
    
    def pair_wise_plot(self):
        new_columns = [self.feature_names[i] for i in [0,1,2,3,4]] + [self.target_variable_name]
        new_data_for_plot = self.data.filter(new_columns,axis=1)
        sns.pairplot(new_data_for_plot, hue=self.target_variable_name)
        plt.show()

    def single_feature_plot(self):
        f1_name = 'PEG'
        column_for_f1 = self.data[f1_name]
        target = self.data[self.target_variable_name]
        plt.scatter(column_for_f1,target)
        plt.show()

    def category_wise_visualization(self):
        class_labels = self.data[self.target_variable_name].unique()
        print(class_labels)
        # print(self.data.loc[self.data[' UNS']=='very_low',['STG']])

def read_data():
    data = pd.read_excel('user_knowledge_data.xls', sheet_name = 'Training_Data')
    return data

def apply_knn(data,n):
    y_string = data[' UNS']
    label_string = ['very_low','Low','Middle','High']
    y = []
    for el in y_string:
        index = label_string.index(el)
        y.append(index)
    y = np.array(y)
    X = data.drop([' UNS'],axis=1)
    X_tr, X_val, y_tr, y_val = train_test_split(X,y,test_size=0.2, random_state=10)
    # print(X_tr.shape,y_tr.shape,X_val.shape)
    knn_object = KNeighborsClassifier(n_neighbors=n)
    knn_object.fit(X_tr,y_tr)
    y_val_prediction = knn_object.predict(X_val)
    y_tr_prediction = knn_object.predict(X_tr)
    acc = 0
    acc_tr = 0
    for el1,el2 in zip(y_val,y_val_prediction):
        if el1 == el2:
            acc += 1
    for el1,el2 in zip(y_tr,y_tr_prediction):
        if el1 == el2:
            acc_tr += 1
    print('Accuracy:', acc/len(y_val)*100)
    print('Training Accuracy:', acc_tr/len(y_tr)*100)

def apply_LR(data,cost):
    y_string = data[' UNS']
    label_string = ['very_low','Low','Middle','High']
    y = []
    for el in y_string:
        index = label_string.index(el)
        y.append(index)
    y = np.array(y)
    X = data.drop([' UNS'],axis=1)
    X_tr, X_val, y_tr, y_val = train_test_split(X,y,test_size=0.2, random_state=10)
    # print(X_tr.shape,y_tr.shape,X_val.shape)
    lr_object = LogisticRegression(C=cost)
    lr_object.fit(X_tr,y_tr)
    y_val_prediction = lr_object.predict(X_val)
    y_tr_prediction = lr_object.predict(X_tr)
    acc = 0
    acc_tr = 0
    for el1,el2 in zip(y_val,y_val_prediction):
        if el1 == el2:
            acc += 1
    for el1,el2 in zip(y_tr,y_tr_prediction):
        if el1 == el2:
            acc_tr += 1
    print('Accuracy:', acc/len(y_val)*100)
    print('Training Accuracy:', acc_tr/len(y_tr)*100)
    # TODO: save the model

def apply_saved_model_on_test_data():
    # TODO: load the model and predict on test data
    pass

if __name__ == '__main__':
    data = read_data()
    visualization_object = VisualizeData(data)
    # visualization_object.pair_wise_plot()
    # visualization_object.single_feature_plot()
    # apply_knn(data,n=13)
    apply_LR(data,cost=1e5)