import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class VisualizeData:
    def __init__(self,data):
        self.data = data
        self.column_names = list(self.data.columns)
        self.target_variable_name = self.column_names[-1]
        self.feature_names = self.column_names[:-1]
    
    def pair_wise_plot(self):
        new_columns = [self.feature_names[i] for i in [0,4]] + [self.target_variable_name]
        new_data_for_plot = self.data.filter(new_columns,axis=1)
        sns.pairplot(new_data_for_plot, hue=self.target_variable_name)
        plt.show()

    def single_feature_plot(self):
        pass

    def category_wise_visualization(self):
        class_labels = self.data[self.target_variable_name].unique()
        print(class_labels)
        # print(self.data.loc[self.data[' UNS']=='very_low',['STG']])

def read_data():
    data = pd.read_excel('user_knowledge_data.xls', sheet_name = 'Training_Data')
    return data


if __name__ == '__main__':
    data = read_data()
    visualization_object = VisualizeData(data)
    visualization_object.pair_wise_plot()


