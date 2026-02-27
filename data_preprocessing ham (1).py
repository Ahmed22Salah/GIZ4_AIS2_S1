import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Data_Preprocessing:
    '''Data preprocessing and visualization utility class.'''

    def __init__(self ,data):
        self.df = data


    def get_info(self):
        '''get some information from the user such as the shape ,dtypes ,columns name'''
        return self.df.info()

    def get_describetion(self):
        '''get descriptive statistics for numeric variables'''
        return self.df.describe()

    def overview(self):
        """Return dataset shape, columns and data types"""
        print(f"The Shape :{self.df.shape}")
        return pd.DataFrame({
            "Dtypes": self.df.dtypes,
            "Unique": self.df.nunique()
        }).T

    def numerical_cols(self):
        ''' get the numberical columns'''
        return self.df.select_dtypes('number').columns
    
    def categorical_cols(self):
        ''' get the categorical columns'''
        return self.df.select_dtypes('object').columns
    
    def is_null(self):
        ''' Get the null ,dublicated values in our dataset
        
        :return: Pandas dataframe contains the number of the nulls ,dublicated values and 
        the ratio of the nulls in the dataset '''

        null = self.df.isnull().sum()
        dup = self.df.duplicated().sum()
        ratio = null /self.df.shape[0]
        return pd.DataFrame({"NULLS" :null,
                            "NULLs Ratio" :ratio,
                            "Dubblicated" :dup }).T
        

    def drop_duplicate(self):
        '''drop the dublicated values in the dataset'''
        self.df = self.df.drop_duplicates()
        print("Dublicated Data are Droped")

    def clean_missing_data(self ,strategy ='mean'):
        ''' drop column that contain a lot of missing data or fill it with approperate strategy'''
        nulls = self.df.isnull().sum()
        ratio = nulls / self.df.shape[0]

        for col in self.df.columns:
            if ratio[col] >= 0.3:
                self.df = self.df.drop(columns=[col])

        for col in self.numerical_cols():
            if strategy =='mean':
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == 'median':
                self.df[col] = self.df[col].fillna(self.df[col].median())
            
        for col in self.categorical_cols():
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        print("Cleaning Data form Nulls is Done")
        

    def get_average(self ,column):
        ''' get the average value of the column'''
        return  self.df[column].mean()
    
    
    def Distribution(self ,column):
        '''visualize the count of the column'''
        numerical =self.numerical_cols()
        # used to remove the outliers
        if column in  numerical:
            upperlimit = self.df[column].quantile(0.99)
            filtered_df = self.df[self.df[column] <= upperlimit]
        else:
            filtered_df = self.df

        plt.figure(figsize=(10,5) ,dpi =200)
        sns.histplot(data= filtered_df ,x= column, bins=50)
        plt.title(f'The Distribution for the user {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    
    def outliers(self , column):
        ''' visualize the outliers in the column using boxplot'''
        plt.figure(figsize=(10,5), dpi =200)
        sns.boxplot(x= self.df[column] , orient='h')
        plt.title(f'The outliers of {column}')
        plt.xlabel(column)
        plt.show()


    def relational_dist(self , col1 ,col2):
        ''' visualize the distribution of certain column depending on the other'''
        plt.figure(figsize=(10,5) ,dpi =200)
        sns.barplot(data= self.df ,x =col1 ,y=col2 )
        plt.title(f"the Relation between {col1} & {col2}")
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()

    def correlation(self):
        ''' get the correlation between the dataset features'''
        
        corr = self.df.corr(numeric_only =True)
        plt.figure(figsize=(10,5) ,dpi =200)
        sns.heatmap(corr,annot= True)
        plt.title('the correlation of the dataset features')
        plt.show()

