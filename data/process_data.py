import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

def load_data(messages_filepath, categories_filepath):
   messages = pd.read_csv(messages_filepath, dtype=str)
   categories = pd.read_csv(categories_filepath, dtype=str)
   df = messages.merge(categories, how = 'outer', on ='id')
   return df

def clean_data(df):
    categories = df['categories'].str.split(pat = ';', expand = True)
    row = categories.iloc[0]
    category_colnames = lambda x: [str(y)[:-2] for y in x]
    categories.columns = category_colnames(row)
    def slicing(x):
        return x[-1]
    for column in categories:
        categories[column] = categories[column].apply(slicing) 
        categories[column] = categories[column].convert_objects(convert_numeric=True)
    df = df.drop(labels = ['categories'], axis = 1)
    df = pd.concat([df, categories], axis = 1)
    df = df.drop_duplicates()
    df.related.replace(2,1, inplace=True)
    df.related.unique()
    
    return df


def save_data(df, database_filename):
   engine = create_engine('sqlite:///{}'.format(database_filename))
   df.to_sql('DisasterTable', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()