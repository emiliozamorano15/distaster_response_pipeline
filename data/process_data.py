import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''reads messages.csv and categories.csv into dataframes, merges them, 
    and returns them as single data frame
    INPUTS:
        messages_filepath: str path to messages.csv
        categories_filepath: str path to categories.csv
    OUTPUTS:
        df: pandas dataframe merged messages and categories df
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'left', on = 'id')

    return df


def clean_data(df):
    ''' splits categories into columns, encodes them per message, and removes duplicates
    INPUTS:
        df: pandas df raw dataframe
    OUTPUTS:
        df: pandas df clean dataframe
    '''
    ## Split categories into separate category columns.
    categories = df['categories'].str.split(";", expand = True)

    ## select the first row of the categories dataframe and use it to get the column names
    row = pd.DataFrame({'categories': categories.iloc[0]})
    row['pos'] = (row['categories'].str.len())
    category_colnames = row.apply(lambda x: x['categories'][0:x['pos']-2], axis = 1)
    categories.columns = category_colnames

    ## Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])  
        categories[column] = pd.to_numeric(categories[column])

    ## Replace categories column in df with new category columns    
    df.drop(['categories'], axis = 1, inplace = True)
    df = df.merge(categories, left_index=True, right_index=True)

    # drop duplicates
    df.drop_duplicates(inplace = True)

    ## relabel column 'related' to binary values instead of multilabel
    df.loc[df['related']==2, 'related'] = 1 
        
    return df

def save_data(df, database_filename):
    ''' saves dataframe into a specific table in an existing sql database
    INPUTS:
        df: pandas df 
        database_filename: path to .db file
    OUTPUTS:
        None
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('tbl_disaster_response', engine) 
    
    pass  


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