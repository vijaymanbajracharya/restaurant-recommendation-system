"""Data Loader"""

import pandas as pd

def load(path,cuisine):
    #having a path as a parameter allows us to play with files of different sizes
    # Load the CSV file into a DataFrame
    df = pd.read_csv(path)
    
    # Assume you want to select rows where the cuisine is 'Italian', can be anything a user wants, and there can be multiple cuisines too
    # selected_cuisine = 'Italian'
    #this is something to decide, pass the cuisine as a parameter(a list) or take it as input here in the function
    #here am assuming it comes in as a list 
    
    # Filter the DataFrame
    filtered_df = df[df['cuisine'].str.lower().isin(map(str.lower, cuisine))]
    #this filtered_Df is the data frame with all the restaurants, which belong to a certain cuisine on which the bees operate on
    return filtered_df
    
