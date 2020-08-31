# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:59:36 2020

@author: Mingkai Liu

This is a module for data cleansing for the tennis data set

It takes the required start and ending years, download the data from
the source, clean it and return the cleaned data matrix for further
exploration and feature engineering 
"""
import numpy as np
import pandas as pd
import os
import string   
from itertools import product
from sklearn.preprocessing import LabelEncoder
import time
import sys
import gc
import pickle
#sys.version_info

def download_tennis_data(year_start, year_end):
    
    import numpy as np
    import pandas as pd
    
    # download the data from  https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/
    matrix = pd.DataFrame()
    
    url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
    
     
    print('Start downloading dataset:')
    year_range = np.arange(year_start,year_end+1)
    
    for year in year_range:
        file_url = url+'atp_matches_'+str(year)+'.csv'
        # read csv from the file_url
        df = pd.read_csv(file_url,index_col=0,parse_dates=[0])
        
        if year==year_range[0]:
            cols = df.columns
    
        # concatenate the dataframe
        matrix = pd.concat([matrix, df], ignore_index=True, sort=False, keys=cols)
        
    print('Download finished.')
    
    return matrix



def clean_tennis_data(dataframe):
    
    print('start cleaning the data')
    matrix = dataframe
    # convert the date information to pd.datetime format
    matrix['tourney_date'] = pd.to_datetime(matrix.tourney_date.round(), 
                                            format='%Y%m%d', errors='ignore')
    # extract information of date
    matrix['year'] = matrix['tourney_date'].dt.year
    matrix['month'] = matrix['tourney_date'].dt.month
    matrix['day'] = matrix['tourney_date'].dt.day
    matrix['day_week'] = matrix['tourney_date'].dt.dayofweek
    
    #split the scores
    matrix['split']=matrix['score'].str.split(' ')
    matrix['split_len']=matrix['split'].map(lambda x: len(x), na_action='ignore')
    
    # further split the set point to winner and loser
    for i in range(1,7):
        
        matrix['split']=matrix['score'].str.split(' ')
        #column for set point
        col_set = 'set_'+str(i)  
        # use na_action='ignore' to ignore the nan, otherwise it throws error    
        matrix[col_set]=matrix['split'].map(lambda x: x[i-1].strip() if len(x)>i-1 else np.nan, na_action='ignore')    
        
        if i<=5:   # we use i<=5 because for i=6, matrix[set_6]=NaN, and it throws error when applying .str to it
            matrix['split']=matrix[col_set].str.split('-')
            # use na_action='ignore' to ignore the nan, otherwise it throws error
            matrix['winner_'+col_set]=matrix['split'].map(lambda x: x[0].strip(), na_action='ignore')
            matrix['loser_'+col_set]=matrix['split'].map(lambda x: x[1].strip() if len(x)>1 else np.nan, na_action='ignore')

    
    # fix the weird data points
    for i in range(1,6):
        col_win_set = 'winner_'+'set_'+str(i)
        col_los_set = 'loser_'+'set_'+str(i)
        
        # first replace nonumerical features with nan 
        matrix[col_win_set] = matrix[col_win_set].map(lambda x:x if x not in
                                                      ['W/O','In','Walkover','RET','','abandoned','ABD','NA',
                                                       'DEF','Played','Progress','and','Default','Def.','unfinished','Unfinished']
                                                     else np.nan, na_action='ignore') 
        
        matrix[col_los_set] = matrix[col_los_set].map(lambda x:x if x not in
                                                      ['W/O','In','Walkover','RET','','abandoned',
                                                       'DEF','Played','Progress','and','Default','Def.','unfinished','Unfinished']
                                                     else np.nan, na_action='ignore') 
        
        # replace 'month' data with number
        matrix[col_win_set] = matrix[col_win_set].replace({'Jun': '6', 'Apr': '4','Feb':'2','00':'0'})
        matrix[col_los_set] = matrix[col_los_set].replace({'Jun': '6', 'Apr': '4','Feb':'2','00':'0'})
        
        #define a tiebreak point columns
        col_set_tb = 'set_'+str(i)+'_tb'
        
        # for winner columns, there is no tiebreak point with (), the only left over are the last game with [],
        # we can remove all the punctuation and only keep the number
        matrix[col_win_set] = matrix[col_win_set].map(lambda x: x.translate(str.maketrans('', '', string.punctuation)), na_action='ignore')
        matrix[col_win_set] = matrix[col_win_set].astype(np.float32)
        
        # for loser columns, there is a tiebreak point with (), we need to further seperate them out
        matrix['split']=matrix[col_los_set].str.split('(')
        # the first element before the '(' is the set point of loser, it could include point with ']'
        matrix[col_los_set] = matrix['split'].map(lambda x: x[0].strip(), na_action='ignore')    
        # remove the punctuation
        matrix[col_los_set] = matrix[col_los_set].map(lambda x: x.translate(str.maketrans('', '', string.punctuation)), na_action='ignore')
        matrix[col_los_set] = matrix[col_los_set].astype(np.float32)
        
        # the element after the ( is the tiebreak point, it could include ')' 
        matrix[col_set_tb] = matrix['split'].map(lambda x: x[1].strip() if len(x)>1 else np.nan, na_action='ignore')
        # remove the punctuation
        matrix[col_set_tb] = matrix[col_set_tb].map(lambda x: x.translate(str.maketrans('', '', string.punctuation)), na_action='ignore')
        matrix[col_set_tb] = matrix[col_set_tb].astype(np.float32)
        
    # drop columns unnecessary    
    matrix.drop(['split','split_len','set_1','set_2','set_3','set_4','set_5','set_6'],axis=1,inplace=True)
    
    # fix the upper/lower case
    matrix['winner_entry']= matrix['winner_entry'].str.upper()
    matrix['loser_entry']=matrix['loser_entry'].str.upper()
    
    # the winner/loser seed are object that mix numerics and strings. Convert the numerics to strings. use 'isinstance(x,float)' to check if the object is float
    matrix['loser_seed'] = matrix['loser_seed'].map(lambda x: str(int(x)) if isinstance(x, float) else x, na_action = 'ignore')
    matrix['winner_seed'] = matrix['winner_seed'].map(lambda x: str(int(x)) if isinstance(x, float) else x, na_action = 'ignore')

    # break point saved
    matrix['w_bpsr'] = matrix['w_bpSaved']/matrix['w_bpFaced']
    matrix['l_bpsr'] = matrix['l_bpSaved']/matrix['l_bpFaced']
    
    # break point converted
    matrix['w_bpc'] = matrix['l_bpFaced'] - matrix['l_bpSaved']
    matrix['l_bpc'] = matrix['w_bpFaced'] - matrix['w_bpSaved']
    
    # break point converted ratio
    matrix['w_bpcr'] = matrix['w_bpc']/ matrix['l_bpFaced']
    matrix['l_bpcr'] = matrix['l_bpc']/ matrix['w_bpFaced']
    # we convert the seed information to numeric. If the seed information is digit, use the number; if it is others, use the draw size
    # we convert the seed information to numeric. If the seed information is digit, use the number; if it is others, use the draw size
    matrix['winner_seed_code'] = matrix['winner_seed'].map(lambda x: float(x) if x.isdigit() else 0, na_action = 'ignore')
    matrix['loser_seed_code'] = matrix['loser_seed'].map(lambda x: float(x) if x.isdigit() else 0, na_action = 'ignore')
    
    matrix['winner_seed_code'] = matrix['winner_seed_code']+matrix['draw_size']*(matrix['winner_seed_code']==0)  
    matrix['loser_seed_code'] = matrix['loser_seed_code']+matrix['draw_size']*(matrix['loser_seed_code']==0)
     
    matrix['winner_hand_code']=matrix['winner_hand'].map(lambda x: 1 if x=='R' else (-1 if x=='L' else 0), na_action = 'ignore')
    matrix['loser_hand_code']=matrix['loser_hand'].map(lambda x: 1 if x=='R' else (-1 if x=='L' else 0), na_action = 'ignore')
    
    # rename some of the columns
    col_winner = ['w_ace','w_df','w_svpt', 'w_1stIn','w_1stWon','w_2ndWon','w_SvGms','w_bpFaced',
             'w_bpSaved','w_bpc','w_bpsr','w_bpcr']
    col_winner_n = ['winner_ace','winner_df','winner_svpt', 'winner_1stIn','winner_1stWon','winner_2ndWon','winner_SvGms','winner_bpFaced',
             'winner_bpSaved','winner_bpc','winner_bpsr','winner_bpcr']
    
    col_loser = ['l_ace','l_df','l_svpt', 'l_1stIn','l_1stWon','l_2ndWon','l_SvGms','l_bpFaced',
             'l_bpSaved','l_bpc','l_bpsr','l_bpcr']
    col_loser_n = ['loser_ace','loser_df','loser_svpt', 'loser_1stIn','loser_1stWon','loser_2ndWon','loser_SvGms','loser_bpFaced',
             'loser_bpSaved','loser_bpc','loser_bpsr','loser_bpcr']
    
    
    for i in range(len(col_winner)):
        matrix.rename(columns={col_winner[i]: col_winner_n[i]}, inplace=True)
        matrix.rename(columns={col_loser[i]: col_loser_n[i]}, inplace=True)
    
    #sort the data according to date, name and match_num
    matrix = matrix.sort_values(by=['tourney_date','tourney_name','match_num'],ascending = True)
    matrix = matrix.reset_index().drop('index',axis=1)
    
    matrix[['draw_size','year','month','day','best_of','day_week','match_num']].astype(np.int64)
    
    print('Finished cleaning the data.')
    
    return matrix


def matrix_p1p2(matrix):
    
    """"This function takes the cleaned matrix with winner and loser information, and turn it into a new matrix
    with player_1 and player_2. For a pair of player, whoever the player id is smaller will be player_1
    """
    #print('Start converting data notation from winner/loser to player_1/player_2')
    # define a new matrix for trainning and re-arange the information for winner and loser as player 1 and player 2. For each pair, player_1_id < player_2_id.
    matrix_n = pd.DataFrame()
    
    # match information
    col_match = ['tourney_name', 'surface', 'draw_size', 'tourney_level', 'tourney_date','year', 'month', 'day', 'day_week',
           'match_num', 'best_of', 'round', 'minutes']
    
    matrix_n[col_match] = matrix[col_match]
    
    # columns for winner and loser
    
    col_w = [item for item in matrix.columns if 'winner' in item] 
    col_l = [item for item in matrix.columns if 'loser' in item] 
    
    # new columns for player 1 and player 2
    col_p1 = [item.replace('winner', 'p1') for item in col_w] 
    col_p2 = [item.replace('winner', 'p2') for item in col_w] 
     
    # re-arange the columns based on p1 and p2
    matrix[['winner_id','loser_id']]=matrix[['winner_id','loser_id']].astype(np.float64)
    
    matrix_n[col_p1] = matrix.loc[matrix.winner_id<matrix.loser_id,col_w] 
    matrix_n[col_p2] = matrix.loc[matrix.winner_id>matrix.loser_id,col_w] 
    
    matrix_n['p1_win'] = matrix_n['p1_id'].map(lambda x: 1 if x>0 else 0, na_action = 'ignore').fillna(0)
    matrix_n['p2_win'] = matrix_n['p2_id'].map(lambda x: 1 if x>0 else 0, na_action = 'ignore').fillna(0)
    
    for i in range(len(col_p1)):
        matrix_n[col_p1[i]].fillna(matrix[matrix.winner_id>matrix.loser_id][col_l[i]],inplace = True)
        matrix_n[col_p2[i]].fillna(matrix[matrix.winner_id<matrix.loser_id][col_l[i]],inplace = True)
    
    # add information for the number of set won by each player
    matrix_n['p1_sets_win'] = 0.0
    matrix_n['p2_sets_win'] = 0.0
    
    for i in range(1,6):
        matrix_n['p1_sets_win'] = matrix_n['p1_sets_win'] + 1.0*(matrix_n['p1_set_'+str(i)]>matrix_n['p2_set_'+str(i)])
        matrix_n['p2_sets_win'] = matrix_n['p2_sets_win'] + 1.0*(matrix_n['p1_set_'+str(i)]<matrix_n['p2_set_'+str(i)])
    
    matrix_n[['p1_id','p2_id']].astype(np.int64)
    
    
    #print('Conversion finished')
    
    return matrix_n 