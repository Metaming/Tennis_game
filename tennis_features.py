# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 19:46:04 2020

@author: Mingkai Liu
"""
import numpy as np
import pandas as pd
#import os
#import string   
#from itertools import product
#from sklearn.preprocessing import LabelEncoder
import time
import sys
#import gc
#import pickle
from time import sleep
from datetime import timedelta


def matrix_p1p2(matrix):
        
    """"This function takes the cleaned matrix with winner and loser information, and turn it into a new matrix
    with player_1 and player_2. For a pair of player, whoever the player id is smaller will be player_1
    """
    t = time.time()
    print('start convert to matrix p1 vs p2')
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
    
    time_elapsed = round((time.time() - t)/60,1)
    print('Time elapsed: {} mins'.format(time_elapsed))
    
    return matrix_n 



def Elo_rating(matrix_n,s=0.4):
    
    
    """ 
    This function takes the matrix with p1 and p2 information and return the Elo ratings,
    Ref: https://www.betfair.com.au/hub/tennis-elo-modelling/.
    Parameter s is to control the influence of future matches. The larger s is, the smaller
    future match will have on Elo scores. By default, s=0.4.
    """
    t = time.time()
    col_i = ['tourney_date','surface','p1_id','p2_id','p1_win','p2_win']
    matrix_e = matrix_n[col_i].replace([np.inf, -np.inf], np.nan)
    total_len = len(matrix_e)
    
    # We start the Elo calculation from 1980
    elo_list = pd.Series()
    # use set to get the unique player_id
    elo_id = set(list(matrix_n.p1_id.unique())+list(matrix_n.p2_id.unique()))
    
    elo_list = pd.Series(data = np.ones(len(elo_id))*1500, index=elo_id)
    elo_list_hard = pd.Series(data = np.ones(len(elo_id))*1500, index=elo_id)
    elo_list_clay = pd.Series(data = np.ones(len(elo_id))*1500, index=elo_id)
    elo_list_grass = pd.Series(data = np.ones(len(elo_id))*1500, index=elo_id)
    elo_list_carpet = pd.Series(data = np.ones(len(elo_id))*1500, index=elo_id)
    
    
    for row in matrix_e.itertuples():
        
        index = row.Index
        
       
        surface = row.surface
    
        p1_id = row.p1_id
        p2_id = row.p2_id
    
        p1_win = row.p1_win
        p2_win = row.p2_win
    
    
        # calculate the total Elo                
    
        # number of matches played by player 1 and player 2 so far
        M1 = matrix_e[(matrix_e.index<index) & ((matrix_e.p1_id ==p1_id) | (matrix_e.p2_id ==p1_id))].shape[0]
        M2 = matrix_e[(matrix_e.index<index) & ((matrix_e.p1_id ==p2_id) | (matrix_e.p2_id ==p2_id))].shape[0]
    
        K1=250/((M1+5)**s)
        K2=250/((M2+5)**s)
    
        Elo_1 = elo_list[p1_id]
        Elo_2 = elo_list[p2_id] 
    
        p1_win_pred = 1/(1+10**((Elo_2-Elo_1)/400))
        p2_win_pred = 1/(1+10**((Elo_1-Elo_2)/400))
    
        matrix_e.loc[index,'Elo_1_s'+str(s)]=elo_list[p1_id]
        matrix_e.loc[index,'Elo_2_s'+str(s)]=elo_list[p2_id]
    
        matrix_e.loc[index,'p1_win_pred_s'+str(s)]=p1_win_pred
        matrix_e.loc[index,'p2_win_pred_s'+str(s)]=p2_win_pred
    
        Elo_1_new = Elo_1+K1*(p1_win-p1_win_pred)
        Elo_2_new = Elo_2+K2*(p2_win-p2_win_pred)
    
    
        if Elo_1_new>=0:
            elo_list[p1_id]=Elo_1_new  
        if Elo_2_new>=0:
            elo_list[p2_id]=Elo_2_new          
        
    
        # we also calculate the Elo for a particular surface since some players are better at certain surface    
    
        # number of matches played by player 1 and player 2 so far on this surface type
        M1_sf = matrix_e[(matrix_e.index<index) & ((matrix_e.p1_id ==p1_id) | (matrix_e.p2_id ==p1_id)) & (matrix_e.surface==surface)].shape[0]
        M2_sf = matrix_e[(matrix_e.index<index) & ((matrix_e.p1_id ==p2_id) | (matrix_e.p2_id ==p2_id)) & (matrix_e.surface==surface)].shape[0]
    
        K1_sf=250/((M1_sf+5)**s)
        K2_sf=250/((M2_sf+5)**s)
    
        if surface == 'Hard':
            Elo_1_sf = elo_list_hard[p1_id]
            Elo_2_sf = elo_list_hard[p2_id]
    
            p1_win_pred_sf = 1/(1+10**((Elo_2_sf-Elo_1_sf)/400))
            p2_win_pred_sf = 1/(1+10**((Elo_1_sf-Elo_2_sf)/400))
    
            matrix_e.loc[index,'Elo_1_hard_s'+str(s)]=elo_list_hard[p1_id]
            matrix_e.loc[index,'Elo_2_hard_s'+str(s)]=elo_list_hard[p2_id]
    
            matrix_e.loc[index,'p1_win_pred_hard_s'+str(s)]=p1_win_pred_sf
            matrix_e.loc[index,'p2_win_pred_hard_s'+str(s)]=p2_win_pred_sf
    
            Elo_1_sf_new = Elo_1_sf+K1_sf*(p1_win-p1_win_pred_sf)
            Elo_2_sf_new = Elo_2_sf+K2_sf*(p2_win-p2_win_pred_sf)
    
            if Elo_1_sf_new>0:
                elo_list_hard[p1_id]=Elo_1_sf_new                
            if Elo_2_sf_new>0:
                elo_list_hard[p2_id]=Elo_2_sf_new
    
    
    
        elif surface == 'Clay':
            Elo_1_sf = elo_list_clay[p1_id]
            Elo_2_sf = elo_list_clay[p2_id]
    
            p1_win_pred_sf = 1/(1+10**((Elo_2_sf-Elo_1_sf)/400))
            p2_win_pred_sf = 1/(1+10**((Elo_1_sf-Elo_2_sf)/400))
    
            matrix_e.loc[index,'Elo_1_clay_s'+str(s)]=elo_list_clay[p1_id]
            matrix_e.loc[index,'Elo_2_clay_s'+str(s)]=elo_list_clay[p2_id]
    
            matrix_e.loc[index,'p1_win_pred_clay_s'+str(s)]=p1_win_pred_sf
            matrix_e.loc[index,'p2_win_pred_clay_s'+str(s)]=p2_win_pred_sf
    
            Elo_1_sf_new = Elo_1_sf+K1_sf*(p1_win-p1_win_pred_sf)
            Elo_2_sf_new = Elo_2_sf+K2_sf*(p2_win-p2_win_pred_sf)
    
            if Elo_1_sf_new>0:
                elo_list_clay[p1_id]=Elo_1_sf_new                
            if Elo_2_sf_new>0:
                elo_list_clay[p2_id]=Elo_2_sf_new
    
    
        elif surface == 'Grass':
            Elo_1_sf = elo_list_grass[p1_id]
            Elo_2_sf = elo_list_grass[p2_id]
    
            p1_win_pred_sf = 1/(1+10**((Elo_2_sf-Elo_1_sf)/400))
            p2_win_pred_sf = 1/(1+10**((Elo_1_sf-Elo_2_sf)/400))
    
            matrix_e.loc[index,'Elo_1_grass_s'+str(s)]=elo_list_grass[p1_id]
            matrix_e.loc[index,'Elo_2_grass_s'+str(s)]=elo_list_grass[p2_id]
    
            matrix_e.loc[index,'p1_win_pred_grass_s'+str(s)]=p1_win_pred_sf
            matrix_e.loc[index,'p2_win_pred_grass_s'+str(s)]=p2_win_pred_sf
    
            Elo_1_sf_new = Elo_1_sf+K1_sf*(p1_win-p1_win_pred_sf)
            Elo_2_sf_new = Elo_2_sf+K2_sf*(p2_win-p2_win_pred_sf)
    
            if Elo_1_sf_new>0:
                elo_list_grass[p1_id]=Elo_1_sf_new                
            if Elo_2_sf_new>0:
                elo_list_grass[p2_id]=Elo_2_sf_new
    
    
        elif surface == 'Carpet':
            Elo_1_sf = elo_list_carpet[p1_id]
            Elo_2_sf = elo_list_carpet[p2_id]
    
            p1_win_pred_sf = 1/(1+10**((Elo_2_sf-Elo_1_sf)/400))
            p2_win_pred_sf = 1/(1+10**((Elo_1_sf-Elo_2_sf)/400))
    
            matrix_e.loc[index,'Elo_1_carpet_s'+str(s)]=elo_list_carpet[p1_id]
            matrix_e.loc[index,'Elo_2_carpet_s'+str(s)]=elo_list_carpet[p2_id]
    
            matrix_e.loc[index,'p1_win_pred_carpet_s'+str(s)]=p1_win_pred_sf
            matrix_e.loc[index,'p2_win_pred_carpet_s'+str(s)]=p2_win_pred_sf
    
            Elo_1_sf_new = Elo_1_sf+K1_sf*(p1_win-p1_win_pred_sf)
            Elo_2_sf_new = Elo_2_sf+K2_sf*(p2_win-p2_win_pred_sf)
    
            if Elo_1_sf_new>0:
                elo_list_carpet[p1_id]=Elo_1_sf_new                
            if Elo_2_sf_new>0:
                elo_list_carpet[p2_id]=Elo_2_sf_new
    
        # show progress
        if index == 0:
            print('Start calculating Elo ratings:')
            
        sys.stdout.write('\r')
        progress = round(100*index/total_len);
        if progress % 5 == 0:
            sys.stdout.write("{}%".format(progress))      
            sys.stdout.flush()
            sleep(0.02)
    
    # combine the information for surface Elo ratings and winning prediction, for each match, only one of the four is nonzero
    
    matrix_e['Elo_1_sf_s'+str(s)]=matrix_e['Elo_1_hard_s'+str(s)].fillna(0)+matrix_e['Elo_1_clay_s'+str(s)].fillna(0)\
                            +matrix_e['Elo_1_grass_s'+str(s)].fillna(0)+matrix_e['Elo_1_carpet_s'+str(s)].fillna(0)
        
    matrix_e['Elo_2_sf_s'+str(s)]=matrix_e['Elo_2_hard_s'+str(s)].fillna(0)+matrix_e['Elo_2_clay_s'+str(s)].fillna(0)\
                                +matrix_e['Elo_2_grass_s'+str(s)].fillna(0)+matrix_e['Elo_2_carpet_s'+str(s)].fillna(0)
    
    matrix_e['p1_win_pred_sf_s'+str(s)]=matrix_e['p1_win_pred_hard_s'+str(s)].fillna(0)+matrix_e['p1_win_pred_clay_s'+str(s)].fillna(0)\
                                +matrix_e['p1_win_pred_grass_s'+str(s)].fillna(0)+matrix_e['p1_win_pred_carpet_s'+str(s)].fillna(0)
    
    matrix_e['p2_win_pred_sf_s'+str(s)]=matrix_e['p2_win_pred_hard_s'+str(s)].fillna(0)+matrix_e['p2_win_pred_clay_s'+str(s)].fillna(0)\
                                +matrix_e['p2_win_pred_grass_s'+str(s)].fillna(0)+matrix_e['p2_win_pred_carpet_s'+str(s)].fillna(0)
                                
    # the most important information is the difference of Elo ratings and the winning prediction
    matrix_e['Elo_1-2_s'+str(s)]=matrix_e['Elo_1_s'+str(s)]-matrix_e['Elo_2_s'+str(s)]
    matrix_e['Elo_sf_1-2_s'+str(s)]=matrix_e['Elo_1_sf_s'+str(s)]-matrix_e['Elo_2_sf_s'+str(s)]
    matrix_e['win_pred_1-2_s'+str(s)]=matrix_e['p1_win_pred_s'+str(s)]-matrix_e['p2_win_pred_s'+str(s)]
    matrix_e['win_pred_sf_1-2_s'+str(s)]=matrix_e['p1_win_pred_sf_s'+str(s)]-matrix_e['p2_win_pred_sf_s'+str(s)]
    
    # select the final columns from matrix_e to merge with the input matrix_n
    col_e = [item for item in matrix_e.columns if item not in col_i]
    matrix_n = pd.merge(matrix_n,matrix_e[col_e], on=matrix_n.index,how='left')
    
    print('finished')
    time_elapsed = round((time.time() - t)/60,1)
    print('Time elapsed: {} mins'.format(time_elapsed))
        
    return matrix_n


def p1p2_lastN(matrix_n, n_past_match, year_start,year_end,col_p = None):  #= min(matrix_n.year) = max(matrix_n.year)
    
    
    """ This is a function that takes the matrix with p1 and p2 information and return
    the averaged performance of every pair of players for their past n matches.
    
    col_p define the columns we want to calculate the averaged record
    
    We will loop through the matrix and every time we encounter a new pair of players,
    we calculate their averaged performance over the entire time span. By default, we
    used the minimal and maximaal years as the start and ending"""
    
    t = time.time()
    n = n_past_match
    total_len = len(matrix_n)
    # choose the part we want to calculate
    # if col_p = None, then we use the default values below
    if col_p == None:
        col_p = ['p1_id', 'p1_win','p1_sets_win', 
           'p1_ace', 'p1_df', 'p1_1stWon', 'p1_2ndWon', 'p1_bpSaved', 'p1_bpFaced',
           'p1_rank', 'p1_rank_points', 'p1_bpsr', 'p1_bpc', 'p1_bpcr', 'p1_seed_code',       
           'p2_id', 'p2_win', 'p2_sets_win',
           'p2_ace', 'p2_df', 'p2_1stWon', 'p2_2ndWon', 'p2_bpSaved', 'p2_bpFaced',
           'p2_rank', 'p2_rank_points', 'p2_bpsr', 'p2_bpc', 'p2_bpcr', 'p2_seed_code'
            ]
    
    col_p1 = [item for item in col_p if (item not in ('p1_id','p2_id')) and ('p1' in item) ]
    col_p2 = [item for item in col_p if (item not in ('p1_id','p2_id')) and ('p2' in item) ]
    
    # features for last N competitions
    col_p1_n = [item+'_last_'+str(n) for item in col_p1]
    col_p2_n = [item+'_last_'+str(n) for item in col_p2]
    col_p1p2_n = [item.replace('p1','1-2') for item in col_p1_n]
    
    # all the numerical features
    col_num = col_p1_n+col_p2_n+col_p1p2_n
    
    #select the years of data we want to processed
    matrix_i = matrix_n[(matrix_n.year>=year_start) & (matrix_n.year<=year_end)][col_p]  
    # pair code
    matrix_i['pair'] = matrix_n['p1_id'].astype(np.int64).map(str) +'/'+matrix_n['p2_id'].astype(np.int64).map(str)
    # add new columns for the moving averaged features
    matrix_i = pd.concat([matrix_i,pd.DataFrame(columns=col_num)])
    
    pair_list = matrix_i['pair'].unique()    
    pair_list = pd.Series(data = np.ones(len(pair_list))*0, index=pair_list)
    
    for row in matrix_i.itertuples():
        
        i = row.Index
        pair = matrix_i.loc[i,'pair']
        
        # if the pair has not been processed
        if pair_list[pair]==0:
            
            df_pair = matrix_i[matrix_i['pair'] == pair][col_p]
            df_pair = pd.concat([df_pair,pd.DataFrame(columns=col_num)])
            #i_min = min(df_pair.index)
            #i_max = max(df_pair.index)
            
            df_pair_reindex = df_pair.reset_index().drop('index',axis=1)
            
            k=0
            for j in df_pair.index:
                
                if k>0: 
                    
                    # use k-1, make sure do not include the current data point
                    df_pair.loc[j,col_p1_n] = df_pair_reindex.loc[0:k-1,col_p1].tail(n).fillna(0).mean().values
                    df_pair.loc[j,col_p2_n] = df_pair_reindex.loc[0:k-1,col_p2].tail(n).fillna(0).mean().values
                    df_pair.loc[j,col_p1p2_n] = df_pair.loc[j,col_p1_n].values - df_pair.loc[j,col_p2_n].values
                
                k+=1
                
            row_i = df_pair.index            
        
            matrix_i.loc[row_i,col_num] = df_pair.loc[row_i,col_num].values
            pair_list[pair]=1
            #print(pair)
            
        # show progress
        if i == 0:
            print('Start calculating record of last {} matches:'.format(n))
            
        sys.stdout.write('\r')
        progress = round(100*i/total_len);
        
        if progress % 2 == 0:
            sys.stdout.write("{}%".format(progress))      
            sys.stdout.flush()
            sleep(0.02)
            
        i+=1
    
    row_i = matrix_i.index 
    
    matrix_n = pd.concat([matrix_n,pd.DataFrame(columns = col_num)])
    matrix_n.loc[row_i,col_num] = matrix_i.loc[row_i,col_num].values
    
    matrix_n['1-2_age']=matrix_n['p1_age'].values-matrix_n['p2_age'].values
    matrix_n['1-2_ht']=matrix_n['p1_ht'].values-matrix_n['p2_ht'].values
    
    print('finished')
    time_elapsed = round((time.time() - t)/60,1)
    print('Time elapsed: {} mins'.format(time_elapsed))
        
    return matrix_n
        
def last_Nmonth(matrix_n, n_past_month, year_start,year_end,col_p = None):  
    
    """ This is a function that takes the matrix with p1 and p2 information and return
    the averaged performance of each player over their past N month
    
    We will loop through the matrix and every time we encounter a new pair of players,
    we calculate their averaged performance over the entire time span. By default, we
    used the minimal and maximaal years as the start and ending"""
    
    
    # start counting time
    t = time.time()
    
    # past n months
    n = n_past_month
    
    total_len = len(matrix_n)
    
    # choose the part we want to calculate
    # if col_p = None, then we use the default values below
    if col_p == None:
        col_p = ['tourney_date','p1_id', 'p1_win','p1_sets_win', 
           'p1_ace', 'p1_df', 'p1_1stWon', 'p1_2ndWon', 'p1_bpSaved', 'p1_bpFaced',
           'p1_rank', 'p1_rank_points', 'p1_bpsr', 'p1_bpc', 'p1_bpcr', 'p1_seed_code',       
           'p2_id', 'p2_win', 'p2_sets_win',
           'p2_ace', 'p2_df', 'p2_1stWon', 'p2_2ndWon', 'p2_bpSaved', 'p2_bpFaced',
           'p2_rank', 'p2_rank_points', 'p2_bpsr', 'p2_bpc', 'p2_bpcr', 'p2_seed_code'
            ]
    
    #get features for p1 and p2
    col_p1 = [item for item in col_p if (item not in ('p1_id','p2_id')) and ('p1' in item) ]
    col_p2 = [item for item in col_p if (item not in ('p1_id','p2_id')) and ('p2' in item) ]
    
    # features for last N competitions
    col_p1_n = [item+'_past_'+str(n)+'m' for item in col_p1]
    col_p2_n = [item+'_past_'+str(n)+'m' for item in col_p2]
    col_p1p2_n = [item.replace('p1','1-2') for item in col_p1_n]
    
    # all the numerical features
    col_num = col_p1_n+col_p2_n+col_p1p2_n
    
    #choose the years we are interested in
    matrix_i = matrix_n[(matrix_n.year>=year_start) & (matrix_n.year<=year_end)][col_p]
    matrix_i = pd.concat([matrix_i,pd.DataFrame(columns=col_num)])
    
    #get all the unique player id 
    player_list = set(list(matrix_i['p1_id'].astype(np.int64))+list(matrix_i['p2_id'].astype(np.int64)))
    #create a series to record if the player's history has been calculated
    player_list = pd.Series(data = np.ones(len(player_list))*0, index=player_list)
    
    for row in matrix_i.itertuples():
        
        i = row.Index
        player_1 = matrix_i.loc[i,'p1_id']
        player_2 = matrix_i.loc[i,'p2_id']
        
        # for each row, check if the record of p1 or p2 has been processed
        for player in [player_1,player_2]:
        
            # if the player has not been processed, calculate the performance sum over the past n month
            # using sum rather than mean here because it also reflects how many matches each player has played
            if player_list[player]==0:
                
                # select the records of player_1 or player_2
                df_p = matrix_i[(matrix_i['p1_id'] == player) | (matrix_i['p2_id'] == player)]
                
                # loop through the record and get the moving performance sum 
                for j in df_p.index:
                    
                    current_date = df_p.loc[j,'tourney_date'] 
                    
                    # find the date that was n month before, approximately n*30 days
                    previous_date = current_date-timedelta(days=n*30)
                    
                    cond = (df_p.tourney_date>=previous_date)&(df_p.tourney_date<current_date)
                    
                    # calculate the sum of the performance when player is recorded as p1 or p2 
                    rec_p1 = df_p[cond & (df_p.loc[j,'p1_id'] == player)][col_p1].sum().values
                    rec_p2 = df_p[cond & (df_p.loc[j,'p2_id'] == player)][col_p2].sum().values
                    
                    # add the sum performance information of past n month to matrix_i
                    if df_p.loc[j,'p1_id'] == player:
                        matrix_i.loc[j,col_p1_n] = rec_p1+rec_p2
                        
                    elif df_p.loc[j,'p2_id'] == player:
                        matrix_i.loc[j,col_p2_n] = rec_p1+rec_p2
                
                #label the player as processed in the list
                player_list[player] = 1  
                      
        # show progress
        if i == 0:
            print('Start calculating record of last {} matches:'.format(n))
            
        sys.stdout.write('\r')
        progress = round(100*i/total_len);
        if progress % 2 == 0:
            sys.stdout.write("{}%".format(progress))      
            sys.stdout.flush()
            sleep(0.02)
            
        i+=1
    
    # add the difference feature of p1 and p2
    matrix_i[col_p1p2_n] =  matrix_i[col_p1_n].values - matrix_i[col_p2_n].values
    
    row_i = matrix_i.index 
    
    matrix_n = pd.concat([matrix_n,pd.DataFrame(columns = col_num)])
    matrix_n.loc[row_i,col_num] = matrix_i.loc[row_i,col_num].values
    
    print('finished')
    time_elapsed = round((time.time() - t)/60,1)
    print('Time elapsed: {} mins'.format(time_elapsed))
    
   
    
    return matrix_n
    
        
        
    
    
    
     
  