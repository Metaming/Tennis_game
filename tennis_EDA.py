# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:41:40 2020

@author: Mingkai Liu

This is a module for exploratory data analysis for the tennis dataset


"""

import numpy as np
import pandas as pd 
#import scipy as sp

import seaborn as sns
import matplotlib.pyplot as plt

#from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


def dist_bar_plot(matrix,col_match,n_row,n_column,hist_color,flag_kde,bins_num,top_n):
    
    """This function take the dataframe of the tennius match and return plots 
    for the chosen columns col_match
    
    if the column is numerical feature, then plot the distribution plot,with 'color','bins','flag_kde' as parameters
    
    if the column is categorical, then plot the top N categories with the most counts
    
    n_row and n_column are the number of row and column for the figure"""
               
    
    for i,col in enumerate(col_match):
    
        if  is_numeric_dtype(matrix[col]):#in (np.float64,np.int64,np.float32,np.int32):
            
            # if the column is numeric features, plot the distribution
            plt.subplot(n_row,n_column,i+1)
            
            if bins_num == 'auto':
                
                sns.distplot(matrix[col],kde=flag_kde,hist=True,color=hist_color,hist_kws={'align':'mid'})
                            
            else:
                sns.distplot(matrix[col],kde=flag_kde,hist=True,color=hist_color,hist_kws={'align':'mid'},bins=bins_num)
            
            plt.grid()
            
        elif (matrix[col].dtypes == np.object):
            
            # if the column is categorical features, plot the number of the top 10 category
            group=matrix.groupby(col).agg({col:'count'})
            group.rename(columns = {col:'count'},inplace=True)
            group = group.sort_values(by='count',ascending = False)
            group = group.reset_index()
            
            plt.subplot(n_row,n_column,i+1)
            sns.barplot(x=col,y='count',data = group[0:top_n],order=group.loc[0:top_n-1,col])
            plt.xticks(rotation = -80)
            plt.grid()
            
    plt.tight_layout() 
    
    return 
                 
    
    
         
def player_record(matrix,player_name,year_min,year_max,year_mov_ave,flag_plot = False,flag_legend=False):
    

    
    """This function take the dataframe of the tennis match, the player name and year range,
    return 2 dataframes:
        'matrix_player_new' includes the full compitition records,
    and 'matrix_player_year_ave'is the performance average over the years,
    
    The parameter 'year_mov_ave' defines the number of years we want to average, 
    e.g. year_mov_ave = n means averaged over the current year and the past n-1 year
    
    flag_plot = True means to plot the moving average perfomrance,
    flag_legend = True means to add legend based on player name. Which will be useful when using this function
    to plot multiple players. 
    """
    
    player_id = matrix[(matrix['winner_name'] == player_name)]['winner_id'].unique()[0]
    matrix_player = matrix[(matrix['winner_id'] == player_id) | (matrix['loser_id'] == player_id)].reset_index()
    matrix_player.drop('index',axis=1,inplace=True)
    matrix_player = matrix_player[(matrix_player.year>=year_min) & (matrix_player.year<=year_max)]
    #win_pf = matrix_player[(matrix_player['winner_id'] == player_id)]
    #los_pf = matrix_player[(matrix_player['loser_id'] == player_id)]
    
    print('From {}-{},{} has played in {} tournaments, won {} tournaments; in total played {} matches, and won {} matches.'
      .format(year_min,year_max,player_name,matrix_player['tourney_date'].nunique(),
              matrix_player[(matrix_player['winner_id'] == player_id)&(matrix_player['round'] =='F')].shape[0],
              matrix_player.shape[0],matrix_player[matrix_player['winner_id'] == player_id].shape[0])  )
    
    
    col_match = ['tourney_name','surface','draw_size','tourney_level','tourney_date','best_of','match_num','round','minutes','year','month','day','day_week']
    col_player_pf_w = ['winner_id','winner_name','winner_age','winner_seed_code','winner_rank','winner_rank_points',
                       'winner_ace','winner_df','winner_svpt', 'winner_1stIn','winner_1stWon','winner_2ndWon','winner_SvGms','winner_bpFaced','winner_bpSaved','winner_bpc','winner_bpsr','winner_bpcr']
    col_player_pf_l = ['loser_id','loser_name','loser_age','loser_seed_code','loser_rank', 'loser_rank_points',
                       'loser_ace','loser_df','loser_svpt', 'loser_1stIn','loser_1stWon','loser_2ndWon','loser_SvGms','loser_bpFaced','loser_bpSaved','loser_bpc','loser_bpsr','loser_bpcr']           


    col_p1 = ['p1_id','p1_name','p1_age','p1_seed_code','p1_rank','p1_rank_points','p1_ace','p1_df','p1_svpt', 'p1_1stIn','p1_1stWon','p1_2ndWon','p1_SvGms','p1_bpFaced','p1_bpSaved','p1_bpc','p1_bpsr','p1_bpcr']
    col_p2 = ['p2_id','p2_name','p2_age','p2_seed_code','p2_rank','p2_rank_points','p2_ace','p2_df','p2_svpt', 'p2_1stIn','p2_1stWon','p2_2ndWon','p2_SvGms','p2_bpFaced','p2_bpSaved','p2_bpc','p2_bpsr','p2_bpcr']

    matrix_player_new = pd.DataFrame()
    matrix_player_new[col_match] = matrix_player[col_match]
    
    matrix_player_new[col_p1]=matrix_player[matrix_player['winner_id'] == player_id][col_player_pf_w]
    matrix_player_new[col_p2]=matrix_player[matrix_player['winner_id'] == player_id][col_player_pf_l]
    matrix_player_new['p1_win'] = matrix_player_new['p1_id'].map(lambda x: 1 if x==player_id else 1, na_action = 'ignore').fillna(0)
     
    for i in range(len(col_p1)):
        matrix_player_new[col_p1[i]].fillna(matrix_player[matrix_player['loser_id'] == player_id][col_player_pf_l[i]],inplace = True)
        matrix_player_new[col_p2[i]].fillna(matrix_player[matrix_player['loser_id'] == player_id][col_player_pf_w[i]],inplace = True)

    
    col_add = ['p1_age','p1_win','p1_rank','p1_rank_points','p1_seed_code','p1_ace','p1_df','p1_svpt', 'p1_1stIn','p1_1stWon','p1_2ndWon',
               'p1_bpFaced','p1_bpSaved','p1_bpc','p1_bpsr','p1_bpcr']
    
    matrix_player_year_ave = pd.DataFrame()
    
    for i in range(year_min,year_max):
        
        group_mean = matrix_player_new[(matrix_player_new.year<=i)&(matrix_player_new.year>=i-year_mov_ave+1)][col_add].mean().T
        group_mean['year'] = i
        group_mean['p1_age'] = matrix_player_new[(matrix_player_new.year==i)]['p1_age'].mean()
        matrix_player_year_ave = matrix_player_year_ave.append(group_mean,ignore_index=True)
          
    
    if flag_plot == True:
        
                
        for i in range(0,len(col_add)):    
            plt.subplot(4,4,i+1)
            plt.grid()    
            plt.plot(matrix_player_year_ave.year,matrix_player_year_ave[col_add[i]],'*-')
            
            plt.ylabel(col_add[i]);
            
            if flag_legend == True:
                plt.legend([player_name])
                
        plt.tight_layout()
    
    return matrix_player_new,matrix_player_year_ave


def player_compare(matrix, player_names,year_min,year_max,year_mov_ave,flag_plot = True,flag_legend=True):
    
    
    players_performance = {}
    
    for i, player in enumerate(player_names):
        _,player_year_ave = player_record(matrix, player,year_min,year_max,year_mov_ave,flag_plot = False,flag_legend=False)
        
        players_performance[player] = player_year_ave

    col_add = ['p1_age','p1_win','p1_rank','p1_rank_points','p1_seed_code','p1_ace','p1_df','p1_svpt', 'p1_1stIn','p1_1stWon','p1_2ndWon',
               'p1_bpFaced','p1_bpSaved','p1_bpc','p1_bpsr','p1_bpcr']
    
    if flag_plot == True:        
                
        for i in range(0,len(col_add)):    
            plt.subplot(4,4,i+1)
            plt.grid()
            
            for player in player_names:
                matrix_player_year_ave = players_performance[player]
                plt.plot(matrix_player_year_ave.year,matrix_player_year_ave[col_add[i]],'^-')
            
            plt.ylabel(col_add[i],fontsize=12);
            
            if flag_legend == True:
                plt.legend(player_names)
                
        plt.tight_layout()
        
        

def player_vs_opponent(matrix,player_name,year_min,year_max,year_mov_ave=1,flag_plot=False):
    
    """This function take the dataframe of the tennis match, the player name, year range, and the number of years of moving_average
    return two dataframes: 'matrix_player_new' includes the full compitition records
    and 'matrix_player_year_ave' and 'matrix_player_year_dev' return the performance average and std over the years
    
    """
    
    # df is the dataframe with record of a player generated using function 'player_record()'
    
    df,_ = player_record(matrix,player_name,year_min-year_mov_ave+1,year_max,year_mov_ave)
    
    player_opponent_record = pd.DataFrame()
    col_group = ['p2_name']

    for i in range(year_min,year_max):
        
        # moving average performance
        
        # number of matches
        group = df[(df.year<=i)&(df.year>=i-year_mov_ave+1)].groupby(col_group).agg({'p1_win':'count'}).reset_index()
        group.rename(columns={'p1_win':'comp_cnt_past_'+str(year_mov_ave)+'yr'},inplace=True)
        
        #number of wins
        group1 = df[(df.year<=i)&(df.year>=i-year_mov_ave+1)].groupby(col_group).agg({'p1_win':'sum'}).reset_index()
        group1.rename(columns={'p1_win':'win_cnt_past_'+str(year_mov_ave)+'yr'},inplace=True)

        group['win_cnt_past_'+str(year_mov_ave)+'yr']=group1['win_cnt_past_'+str(year_mov_ave)+'yr']

        group['year'] = i
        group['year']=group['year'].astype(np.int64)
        
        #combine all things in a dataframe
        player_opponent_record = pd.concat([player_opponent_record,group],ignore_index=True,sort = True)

    player_opponent_record['win_ratio_past_'+str(year_mov_ave)+'yr'] = player_opponent_record['win_cnt_past_'+str(year_mov_ave)+'yr']/player_opponent_record['comp_cnt_past_'+str(year_mov_ave)+'yr']
    
    
    # sort the opponents according to the total number of competition
    top_opponents = pd.DataFrame()
    col_group = ['p2_name']
    top_opponents['num_comp'] = df[(df.year<=year_max)&(df.year>=year_min)].groupby(col_group)['p2_name'].count().sort_values(ascending=False)
    top_opponents['num_win'] = df[(df.year<=year_max)&(df.year>=year_min)].groupby(col_group)['p1_win'].sum()
    top_opponents['win_ratio'] = top_opponents['num_win']/top_opponents['num_comp']
    top_opponents = top_opponents.fillna(0)
    top_opponents=  top_opponents.reset_index()
    
    
    # if flag_plot = true, plot the top 6 opponents
    if flag_plot == True:
    
        # for m in range(0,3):
        # player = player_list[m]
        # player_5y,player_opponents = win_record_mavg(matrix,player,2009,2020)
    
        # display the top 6 opponents
        name_opp = top_opponents[top_opponents.index<6]['p2_name']
    
        fig, axes = plt.subplots(2,3, figsize = (14,6), sharey=True)
    
        for i, ax in enumerate(axes):
    
            for j,ax1 in enumerate(ax):
    
                x = player_opponent_record[player_opponent_record.p2_name ==name_opp[len(ax)*i+j]]['year']
                x = round(x)
                y1 = player_opponent_record[player_opponent_record.p2_name ==name_opp[len(ax)*i+j]]['comp_cnt_past_'+str(year_mov_ave)+'yr']
                y2 = player_opponent_record[player_opponent_record.p2_name ==name_opp[len(ax)*i+j]]['win_ratio_past_'+str(year_mov_ave)+'yr']
    
                ax2 = ax1.twinx()
                ax1.bar(x, y1, color=(0.3,0.6,0.3,0.8)) 
                #ax1.bar(x, y1, color='green')
                ax2.plot(x, y2, 'o-', color=(0.6,0.1,0.2,0.8))
                #ax2.plot(x, y2, color='red')
                
                ax1.set_xlabel('Year')
                ax1.set_ylabel('Num. match_past_'+str(year_mov_ave)+'yr', color='green')
                ax2.set_ylabel('Win ratio past_'+str(year_mov_ave)+'yr', color='red')
                
                xticks = np.arange(min(x),max(x)+1,3)
                ax1.set_xticks(xticks)
                plt.title(player_name+' vs '+name_opp[len(ax)*i+j])
                #plt.grid()
    
        plt.tight_layout()       
        
             
    return player_opponent_record,top_opponents



def player_opponent_lastN(matrix,player_1_name,player_2_name,year_min,year_max, n_past_match,flag_plot=False):
    
    """ This is a function that takes the matrix with winner and loser information, 
    return the average performance of two chosen players when
    they compete with each other over the last N matches"""
    
    n = n_past_match

    df = matrix[((matrix.winner_name == player_1_name) & (matrix.loser_name ==player_2_name))|\
                     ((matrix.loser_name == player_1_name) & (matrix.winner_name ==player_2_name))].reset_index()

    df.drop('index',axis=1,inplace=True)

    df = df[(df.year>=year_min) & (df.year<=year_max)]
    

    print('From {} to {}, {} and {} have competed in {} tournaments, in total played {} matches, {} won {} matches, {} won {} matches.'
          .format(year_min,year_max,player_1_name,player_2_name,df['tourney_date'].nunique(),df.shape[0],
              player_1_name,df[df.winner_name == player_1_name].shape[0],
              player_2_name,df[df.winner_name == player_2_name].shape[0]
             ))
    
    import data_cleansing
    # return a new matrix labeled in p1 and p2
    df_n = data_cleansing.matrix_p1p2(df).reset_index().drop('index',axis=1)
    
    #print(df_n.columns)
    
    col_p = ['tourney_date',
        'p1_id', 'p1_win','p1_sets_win', 'p1_age','p1_ht',
           'p1_ace', 'p1_df', 'p1_1stWon', 'p1_2ndWon', 'p1_bpSaved', 'p1_bpFaced',
           'p1_rank', 'p1_rank_points', 'p1_bpsr', 'p1_bpc', 'p1_bpcr', 'p1_seed_code',       
           'p2_id', 'p2_win', 'p2_sets_win','p2_age','p2_ht',
           'p2_ace', 'p2_df', 'p2_1stWon', 'p2_2ndWon', 'p2_bpSaved', 'p2_bpFaced',
           'p2_rank', 'p2_rank_points', 'p2_bpsr', 'p2_bpc', 'p2_bpcr', 'p2_seed_code'
            ]
    
    col_p1 = [item for item in col_p if (item not in ('p1_id','p2_id')) and ('p1' in item)]
    col_p2 = [item for item in col_p if (item not in ('p1_id','p2_id')) and ('p2' in item)]
    
    # features for last N competitions
    col_p1_n = [item+'_last_'+str(n) for item in col_p1]
    col_p2_n = [item+'_last_'+str(n) for item in col_p2]
    col_p1p2_n = [item.replace('p1','1-2') for item in col_p1_n]
    
    col_num = col_p1_n+col_p2_n+col_p1p2_n
    
    # print(df_n.columns)
    
    df_n = pd.concat([df_n,pd.DataFrame(columns = col_num)])
    
    # print(df_n.columns)
    # print(col_num)

    # # choose the numerical features to calculate the moving average
    # col_p1 = [item for item in df_n.columns if 'p1' in item 
    #       and df_n[item].dtypes in [np.float64,np.float32,np.int32,np.int64]] 
    # col_p1 = [item for item in col_p1 if item not in ('p1_id','p1_age','p1_seed','p1_hand_code')]
    
    # col_p2 = [item for item in df_n.columns if 'p2' in item 
    #       and df_n[item].dtypes in [np.float64,np.float32,np.int32,np.int64]]
    # col_p2 = [item for item in col_p2 if item not in ('p2_id','p2_age','p2_seed','p2_hand_code')]
    
    # col_p1_n = [item+'_last_'+str(n) for item in col_p1]
    # col_p2_n = [item+'_last_'+str(n) for item in col_p2]
    # col_p1p2_n = [item.replace('p1','1-2') for item in col_p1_n]
    
    #print(col_p1_n)
    
    
    for row in df_n.itertuples():
        i = row.Index
        
        if i>0:
            df_n.loc[i,col_p1_n] = df_n.loc[0:i-1,col_p1].tail(n).fillna(0).mean().values
            df_n.loc[i,col_p2_n] = df_n.loc[0:i-1,col_p2].tail(n).fillna(0).mean().values
            df_n.loc[i,col_p1p2_n] = df_n.loc[i,col_p1_n].values - df_n.loc[i,col_p2_n].values
            
        i+=1
    
    df_n[col_num] = df_n[col_num].fillna(0)   
    
    
    
    if flag_plot == True:
        
        # fig, ax1 = plt.subplots(2,3, sharey=True)        
         
        # ax2 = ax1.twinx()
        # sns.lineplot(data=df_n,x='tourney_date',y='1-2_win_last_'+str(n),ax=ax1,color='red',markers=['o'])
        # ax1.set_ylabel('1-2_win_ratio_last_'+str(n),color='red')
        # sns.lineplot(data=df_n,x='tourney_date',y='1-2_sets_win_last_'+str(n),ax=ax2,color='green',markers=['*'])
        # ax2.set_ylabel('1-2_sets_win_mean_last_'+str(n),color='green')
        # plt.title('p1:'+df_n['p1_name'][0]+' vs '+ 'p2:'+df_n['p2_name'][0])

        for i in range(0,len(col_p1p2_n)):    
            plt.subplot(4,4,i+1)
            plt.grid()
            
            plt.plot(df_n['tourney_date'],df_n[col_p1p2_n[i]],'^-')
            
            plt.ylabel(col_p1p2_n[i]);
            
        plt.suptitle('p1:'+df_n['p1_name'][0]+' vs '+ 'p2:'+df_n['p2_name'][0])
            
        plt.tight_layout()        
                
    return df_n


def player_opponent_match(matrix,col_match,player_1_name,player_2_name,year_min,year_max):
    
    """"This function takes the matrix and two chosen player, and return the number of games won by each player
    for different match-related parameters"""
    n_past_match = 5
    
    df_n = player_opponent_lastN(matrix,player_1_name,player_2_name,year_min,year_max, n_past_match,flag_plot=False)
    
    df_n[['draw_size','year','month','day','best_of','day_week','match_num']] = df_n[['draw_size','year','month','day','best_of','day_week','match_num']].astype(np.int64)

    plt.figure(figsize=(15,8))
    
    
    
    for i in range(len(col_match)):

        col=col_match[i]
        plt.subplot(2,4,i+1)

        if df_n[col].dtypes in (np.float64,np.float32):

        # if the column is numeric features, plot the distribution
            plt.grid()
            n_bins = range(70,300,15)
            
            # use .histogram to return the counts at each bins and the bin edges
            bin_bottom, bin_edge = np.histogram(df_n[df_n.p1_win == 1][col], bins=n_bins)

            plt.hist(df_n[df_n.p1_win == 1][col], n_bins, density=False, histtype='bar', stacked=False,color='#7f6d5f')
            plt.hist(df_n[df_n.p2_win == 1][col], n_bins, density=False, histtype='bar', stacked=False,color='#2d7f5e',bottom=bin_bottom)
            plt.legend([df_n['p1_name'][0],df_n['p2_name'][0]])
            plt.xlabel(col)


        elif (df_n[col].dtypes in (np.int64,np.object)):

            # if the column is categorical features, plot the number of the top 10 category
            group = df_n.groupby(col).agg({col:'count','p1_win':'sum','p2_win':'sum'})
            group.rename(columns = {col:'count'},inplace=True)
            group = group.reset_index()

            group = group.sort_values(by='count',ascending = False)
            group = group.reset_index()

            # Create brown bars
            plt.bar(group[col],group['p1_win'], color='#7f6d5f', edgecolor='white')
            plt.bar(group[col],group['p2_win'], bottom=group['p1_win'],color='#2d7f5e', edgecolor='white')

            plt.xticks(rotation = -85)
            plt.legend([df_n['p1_name'][0],df_n['p2_name'][0]])
            plt.xlabel(col)

    plt.suptitle('Number of games won by {} and {} from {} to {}'.format(df_n['p1_name'][0],df_n['p2_name'][0],year_min,year_max))
    
    plt.tight_layout()
    
    return  