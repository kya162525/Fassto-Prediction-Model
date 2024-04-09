import pandas as pd
import numpy as np
from preproc.tools import *
from preproc.config import *
import datetime

class PreProcessor:

    """
    Constructor
        cs_df : customer.csv like
        ga_df : Google Analytics log datas
        user_df : user_id.csv like
    """
    def __init__(self,
                 cs_df: pd.DataFrame,
                 ga_df: pd.DataFrame,
                 user_df: pd.DataFrame,
                 predict:bool=False,
                 verbose:bool=False):

        # 필요한 칼럼 가지고 있는지 체크
        if not self.column_check(cs_df, ga_df, user_df, predict): exit(1)
        self.verbose = verbose
        self.cs_df = cs_df
        self.ga_df = ga = init_ga(cs_df, ga_df, user_df, predict=predict)
        self.max_time = ga['event_timestamp'].max()

        self.df_before = ga[ga['curr_service_status'] == 1].copy()
        self.df_while = ga[ga['curr_service_status'] == 2].copy()
        self.df_after = ga[(ga['curr_service_status'] == 2) | (ga['curr_service_status'] == 3)].copy()
        self.df_whole = ga[(ga['curr_service_status'] == 1) | (ga['curr_service_status'] == 2) | (ga['curr_service_status'] == 3)].copy()
        if predict: self.ga_check(cs_df = self.cs_df, ga_df = self.df_whole) # ga 로그 없는 유저 있는지 체크
        
        self.df_before['event_hour_before'] = self.df_before.event_timestamp.dt.hour
        self.df_after['event_hour_after'] = self.df_after.event_timestamp.dt.hour
        self.df_whole['event_hour_whole'] = self.df_whole.event_timestamp.dt.hour

        # cs_df에서 ga데이터와 연결되지 않는 행 제거
        self.cs_df_before = cs_df[cs_df['cst_cd'].isin(self.df_before['cst_cd'])].drop_duplicates(subset='cst_cd', keep='first')
        self.cs_df_whole = cs_df[cs_df['cst_cd'].isin(self.df_whole['cst_cd'])].drop_duplicates(subset='cst_cd', keep='first')



    def column_check(self, cs_df: pd.DataFrame, ga_df: pd.DataFrame, user_df: pd.DataFrame,
                 predict:bool):
        df_names = ['customer.csv', 'ga.csv', 'user.csv']
        dfs = [cs_df, ga_df, user_df]
        columns_needed = [cs_needed, ga_needed, user_needed]
        if predict: columns_needed[0].remove('in_yn')

        ret = True
        for name, df, needed in zip(df_names, dfs, columns_needed):
            cols = list(df.columns)

            for col in needed:
                if col not in cols:
                    print(f"{col} column is not included in your {name}-like file.")
                    ret = False
        return ret

    def ga_check(self, cs_df: pd.DataFrame, ga_df: pd.DataFrame):
        # ga 기록이 없는 유저가 있는지 체크
        ret = True
        
        ga_cst_cd = ga_df['cst_cd'].unique().tolist()
        for cd in cs_df['cst_cd']:
            if cd not in ga_cst_cd:
                print(f"고객코드 {cd} 고객의 GA 로그가 없습니다.")
                ret = False
        return ret

    """
    get_preprocessed_data
        INPUT
            cols_for_final : 
            cols_to_log_scale : 
            cols_to_ohe : 
            cols_to_drop : 
            cols_to_time_scale
        OUTPUT
            preprocessed dataframe
    """
    def get_preprocessed_data(self,
                     applied=True,
                     cols_for_final: list = cols_for_approved,
                     cols_to_time_scale: list = cols_to_time_scale,
                     cols_to_ohe:list = cols_to_ohe,
                     cols_to_log_scale: list = cols_to_log_scale,
                     predict:bool = False,
                     ):
        if applied: cols_for_final = cols_for_applied
        ret = self.cs_df_before.copy() if applied else self.cs_df_whole.copy()

        if predict:
            cols_for_final.pop(0) # isin pop
            ret = self.cs_df # use whole data

        process = self.applied_methods if applied else self.approved_methods

        # make columns
        if self.verbose:
            print("Current : ")
            print(ret.columns)
            prev = list(ret.columns)
        for fun in process:
            ret = fun(self, cs_df=ret, applied=applied)
            if self.verbose:
                temp = list(ret.columns)
                print(set(temp).difference(set(prev)), end = f" added, shape {ret.shape}\n")
                prev = temp

        # scaling & ohe
        if len(cols_to_time_scale) > 0: ret = make_time_scaled_column(cs_df=ret, ga_df=self.ga_df,
                                                                      cols_to_scale=cols_to_time_scale,
                                                                      max_time=self.max_time)
        if len(cols_to_log_scale)>0: ret = make_log_column(cs_df = ret, cols_to_scale=cols_to_log_scale)
        if len(cols_to_ohe)>0: ret = make_one_hot_encoded(cs_df = ret, cols_to_ohe=cols_to_ohe)
        ret.fillna(0, inplace=True)
        ret.set_index('cst_cd', inplace=True)
        return ret[cols_for_final].copy()

    # cs
    def make_category(self,cs_df, applied): return make_category(cs_df, applied)
    def make_is_corp(self, cs_df, applied): return make_is_corp(cs_df, applied)
    def make_is_nfa(self, cs_df, applied): return make_is_nfa(cs_df, applied)
    def make_monthly_out_count(self, cs_df, applied): return make_monthly_out_count(cs_df, applied)
    def make_is_basic_fee(self, cs_df, applied): return make_is_basic_fee(cs_df, applied)
    def make_is_stand_cost(self, cs_df, applied): return make_is_stand_cost(cs_df, applied)
    # ga
    def make_num_log(self, cs_df, applied): return make_num_log(cs_df, self.df_before, self.df_after, self.df_whole, applied)
    def make_event_hour(self, cs_df, applied): return make_event_hour(cs_df, self.df_before, self.df_after, self.df_whole, applied)
    def make_event_hour_gini(self, cs_df, applied): return make_event_hour_gini(cs_df, self.df_before, self.df_after, self.df_whole, applied)
    def make_bundle_sequence_cnt(self, cs_df, applied): return make_bundle_sequence_cnt(cs_df, self.df_before, self.df_after, self.df_whole, applied)
    def make_is_from_sa(self, cs_df, applied): return make_is_from_sa(cs_df, self.df_whole, applied)
    def make_visit_count(self, cs_df, applied): return make_visit_count(cs_df, self.df_before, self.df_after, self.df_whole, applied)
    def make_page_visit(self, cs_df, applied): return make_page_visit(cs_df, self.df_before, self.df_after, self.df_whole, applied)

    applied_methods = [
        # customer
        make_is_corp,
        make_is_nfa,
        make_monthly_out_count,
        # ga
        make_num_log,
        make_event_hour,
        make_event_hour_gini,
        make_bundle_sequence_cnt,
        make_is_from_sa,
        make_visit_count,
        make_page_visit,
    ]

    approved_methods = [
        # customer
        make_is_corp,
        make_is_nfa,
        make_monthly_out_count,
        make_is_basic_fee,
        make_is_stand_cost,
        # ga
        make_num_log,
        make_event_hour,
        make_event_hour_gini,
        make_bundle_sequence_cnt,
        make_is_from_sa,
        make_visit_count,
        make_page_visit,
    ]
