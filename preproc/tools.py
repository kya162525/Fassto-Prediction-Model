import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from constant import constants as const
import re, os, pickle
import preproc.config as CONFIG
#################################################################################
#                                  general                                      #
#################################################################################
def make_log_column(cs_df: pd.DataFrame, cols_to_scale: list):
    cols_to_scale = list(set(cols_to_scale) & set(list(cs_df.columns)))
    for col in cols_to_scale:
        cs_df[col] = cs_df[col].fillna(-1).apply(lambda x: np.log1p(x) if x != -1 else np.NaN)
    return cs_df

def make_one_hot_encoded(cs_df: pd.DataFrame, cols_to_ohe:list):
    cols_to_ohe = list(set(cols_to_ohe) & set(list(cs_df.columns)))
    for col in cols_to_ohe:
        cs_df = pd.concat( [cs_df, pd.get_dummies(cs_df[col], prefix=col, drop_first=True) ], axis=1)
        cs_df.drop(columns = [col], inplace=True)
    return cs_df


def make_time_scaled_column(cs_df: pd.DataFrame, ga_df: pd.DataFrame, cols_to_scale: list, max_time: datetime, denom=1):
    cols_to_scale = list(set(cols_to_scale) & set(list(cs_df.columns)))

    # denom: division error 방지하기 위해 더해주는 term, default 1
    cs_df2 = cs_df.copy()
    first_visit = ga_df.sort_values(by='event_timestamp').drop_duplicates('cst_cd')[['cst_cd', 'event_timestamp']]
    first_visit = first_visit.rename(columns={'event_timestamp': 'first_visit_time'})
    cs_df2 = cs_df2.merge(first_visit, on='cst_cd', how='left')

    # 첫 접속부터 서비스 신청까지 소요시간
    cs_df2['bf_time'] = cs_df2.apply(
        lambda x: (x.service_apply_time - x.first_visit_time).total_seconds() / 86400 if pd.notnull(
            x.service_apply_time) and pd.notnull(x.first_visit_time) else np.nan, axis=1)

    # 서비스 신청부터 입고 요청까지 (혹은 현재까지) 소요시간
    cs_df2['aft_time'] = cs_df2.apply(
        lambda x: (x.fst_in_req_time - x.service_apply_time).total_seconds() / 86400 if pd.notnull(
            x.fst_in_req_time) else (max_time - x.service_apply_time).total_seconds() / 86400 if pd.notnull(
            x.service_apply_time) else np.nan, axis=1)

    idx = cs_df2[(pd.notnull(cs_df2.bf_time)) & (pd.notnull(cs_df2.aft_time))].index

    # 첫 접속부터 입고 요청까지 (혹은 현재까지) 소요시간
    cs_df2.loc[idx, 'whl_time'] = cs_df2.loc[idx, 'bf_time'] + cs_df2.loc[idx, 'aft_time']
    cs_df2 = cs_df2[['cst_cd', 'bf_time', 'aft_time', 'whl_time']].dropna(how='all')
    idx = cs_df2[cs_df2['bf_time'] < 0].index
    cs_df2.loc[idx, 'bf_time'] = denom
    cs_df2.drop_duplicates('cst_cd', inplace=True)

    cs_df = cs_df.merge(cs_df2, on='cst_cd', how='left')
    for col in cols_to_scale:
        if ("bf" in col) or ("before" in col):
            cs_df[col] /= ( cs_df['bf_time'] + denom )
        elif "aft" in col:
            cs_df[col] /= (cs_df['aft_time'] + denom)
        else:
             cs_df[col] /= (cs_df['whl_time'] + denom)
    cs_df.drop(columns=['bf_time', 'aft_time', 'whl_time'], inplace=True)
    return cs_df

def drop_columns(df:pd.DataFrame, cols_to_drop:list):
    cols_to_drop = list(set(cols_to_drop) & set(list(df.columns)))
    df.drop(columns = cols_to_drop, inplace = True)
    return df

#################################################################################
#                               Customer based                                  #
#################################################################################
def make_category(cs_df: pd.DataFrame, applied):
    cs_df['category'] = clean_categories(cs_df.goods_category)
    cs_df.drop(columns = ['goods_category'], inplace=True)
    return cs_df

def make_is_corp(cs_df: pd.DataFrame, applied):
    cs_df['is_corp'] = (cs_df['comp_type'] == '법인').astype(int)
    cs_df.drop('comp_type', axis= 1, inplace=True)
    return cs_df


def make_is_nfa(cs_df: pd.DataFrame, applied):
    cs_df['is_nfa'] = (cs_df['promotion'] == 'nfa').astype(int)
    cs_df.drop('promotion', axis=1, inplace=True)
    return cs_df


def make_monthly_out_count(cs_df: pd.DataFrame, applied):
    # under_100
    cs_df['monthly_out_count_under_100'] = (cs_df['monthly_out_count'] == '0건 이상~100건 미만').astype(int)
    cs_df.drop('monthly_out_count', axis=1, inplace=True)
    return cs_df

def make_is_basic_fee(cs_df: pd.DataFrame, applied):
    cs_df['is_basic_fee'] = (cs_df['fee_type'] == 'BASIC 요금제').astype(int)
    cs_df.drop('fee_type', axis=1, inplace=True)
    return cs_df


def make_is_stand_cost(cs_df: pd.DataFrame, applied):
    cs_df['is_stand_cost'] = (cs_df['cost_type'] == '홈페이지단가').astype(int)
    cs_df.drop('cost_type', axis=1, inplace=True)
    return cs_df



#################################################################################
#                                  GA based                                     #
#################################################################################


# 로그수
def make_num_log(cs_df: pd.DataFrame, df_before, df_after, df_whole, applied):
    num_log_before = df_before.groupby('cst_cd').size().reset_index(name='num_log_before')
    cs_df = pd.merge(cs_df, num_log_before, on='cst_cd', how='left')
    if applied: return cs_df

    num_log_after = df_after.groupby('cst_cd').size().reset_index(name='num_log_after')
    cs_df = pd.merge(cs_df, num_log_after, on='cst_cd', how='left')
    return cs_df



# 주 활동시각
def make_event_hour(cs_df: pd.DataFrame, df_before, df_after, df_whole, applied):
    if applied:
        event_hour_before = df_before.groupby(['cst_cd', 'event_hour_before']).size().reset_index(name='hour_count')
        event_hour_before = event_hour_before.sort_values(by='hour_count', ascending=False).drop_duplicates('cst_cd')
        event_hour_before = event_hour_before.drop(columns='hour_count')
        cs_df = pd.merge(cs_df, event_hour_before, on='cst_cd', how='left')
        return cs_df
    else:
        event_hour_after = df_whole.groupby(['cst_cd', 'event_hour_whole']).size().reset_index(name='hour_count')
        event_hour_after = event_hour_after.sort_values(by='hour_count', ascending=False).drop_duplicates('cst_cd')
        event_hour_after = event_hour_after.drop(columns='hour_count')
        event_hour_after.rename(columns={'event_hour_whole': 'event_hour_after'}, inplace=True)
        cs_df = pd.merge(cs_df, event_hour_after, on='cst_cd', how='left')
        return cs_df

# 활동 시각 지니계수
def make_event_hour_gini(cs_df: pd.DataFrame, df_before, df_after, df_whole, applied):
    if applied: 
        df_before = df_before.copy()
        df_before['event_hour_before'] = df_before.event_timestamp.dt.hour
        event_hour_gini_before = 1 - pow(df_before.groupby(['cst_cd', 'event_hour_before']).size() / df_before.groupby(['cst_cd']).size(), 2).groupby(level=0).sum()
        event_hour_gini_before = event_hour_gini_before.to_frame('event_hour_gini_before').reset_index()
        cs_df = pd.merge(cs_df, event_hour_gini_before, on='cst_cd', how='left')
        return cs_df
    else:
        df_after = df_whole.copy()
        df_after['event_hour_whole'] = df_after.event_timestamp.dt.hour
        event_hour_gini_after = 1 - pow(df_after.groupby(['cst_cd', 'event_hour_whole']).size() / df_after.groupby(['cst_cd']).size(), 2).groupby(level=0).sum()
        event_hour_gini_after = event_hour_gini_after.to_frame('event_hour_gini_after').reset_index()
        cs_df = pd.merge(cs_df, event_hour_gini_after, on='cst_cd', how='left')
        return cs_df

# event_bundle_sequence 개수
def make_bundle_sequence_cnt(cs_df: pd.DataFrame, df_before, df_after, df_whole, applied):
    bundle_sequence_cnt_before = df_before.groupby(['cst_cd', 'event_bundle_sequence_id']).size().groupby(level=0).count()
    bundle_sequence_cnt_before = bundle_sequence_cnt_before.to_frame('bundle_sequence_cnt_before').reset_index()
    cs_df = pd.merge(cs_df, bundle_sequence_cnt_before, on='cst_cd', how='left')
    if applied: return cs_df

    bundle_sequence_cnt_after = df_after.groupby(['cst_cd', 'event_bundle_sequence_id']).size().groupby(level=0).count()
    bundle_sequence_cnt_after = bundle_sequence_cnt_after.to_frame('bundle_sequence_cnt_after').reset_index()
    cs_df = pd.merge(cs_df, bundle_sequence_cnt_after, on='cst_cd', how='left')
    return cs_df


# 방문횟수
def make_visit_count(cs_df: pd.DataFrame, df_before, df_after, df_whole, applied):
    visit_count_before = df_before.groupby('cst_cd')['ga_session_number'].max().reset_index()
    visit_count_before = visit_count_before.rename(columns={'ga_session_number':'visit_count_before'})
    cs_df = pd.merge(cs_df, visit_count_before, on='cst_cd', how='left')
    if applied: return cs_df

    visit_count_after = df_whole.groupby('cst_cd')['ga_session_number'].max().reset_index()
    visit_count_after = visit_count_after.rename(columns={'ga_session_number': 'visit_count_after'})
    cs_df = pd.merge(cs_df, visit_count_after, on='cst_cd', how='left')
    return cs_df


# traffic_medium에서 sa 여부
def make_is_from_sa(cs_df: pd.DataFrame, df_whole, applied):
    is_from_sa = df_whole[['cst_cd', 'event_timestamp', 'traffic_medium']].sort_values('event_timestamp').drop_duplicates('cst_cd', keep='first')
    is_from_sa = is_from_sa[~is_from_sa['cst_cd'].isnull()]
    is_from_sa['is_from_sa'] = is_from_sa['traffic_medium'].apply(lambda x: 1 if x == 'sa' else 0)
    cs_df = pd.merge(cs_df, is_from_sa[['cst_cd', 'is_from_sa']], on='cst_cd', how='left')
    cs_df['is_from_sa'].fillna(0, inplace=True)
    return cs_df


def make_page_visit(cs_df: pd.DataFrame, df_before, df_after, df_whole, applied):

    df_bef = df_before.copy(); df_aft = df_after.copy()
    ## dummy user data (for page labeling)
    for df in [df_bef, df_aft]:
        df.reset_index(drop=True, inplace=True)
        for page in CONFIG.titles_to_use:
            idx = len(df)
            df.loc[idx] = ''
            df.loc[idx, 'page_label'] = page
            df.loc[idx, 'cst_cd'] = CONFIG.dummy_cstcd

    page_visit_count_before = df_bef.groupby(['cst_cd', 'page_label']).size().to_frame('cnt').reset_index().pivot(
        index='cst_cd', columns='page_label', values='cnt').add_prefix('bf_cnt_').reset_index().fillna(0)
    cs_df = pd.merge(cs_df, page_visit_count_before, on='cst_cd', how='left')
    if applied: return cs_df

    page_visit_count_after = df_aft.groupby(['cst_cd', 'page_label']).size().to_frame('cnt').reset_index().pivot(
      index='cst_cd', columns='page_label', values='cnt').add_prefix('aft_cnt_').reset_index().fillna(0)
    cs_df = pd.merge(cs_df, page_visit_count_after, on='cst_cd', how='left')
    return cs_df


#################################################################################
#                               GA initialize                                   #
#################################################################################
def init_ga(cs_df, ga_df, user_df, predict:bool, except_n=10):
    # cst_cd & 입고 여부 합치기

    # in_yn 없는 파일 경우
    if predict:
        ga_df = ga_df.merge(user_df[['user_id', 'cst_cd']].drop_duplicates(['user_id']), how='left',
                            on='user_id')
        ga_df = ga_df.merge(user_df[['user_pseudo_id', 'cst_cd']].drop_duplicates(['user_pseudo_id']),
                            how='left', on='user_pseudo_id')
        ga_df['cst_cd'] = ga_df['cst_cd_x'].combine_first(ga_df['cst_cd_y'])
        ga_df.drop(columns=['cst_cd_x', 'cst_cd_y'], inplace=True)
    else:
        cs_df.rename(columns={'in_yn':'is_in'}, inplace=True)
        in_yn_info = user_df.merge(cs_df[['cst_cd', 'is_in']], how='left')
        ga_df = ga_df.merge(in_yn_info[['user_id', 'is_in', 'cst_cd']].drop_duplicates(['user_id']), how='left', on='user_id')
        ga_df = ga_df.merge(in_yn_info[['user_pseudo_id', 'is_in', 'cst_cd']].drop_duplicates(['user_pseudo_id']), how='left', on='user_pseudo_id')
        ga_df['is_in'] = ga_df['is_in_x'].combine_first(ga_df['is_in_y'])
        ga_df['cst_cd'] = ga_df['cst_cd_x'].combine_first(ga_df['cst_cd_y'])
        ga_df.drop(columns=['is_in_x', 'is_in_y', 'cst_cd_x', 'cst_cd_y'], inplace=True)

    # datetime으로 변환
    ga_df['event_timestamp'] = pd.to_datetime(ga_df['event_timestamp'] / 1000000, unit='s') + timedelta(hours=9)
    idx = cs_df.query('service_approval_time== "0000-00-00 00:00:00"').index
    cs_df.loc[idx, 'service_approval_time'] = np.nan
    idx = cs_df.query('fst_in_req_time== "0000-00-00 00:00:00"').index
    cs_df.loc[idx, 'fst_in_req_time'] = np.nan
    cols_for_datetime = ['service_apply_time', 'service_approval_time', 'fst_in_req_time']
    cs_df[cols_for_datetime] = cs_df[cols_for_datetime].apply(pd.to_datetime)

    # 활동 시각 추가
    ga_df['event_hour'] = ga_df['event_timestamp'].dt.hour

    # 페이지 라벨
    ga_df['page_label'] = ga_df['page_title'].apply(lambda x: const.page_label[x] if x in const.page_label.keys() else '기타')

    # 10일 이내 첫 방문자 제거
    if not predict:
        first_visit = ga_df.sort_values(by='event_timestamp').drop_duplicates('cst_cd')
        first_visit = first_visit[(first_visit['event_timestamp']>ga_df['event_timestamp'].max()-timedelta(10))&(first_visit['is_in']==0)].cst_cd
        idx = ga_df[ga_df['cst_cd'].isin(list(first_visit))].index
        ga_df.drop(index=idx, inplace=True)

    # 서비스 상태 라벨 - 신청전 1 | 심사중 2 | 승인후 입고전 3 | 입고후 4
    apply_time = cs_df.query('service_apply_time.isna()==False')[['cst_cd', 'service_apply_time']].set_index('cst_cd')
    apply_time = apply_time.to_dict()['service_apply_time']
    approval_time = cs_df.query('service_approval_time.isna()==False')[['cst_cd', 'service_approval_time']].set_index('cst_cd')
    approval_time = approval_time.to_dict()['service_approval_time']
    fst_in_req_time = cs_df.query('fst_in_req_time.isna()==False')[['cst_cd', 'fst_in_req_time']]
    fst_in_req_time = fst_in_req_time.set_index('cst_cd').to_dict()['fst_in_req_time']


    def status_labeling(df):
        if (df['cst_cd'] in apply_time.keys()):
            apply = apply_time[df['cst_cd']]
            if df['event_timestamp'] < apply: return 1
            if (df['cst_cd'] in approval_time.keys()):
                approval = approval_time[df['cst_cd']]
                if df['event_timestamp'] < approval: return 2
                if (df['cst_cd'] in fst_in_req_time.keys()):
                    fst_in = fst_in_req_time[df['cst_cd']]
                    if df['event_timestamp'] < fst_in:
                        return 3
                    else:
                        return 4
                else:
                    return 3
            else:
                return 2

    ga_df['curr_service_status'] = ga_df.apply(lambda x: status_labeling(x), axis=1)

    return ga_df

#################################################################################
#                                   Legacy                                      #
#################################################################################
def standardize_categories(cat: pd.Series):
    # 고객사가 직접 입력한 카테고리(자유형식)를 정형화
	# 첫 번째로 등장하는 단어만 사용 (예: '컴퓨터/노트북/태블릿' -> '컴퓨터')
	cat = cat.apply(lambda x: x[0:re.search(r'[>/, ]', x).start()] if re.search(r'[>/, ]', x) else x)
	# 사용되는 표현 통일 (예: '옷', '패션', '패션의류', ... -> '의류')
	cat.replace(const.category, inplace=True)
	return cat


def get_major_categories(min_encounter=10):
	# 전체 서비스신청 이력 중에서 n회 이상 등장(정형화 후 기준)한 카테고리만 사용
	major = pd.read_csv(os.path.join('data', 'category.csv'))
	major.category = standardize_categories(major.category)
	major = major.groupby('category').sum()
	major = major[major.cnt > min_encounter].index.tolist()
	return major


def clean_categories(cat: pd.Series):
	major_cats = get_major_categories()
	cat = standardize_categories(cat)
	cat = cat.map(lambda x: x if x in major_cats else '기타')
	return cat


def clean_features(df: pd.DataFrame):
	df.category = clean_categories(df.category)
	df.sku = np.log(df.sku + 1)
	df.region = df.region.replace(const.region)

	return df

def one_hot_encoding(df: pd.DataFrame, segment, train=False):
	if train == True:
		for p in ['bef_pages', 'aft_pages']:
			df_pages = (df[p].explode().str.get_dummies().groupby(level=0).sum().add_prefix(p + '_'))
			df = df.drop(p, axis=1).join(df_pages)
		df = pd.get_dummies(df)
	else:
		cols = pickle.load(open(f'pickles/cols_{segment}.pkl', 'rb'))
		df = pd.get_dummies(df)
		df = df.reindex(columns=cols, fill_value=0)
	return df
