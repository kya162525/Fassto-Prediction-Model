"""
    Dummy user cst_cd
"""

dummy_cstcd = 987654321

"""
    pages
"""
titles_to_drop = [
    '고객 후기', '공지', '기업 정보(채용)', '기타','서비스 소개', '파스토셀프', '해외'
]

titles_to_use = [
    '1:1 문의', '배송', '서비스 신청', '쇼핑몰 연동', '신규 상품 등록', '요금', '입고', '출고'
    , '이용 가이드(기타)', '이용 전 가이드(기타)'
]

titles_total = titles_to_drop + titles_to_use

pages_to_drop = ['bf_cnt_' + title for title in titles_to_drop] + \
    ['aft_cnt_' + title for title in titles_to_drop] + \
    ['whl_cnt_' + title for title in titles_total]

pages_to_use = ['bf_cnt_' + title for title in titles_to_use] + \
    ['aft_cnt_' + title for title in titles_to_use]
"""
    General
"""
cols_for_applied = [
    "is_in",

    "is_recommended",
    "is_corp",
    "is_nfa",
    "is_from_sa",
    "monthly_out_count_under_100",

    "num_log_before",
    "event_hour_before",
    "event_hour_gini_before",
    "bundle_sequence_cnt_before",
    "visit_count_before",
    "bf_cnt_1:1 문의",
    "bf_cnt_배송",
    "bf_cnt_서비스 신청",
    "bf_cnt_쇼핑몰 연동",
    "bf_cnt_신규 상품 등록",
    "bf_cnt_요금",
    "bf_cnt_이용 가이드(기타)",
    "bf_cnt_이용 전 가이드(기타)",
    "bf_cnt_입고",
    "bf_cnt_출고"
]

cols_for_approved = [
    "is_in",

    "is_recommended",
    "car_out",
    "is_corp",
    "is_nfa",
    "is_from_sa",
    "monthly_out_count_under_100",

    "is_basic_fee",
    "is_stand_cost",
    "num_log_before",
    "num_log_after",
    "event_hour_after",
    "event_hour_gini_after",
    "bundle_sequence_cnt_before",
    "bundle_sequence_cnt_after",
    "visit_count_before",
    "visit_count_after",
    "bf_cnt_1:1 문의",
    "bf_cnt_배송",
    "bf_cnt_서비스 신청",
    "bf_cnt_쇼핑몰 연동",
    "bf_cnt_신규 상품 등록",
    "bf_cnt_요금",
    "bf_cnt_이용 가이드(기타)",
    "bf_cnt_이용 전 가이드(기타)",
    "bf_cnt_입고",
    "bf_cnt_출고",
    "aft_cnt_1:1 문의",
    "aft_cnt_배송",
    "aft_cnt_서비스 신청",
    "aft_cnt_쇼핑몰 연동",
    "aft_cnt_신규 상품 등록",
    "aft_cnt_요금",
    "aft_cnt_이용 가이드(기타)",
    "aft_cnt_이용 전 가이드(기타)",
    "aft_cnt_입고",
    "aft_cnt_출고",
]

cols_to_drop = [
    'wh_cd',
    'service_status',
    'service_apply_time',
    'service_approval_time',
    'cost_type',
    'fee_type',
    'goods_category',
    'region',
    # ??
    'tosspayments',
    'one_day_delivery',
    'dawn_delivery',
    'fst_in_dt',
    'fst_in_req_time',
    'basic_fee', # is basic fee ?
    'stand_cost', # is stand_cost ?
    # losers
    'sku', 'category',
    'monthly_out_count_med', 'monthly_out_count_rank', 'monthly_out_count_cat',
    'num_log_whole',
    'time_per_session_before', 'time_per_session_after', 'time_per_session_whole',
    'click_count_before', 'click_count_after', 'click_count_whole',
    'bundle_sequence_cnt_whole',
    'direct_cnt_before', 'direct_cnt_after', 'direct_cnt_whole',
    'is_from_cpc',
    'is_unique_search_term_before', 'is_unique_search_term_whole',
    'visit_count_whole',
    'desktop_ratio_before', 'desktop_ratio_after', 'desktop_ratio_whole',
] + pages_to_drop

cols_to_time_scale = [
 "num_log_before",
 "num_log_after",
 "bundle_sequence_cnt_before",
 "bundle_sequence_cnt_after",
 "visit_count_before",
 "visit_count_after",
] + pages_to_use

cols_to_log_scale = [
    'latest',
] + cols_to_time_scale

cols_to_ohe = [

]

"""
    columns needed
"""
cs_needed = [
    "cst_cd",
    "comp_type",
    "promotion",
    "monthly_out_count",
    "service_apply_time",
    "service_approval_time",
    "cost_type",
    "fee_type",
    "car_out",
    "is_recommended",
    "fst_in_req_time",
    "in_yn"
]

ga_needed = [
    "user_id",
    "user_pseudo_id",
    "event_timestamp",
    "event_bundle_sequence_id",
    "event_name",
    "ga_session_id",
    "ga_session_number",
    "page_title",
    "traffic_medium"
]

user_needed = [
    "user_pseudo_id",
    "user_id",
    "cst_cd"
]