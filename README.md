# 개요

인바운드 SME 고객사의 입고(매출 발생)여부 예측을 위한 분류 모형

# Feature Description

| 피쳐명                                                                                                                    | 적용 시점  | 설명 |
|------------------------------------------------------------------------------------------------------------------------|--------| --- |
| is_corp                                                                                                                | 둘 다    | 법인 여부 |
| is_nfa                                                                                                                 | 둘 다    | nfa 여부 |
| monthly_out_count_under_100                                                                                            | 둘 다    | 월 출고량 100 이하 여부 |
| is_stand_cost                                                                                                          | 승인 후   | 표준 요금 여부 |
| is_basic_fee                                                                                                           | 승인 후   | 기본 요금 여부 |
| car_out                                                                                                                | 승인 후   | 차량 출고 이용 여부 |
| is_recommended                                                                                                         | 둘 다    | 추천인 여부 |
| num_log                                                                                                                | 둘 다    | 로그 수 |
| bundle_sequence_cnt                                                                                                    | 둘 다    | 이벤트 번들 수 |
| event_hour                                                                                                             | 둘 다    | 주 활동 시간 |
| event_hour_gini                                                                                                        | 둘 다    | 활동 시각 다양성 지수 |
| visit_count                                                                                                            | 둘 다    | 방문 횟수 |
| is_from_sa                                                                                                             | 둘 다    | 유입경로 sa 여부 |
| x_visit_before <br/>x = {'1:1 문의', '배송', '서비스 신청', '쇼핑몰 연동', <br/>'신규 상품 등록', ‘요금', ‘입고', '출고’ , <br/>'이용 가이드(기타)', '이용 전 가이드(기타)'} | 둘 다    | 각 페이지 별 방문 횟수 |)



# 모델 학습 방법

## 1. 데이터 준비

- 전처리 된 학습 직전의 데이터: data/preprocessed
    - 전처리 된 데이터 입력 시 최소 필요 칼럼들: Feature Description 참고
- 전처리 되지 않은 raw 데이터: data/raw에 고객 데이터, GA 데이터, user_id 데이터 준비
    - Raw 데이터 입력 시 최소 필요 칼럼들
    
    | 입력 데이터 | 필요 칼럼 |
    | --- | --- |
    | 고객 데이터 | cst_cd, comp_type, promotion, monthly_out_count, service_apply_time, service_approval_time, cost_type, fee_type, car_out, is_recommended, fst_in_req_time, in_yn |
    | user_id 데이터 | user_pseudo_id, user_id, cst_cd |
    | GA 데이터 | user_id, user_pseudo_id, event_name, event_timestamp, event_bundle_sequence_id, ga_session_id, ga_session_number, page_title, traffic_medium |

## 2. 데이터 학습

- 전처리 된 데이터 학습: python [test.py](http://test.py) -p (-i 7) (-s) file1.csv
- Raw 데이터 학습: python [test.py](http://test.py) (-i 7) (-s) (-b) customer-like.csv ga-like.csv user_id-like.csv
    - -p (—preprocessed)
        - -p를 포함할 경우 전처리 과정을 생략, 미포함시 전처리 거침
    - -i n (—iteration n)
        - 랜덤시드 n번 반복
        - 미지정 시 기본값 100
    - -s (—simple)
        - 하이퍼파라미터 튜닝 생략
        - 빠른 결과 확인용
    - -b (—applied)
        - -b를 포함할 경우 전처리 과정에서 applied(신청 전) 모델용 학습 데이터 생성

## 3. 결과 확인

result 디렉터리에 입력 파일 이름과 같은 이름의 폴더가 생성됨.

| 파일명                | 설명 |
|--------------------| --- |
| result.txt         | validation 성능 지표 |
| (파일명)_extended.csv | 모델 입력 파일에 확률값 계산한 결과 포함 |
| features.csv       | 변수 중요도 혹은 가중치 |

trained_model 디렉터리에 학습된 모델이 저장됨

| 파일명                                   | 설명 |
|---------------------------------------| --- |
| model_applied.pkl, model_approved.pkl | applied, approved 모델을 확률값 예측을 위해 저장 |

## 4. 입고 확률 예측

### 데이터 준비

- 전처리 된 데이터 예측: data/preprocessed
- 전처리 되지 않은 raw 데이터 예측: data/raw에 예측하고자 하는 cst_cd들에 대한 고객 데이터, user_id 데이터, 그리고 누적 GA 데이터 준비 (GA 데이터는 가공 없이 input 가능)

### 확률 예측

- 전처리 된 데이터 예측 : python [predict.py](http://predict.py) (-p) (-b) file1.csv
- Raw 데이터 예측: python [predict.py](http://predict.py) (-b) customer-like.csv ga-like.csv user_id-like.csv
    - -p (—preprocessed)
        - 포함할 경우 전처리 과정 생략
    - -b (—applied)
        - 포함할 경우 applied 모델로 예측, 미포함할 경우 approved 모델로 예측

### 결과 확인

result 디렉터리에 입력 파일 이름과 같은 이름의 폴더가 생성됨.

| 파일명            | 설명 |
|----------------| --- |
| pred_(파일명).csv | 모델 입력 파일에 예측 확률값 계산한 결과 포함 |