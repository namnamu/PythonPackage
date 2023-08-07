# 결측치를 컬럼 평균으로 전환
def columns_null(df):
    for col in df.columns:
        means=df[col].mean()
        # print(col,means)
        df[col]=df[col].fillna(means)
    return df
# 데이터 프레임 컬럼별 널값 갯수 및 비율 확인
def check_null(df):
    null_cnt_df = pd.DataFrame(df.isnull().sum()).rename(columns = {0:'null_count'}).reset_index()
    null_cnt_df['null_ratio'] = round(null_cnt_df['null_count']/len(df) * 100, 2)
    return null_cnt_df
#