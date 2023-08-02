# 결측치를 컬럼 평균으로 전환
def columns_null(df):
    for col in df.columns:
        means=df[col].mean()
        # print(col,means)
        df[col]=df[col].fillna(means)
    return df
