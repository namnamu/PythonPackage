# 결측치를 컬럼 평균으로 전환
def columns_null(df):
    for col in df.columns:
        means=df[col].mean()
        # print(col,means)
        df[col]=df[col].fillna(means)
    return df
# 데이터 프레임 컬럼별 널값 갯수 및 비율 확인
def check_null(df):
    import pandas as pd
    null_cnt_df = pd.DataFrame(df.isnull().sum()).rename(columns = {0:'null_count'}).reset_index()
    null_cnt_df['null_ratio'] = round(null_cnt_df['null_count']/len(df) * 100, 2)
    return null_cnt_df
def check_null_simple(df):
    import pandas as pd
    null_count=df.isnull.sum()
    return null_count/len(df)*100

# 상관계수 시각화
def heatmap(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    float_li, cate_li=flo_cate(df)
    float_df=df[float_li]
    plt.figure(figsize = (7, 5))
    sns.heatmap(float_df.corr(), annot = True, cmap = 'coolwarm', fmt = '.2f')
    plt.show()

# 데이터프레임을 실수형식과 아닌 것으로 나눔
def flo_cate(df):
    cate_li=[]
    flo_li=[]
    for col in df.columns:
        if df[col].dtype in ['float64','int64']:
            flo_li.append(col)
        else:
            cate_li.append(col)
    return flo_li, cate_li
# 컬럼형 라벨링(우위관계가 없을 때 사용, 50개가 넘으면 쓰지 말 것.)
def onehot(array):
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np

    np_array = np.array(array).reshape(-1, 1) # 1열을 가지는 2차원 배열로 재생성

    encoder = OneHotEncoder()# 원-핫 인코딩
    encoder.fit(np_array) # 인코딩 학습
    labels = encoder.transform(np_array) # 새 인코딩 확인용이지만 따로 나눌 필요는 없는듯.
    
    # print('원-핫 인코딩 데이터\n',labels)
    print(labels.toarray()) #희소행렬(Sparse Matrix) 
    print(labels.shape) #희소행렬 차원
    return labels