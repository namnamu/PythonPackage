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
    null_count=df.isnull().sum()
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
# 바꾸고 싶은 배열만 들어올 것.
def onehot(array):
    # 판다스일때는 이거 안쓰고 pd.get_dummies(df, columns=['A컬럼', 'B컬럼', 'C컬럼', '컬럼명...'])하면 각 컬럼들마다 알아서 인코딩된다.
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    import pandas as pd

    np_array = np.array(array).reshape(-1, 1) # 1열을 가지는 2차원 배열로 재생성

    encoder = OneHotEncoder()# 원-핫 인코딩
    encoder.fit(np_array) # 인코딩 학습
    labels = encoder.transform(np_array) # 새 인코딩 확인용이지만 따로 나눌 필요는 없는듯.

    # 사용법
    # print('원-핫 인코딩 데이터\n',labels)
    # print(labels.toarray()) #희소행렬(Sparse Matrix) 
    # print(labels.shape) #희소행렬 차원

    df=pd.DataFrame(labels.toarray(),columns=array.unique())
    return df # 원하는 컬럼만 가지고 가서 데이터셋Y로 활용

# 날짜 라벨 인코딩(만들어지는 순서가 영향을 받을 수도 있음.)
# 바꾸고 싶은 배열만 들어올 것.
def label_encoding(array):
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(array)
    return encoder.transform(array)


# 스탠다드 스케일링 : 평균 0, 분산이 1인 정규분포
""" 데이터 왜곡이라는 비판을 받는다."""
def standard_scaling(X): # pandas여도 되고, 2차원 배열도 된다.
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X) # numpy ndarry로 반환 
# 민맥스 스케일링 : 데이터를 0과 1사이의 값으로 축소
""" 데이터가 한쪽으로 쏠리는 현상이 발생한다. """
def minmax_scaling(X):
    from sklearn.preprocessing import MinMaxScaler
    return MinMaxScaler().fit_transform(X) # numpy ndarry로 반환 
# 차원축소 PCA: 주성분 분석. 
# 매 number만큼 분산이 가장 큰 방향(주성분)으로 사영한다.
def dimension_reduction_pca(X,number):
    # Target 값을 제외한 모든 속성 값을 StandardScaler를 이용하여 표준 정규 분포를 가지는 값들로 변환
    # PCA는 여러 속성의 값을 연산해야 하므로 속성의 스케일에 영향을 받는다.
    # 따라서, PCA로 압축하기 전에 각 속성값을 동일한 스케일로 변환하는 것이 필요
    scaled=standard_scaling(X)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=number)# 몇차원으로 줄일 것인지
    done= pca.fit_transform(scaled) # numpy ndarry로 반환 
    print("각 PCA Component별 변동성 비율:",pca.explained_variance_ratio_)
    return done
# 차원축소 LDA: 입력 데이터의 변동성이 가장 큰 축
# 분류에 최적화되어있다.
def dimension_reduction_lda(X,y,number):
    # 스케일링 필요
    scaled=standard_scaling(X)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=number)
    # fit()호출 시 target값 입력
    lda.fit(scaled, y)
    done = lda.transform(scaled)
    print("각 LDA Component별 변동성 비율:",lda.explained_variance_ratio_)
    return done
    