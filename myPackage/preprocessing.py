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

def scatter3d(df,x,y,z,color):
    # 시각화
    import plotly.express as px
    # 3d그리기
    fig=px.scatter_3d(df, 
            x=x, y=y, z=z,
            color=color, hover_data=[y],
            # size='' 투명도 조절 가능
                )
    # 마커 크기
    fig.update_traces(marker_size = 5)
    return fig