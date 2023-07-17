def to_number_value(df,add=False):
    import pandas as pd
    new_df=pd.DataFrame()
    for c in df.columns:
        if df[c].dtype not in ['int64','float']:
            kind=list(df[c].unique())
            number=len(kind)
            new_name=c+"_numbering"
            new_df[new_name]=df[c].apply(lambda x: kind.index(x))
        else:
            #숫자형 데이터는 그대로
            new_df[c]=df[c]
    if add:
        new_df=pd.concat([new_df,df],axis=1)
    return new_df

def normal_check(df):
    for c in df.columns:
        try:
            normal_distribution=stats.shapiro(df[c])
            print(c,"컬럼의 정규분포는 " ,end='')
            if normal_distribution.pvalue>0.05:
                print("정규성 O.")
            else:
                print("정규성 X",normal_distribution.pvalue)
        except:
            print("문자열 등으로 인해 정규화 불가")
        
def visualization(df,x_name,y_name,shape='none'):
    import seaborn as sns
    import plotly.express as px
    import matplotlib.pyplot as plt
    plt.rc('font',family='Malgun Gothic')
    import matplotlib as mpl
    mpl.rcParams['axes.unicode_minus']=False
    x=df[x_name]
    y=df[y_name]
    if shape=="bar":
        sns.barplot(df,x=x,y=y)
    elif shape=="line":
        sns.lineplot(df,x=x,y=y)
    elif shape=="pie":
        fig=px.pie(df,values=y_name,names=x_name)
        fig.show()
    elif shape=="histo":
        sns.histplot(data=df, x=x_name, kde=True)
        print("히스토그램은 ",x_name,"의 갯수를 표현하는 그래프임.")
    elif shape=="stem":
        plt.stem(x,y)
    elif shape=="scatter":
        sns.scatterplot(x=x,y=y)
    elif shape=="box":
        sns.boxenplot(df,x=x_name,y=y_name)
    else:
        print("bar, line, pie, histo, stem, scatter, box 중 입력")