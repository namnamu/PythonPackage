from tokenize import cookie_re


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
    import matplotlib.pyplot as plt
    plt.rc('font',family='Malgun Gothic')
    import matplotlib as mpl
    mpl.rcParams['axes.unicode_minus']=False
    x=df[x_name]
    y=df[y_name]
    if shape=="bar":
        sns.barplot(df,x=x,y=y,color='#FC7F77')
    elif shape=="line":
        sns.lineplot(df,x=x,y=y)
    elif shape=="pie": # 컬럼에 오류가 있으면 에러를 출력하지 않고 빈 화면만 출력한다.
        import plotly.express as px
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


def many_case_ttest(df_col):
    li=list(df_col.unique())
    n=[]
    while len(li)>0:
        last=li.pop()
        for l in li:
            n.append(((l),(last)))
    return n

def ttest_auto(user_df,col,value):
    #pip install statannot
    from statannot import add_stat_annotation
    import seaborn as sns
    f1=sns.boxplot(data=user_df, x=col,y=value)
    f2=sns.stripplot(data=user_df, x=col,y=value,color='k',alpha=.5) #겹쳐서 그린듯
    add_stat_annotation(f1, 
                        data=user_df, 
                        x = col, 
                        y = value,
                        box_pairs=many_case_ttest(user_df[col]), #어떤것들이 박스인지 박스 이름을 적어준다.
                        test='t-test_ind', #어떤 test를 할 것인가. 이름일부러 잘못적으면 이것들이 있다고 알려준다. 안외워도 된다.
                        text_format='star', # p-value가 몇인지 마크를 별로 나타내는 것. 에러나면 뭐 적으라고 알려줌.
                        loc='outside', # 두개 p-value나타내는걸 어디다가 적을 것인가
                        verbose=2 # 0이면 ttest에 대해 아무 말도 안한다. 과 1이면 일부만 말한다. 2이면 전부 말한다.
                        )