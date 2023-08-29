'''
to_number_value: (dataframe[,bool])
        데이터 프레임을 컬럼기준으로 숫자화한다. 기존컬럼이름+"_numbering"되어서 반환된다.
        bool은 기본적으로 False이며, 새로운 데이터 프레임을 반환한다.
        True로 설정할 경우 기존 데이터프레임의 컬럼 옆에 붙여서 반환한다.
        데이터프레임은 주 컬럼에 이름이 붙어있어야함.
'''
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
'''
normal_check: (dataframe)
    데이터프레임이 가지는 모든 컬럼에 대해 정규성을 확인한다.
'''
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
    
'''
def many_case_ttest(df_col)
    시리즈로 들어오는 모든 경우의 수를 리그전 형식으로 구한다
'''
def many_case_ttest(df_col):
    li=list(df_col.unique())
    n=[]
    while len(li)>0:
        last=li.pop()
        for l in li:
            n.append(((l),(last)))
    return n
'''
ttest_auto:(user_df,col,value)
    보페로니 교정방식의 다변량 T검정
    user_df: 입력할 데이터프레임 이름
    col: x축 컬럼 이름 (다변량그룹명)
    value: y축 vlaue이름 (col기준으로 분류된 이름)
'''
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
'''
visualization: (dataframe,str,str[,str])
    시각화하고 싶은 데이터 프레임과 컬럼 이름을 적는다. 
    histo의 경우 문자열은 동작하지 않는다.   
'''
def visualization(df,x_name,y_name,shape='none'):
    import seaborn as sns
    import plotly.express as px
    import matplotlib.pyplot as plt
    plt.rc('font',family='Malgun Gothic')# 한국어 표시
    import matplotlib as mpl
    mpl.rcParams['axes.unicode_minus']=False # -표시
    
    x=df[x_name]
    y=df[y_name]
    
    plt.figure(figsize=(10,6))
    plt.xticks(rotation=90)
    
    if shape=="bar":
        bar=plt.bar(x,y)
        plt.ylim([min(y),max(y)*1.081])
        # 숫자 넣는 부분
        for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % height, ha='center', va='bottom', size = 12)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
    elif shape=="line":
        sns.lineplot(df,x=x,y=y)
    elif shape=="pie": # 컬럼에 오류가 있으면 에러를 출력하지 않고 빈 화면만 출력한다.
        fig=px.pie(df,values=y_name,names=x_name)#value=숫자형 names=컬럼형
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
    elif shape=="countplot": 
        #bar그룹형 (histo처럼 수를 센다.)
        sns.countplot(data = df, x = x_name, hue = y_name)
    elif shape=='reg':
        sns.regplot(data=df, x=x_name,y=y_name,line_kws={"color": "black"})
    else:
        print("bar, line, pie, histo, stem, scatter, box 중 입력")
'''
3d 시각화
'''
def scatter3d(df,x,y,z,color):
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
