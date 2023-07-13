def to_number_value(df,add=False):
    pandas.core.indexes.range.RangeIndex
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
        



