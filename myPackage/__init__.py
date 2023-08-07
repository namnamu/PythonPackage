txt='''
dfhelper:
    편의성을 위해 직접 만든 외부함수모듈입니다.
    to_number_value: (dataframe[,bool])
        데이터 프레임을 컬럼기준으로 숫자화한다. 기존컬럼이름+"_numbering"되어서 반환된다.
        bool은 기본적으로 False이며, 새로운 데이터 프레임을 반환한다.
        True로 설정할 경우 기존 데이터프레임의 컬럼 옆에 붙여서 반환한다.
        데이터프레임은 주 컬럼에 이름이 붙어있어야함.

    normal_check: (dataframe)
        데이터프레임이 가지는 모든 컬럼에 대해 정규성을 확인한다.

    visualization: (dataframe,str,str[,str])
        시각화하고 싶은 데이터 프레임과 컬럼 이름을 적는다. 
        histogram을 이용하고 싶다면 두번째 문자열에 아무 문자열이나 적는다.   
    
    def many_case_ttest(df_col)
        리스트로 들어오는 모든 경우의 수를 리그전 형식으로 구한다.
        df_col: 시리즈

    ttest_auto:(user_df,col,value)
        보페로니 교정방식의 다변량 T검정
        user_df: 입력할 데이터프레임 이름
        col: x축 컬럼 이름 (다변량그룹명)
        value: y축 vlaue이름 (col기준으로 분류된 이름)
processing:
    귀찮음 각 파일 보셈
'''
print(txt)