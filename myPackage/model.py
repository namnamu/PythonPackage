# 모델생성(비호출용)
def create_model(con):
    '''
    결정트리
    질문을 계속해서 분류할때 사용
    (과적합이 일어나기 쉬움)
    '''
    if con=='decisionTree':
        from sklearn.tree import DecisionTreeClassifier
        dt_clf = DecisionTreeClassifier(random_state=156)
        return dt_clf
    else:
        from sklearn.ensemble import RandomForestClassifier
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=8)
        return rf_clf

# 머신러닝 모델(호출용)
def model(X,Y,con='decisionTree'):
    # 데이터 셋 분리
    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X, Y, test_size=0.2)# random_state=11
    
    # 생성
    model=create_model(con)
    # 학습
    model.fit(X_train , y_train)
    # 평가
    import sklearn.metrics as mt

    y_pred = model.predict(X_test) # 예측값
    accuracy = mt.accuracy_score(y_test, y_pred) #정확도
    precision = mt.precision_score(y_test, y_pred)# 정밀도
    recall = mt.recall_score(y_test, y_pred)# 재현율
    auc = mt.roc_auc_score(y_test, y_pred) #auc너비. 보고할때는 항상 auc로
    matrix = mt.confusion_matrix(y_test, y_pred) # 오차행렬

    return {"predict":y_pred,"accuracy":accuracy,"precision":precision,"recall":recall,"auc":auc,"matrix":matrix,
            'y_test':y_test,'model':model,'X_test':X_test, "X_train": X_train , "y_train":y_train}


"""
RoC커브 시각화
분류모델에서 matrix만 있다면 그릴 수 있다.
"""
def RoC(model,matrix,X_test,y_test):
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    plt.rc('font',family='Malgun Gothic') # 한국어 표시
    import matplotlib as mpl
    mpl.rcParams['axes.unicode_minus']=False # -표시
    
    # x축과 y축을 위한 공식
    [[tn,fp],[fn,tp]]=matrix
    fallout=fp/(fp+tn)
    recall=tp/(tp+fn)

    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    print(f'{fpr=},\n{tpr=},\n{thresholds=}')

    plt.plot(fpr, tpr, 'o-', label="Logistic Regression")
    plt.plot([0, 1], [0, 1], 'k--', label="random guess")
    plt.plot([fallout], [recall], 'ro', ms=10)
    plt.xlabel('위양성률(Fall-Out)')
    plt.ylabel('재현률(Recall)')
    plt.title('Receiver operating characteristic example')
    plt.show()

"""
분류모델에서 가장 영향을 많이 미친 특성
"""
def best_feature(model,X_train):
    ftr_importances_values = model.feature_importances_
    ftr_importances = pd.Series(ftr_importances_values,index=X_train.columns)
    return ftr_importances.sort_values(ascending=False)[:20]