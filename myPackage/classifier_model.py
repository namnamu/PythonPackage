# 모델생성(비호출용)
'''
    결정트리
질문을 계속해서 분류할때 사용
(과적합이 일어나기 쉬움)
    랜덤포레스트
배깅의 알고리즘: 여러 샘플링에 대해 결정트리 수행, 예측의 최빈값
(다양성 추가로 분산 감소, 독립성 증가, 일반화 성능 좋음)
    앙상블
편향과 분산을 줄인 모델=분류기 여러개 사용
    1) 보팅
    여러 분류기(다른 알고리즘)에 넣어 투표로 최종 예측 결과를 선정
    - 하드 보팅: 다수의 모델이 낸 결과로 선정
    - 소프트 보팅: 각 클래스들이 예측한 확률을 총 정리
    2) 배깅-로지스틱 회귀를 분류로 사용
    데이터 샘플링 무작위로 예측값들의 최빈값을 최종 예측값으로 선정
    - 로지스틱: 배깅의 모델 1개,샘플링 여러개의 모델에 로지스틱 회귀를 사용한것
    - 스태킹: 여러 모델에 넣는다. 각 모델 예측값에 대해 새로운 학습 데이터로 정리해 학습시킨다.
    3) 부스팅
    앞의 분류기가 가중치를 부여하여 다음 분류기에 전달해 개선해 나간다.
    - 에이다부스트: 가중치부여
    - 그래디언트부스팅: 오차보정. 예측값과 실제값을 비교해 새로 학습
    knn
거리에 따라 가까운대로 분류
'''
def create_classifier_model(con):
    if con=='decision_tree':
        from sklearn.tree import DecisionTreeClassifier
        dt_clf = DecisionTreeClassifier(random_state=156)
        # dt_clf.feature_importances_로 어느 요소가 영향을 제일 많이 끼쳤는지 확인가능
        return dt_clf
    elif con=='forest':
        from sklearn.ensemble import RandomForestClassifier
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=8)
        # rf_clf.feature_importances_로 어느 요소가 영향을 제일 많이 끼쳤는지 확인가능

        return rf_clf
    elif con=='bagging_logistic':
        from sklearn.ensemble import BaggingClassifier
        from sklearn.linear_model import LogisticRegression
        lr_clf = LogisticRegression(solver='liblinear') # 로지스틱 회귀가 베이스
        bagging_clf = BaggingClassifier(estimator=lr_clf) # 배깅 모델
        return bagging_clf
    elif con=="gradient_boosting":# GBM
        from sklearn.ensemble import GradientBoostingClassifier
        gb_clf = GradientBoostingClassifier(random_state=0)
        return gb_clf
    elif con=="knn":
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=1)
        return knn
    elif con=="voting":
        from sklearn.ensemble import VotingClassifier # 보팅 기법 모듈

        # 개별 모델은 KNN와 DecisionTree 임.
        knn_clf = create_classifier_model('knn')
        dt_clf = create_classifier_model('decision_tree')

        # 개별 모델을 소프트 보팅 기반의 앙상블 모델로 구현한 분류기
        vo_clf = VotingClassifier(estimators=[('KNN',knn_clf),('DT',dt_clf)] , voting='soft' )
        #'KNN'은 별명임.
        
        return vo_clf
    elif con=="linear":
        # 로지스틱 회귀와 소프트맥스 회귀
        # 클래스 분류가 2개가 넘으면 자동으로 소프트맥스 회귀가 적용됨.
        from sklearn.linear_model import LogisticRegression

        log_reg = LogisticRegression(C=30)# C=30은 약한 규제의 의미
        return log_reg
    else:
        print("종류에러\ndecision_tree,forest,bagging_logistic,gradient_boosting,knn,voting,linear중 입력\n기본:decisionTree수행")
        from sklearn.tree import DecisionTreeClassifier
        dt_clf = DecisionTreeClassifier(random_state=156)
        return dt_clf

# 머신러닝 모델(호출용)
def classifier(X,Y,con='decisionTree'):
    import time
    start_time = time.time()
    # 데이터 셋 분리
    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X, Y, test_size=0.2)# random_state=11
    
    # 생성
    model=create_classifier_model(con)
    # 학습
    model.fit(X_train , y_train)
    # 평가
    import sklearn.metrics as mt
    y_pred = model.predict(X_test) # 예측값
    y_proba = model.predict_proba(X_test) # 각 클래스에 대한 확률값
    accuracy = mt.accuracy_score(y_test, y_pred) #정확도
    precision = mt.precision_score(y_test, y_pred)# 정밀도
    recall = mt.recall_score(y_test, y_pred)# 재현율
    auc = mt.roc_auc_score(y_test, y_pred) #auc너비. 보고할때는 항상 auc로
    matrix = mt.confusion_matrix(y_test, y_pred) # 오차행렬

    decision_boundary='only linear(maybe...)'
    if con=="linear":
        import numpy as np
        #결정 경계 확인
        X_new = np.linspace(0, 3, 1000).reshape(-1, X.shape[1])  # X의 최소와 최대지만, 꼭 그렇게 한정한 것은 아니고 크게 둘른 모양, 중앙값을 확실히 알기 위해서 
        y_proba = model.predict_proba(X_new) # 정답일 확률/오답일 확률
        decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0] # 첫 클래스의 분류경계. 이진분류시, 0과 1을 나누는 중간 값.
        

    return {"predict":y_pred,"accuracy":accuracy,"precision":precision,"recall":recall,"auc":auc,"matrix":matrix,
            'y_test':y_test,'model':model,'X_test':X_test, "X_train": X_train , "y_train":y_train,
            "time":str(time.time() - start_time), 'y_proba':y_proba,'decision_boundary':decision_boundary}


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