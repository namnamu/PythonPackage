'''
 결정트리
질문을 계속해서 분류할때 사용
(과적합이 일어나기 쉬움)
'''
def Decison_Tree(X,Y):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # DecisionTree Classifier 생성
    dt_clf = DecisionTreeClassifier(random_state=156)

    # 데이터 셋 분리
    X_train , X_test , y_train , y_test = train_test_split(X, Y, test_size=0.2,  random_state=11)

    # DecisionTreeClassifer 학습.
    dt_clf.fit(X_train , y_train)

    # 성능지표
    import sklearn.metrics as mt

    y_pred = dt_clf.predict(X_test) # 예측값
    accuracy = mt.accuracy_score(y_test, y_pred) #정확도
    precision = mt.precision_score(y_test, y_pred)# 정밀도
    recall = mt.recall_score(y_test, y_pred)# 재현율
    auc = mt.roc_auc_score(y_test, y_pred) #auc너비. 보고할때는 항상 auc로
    matrix = mt.confusion_matrix(y_test, y_pred) # 오차행렬

    # 사용법
    # print("Decision Tree 정확도: {:.4f}".format(accuracy))
    # print("Decision Tree 정밀도: {:.4f}".format(precision))
    # print("Decision Tree 재현율: {:.4f}".format(recall))
    # print("Decision Tree AUC: {:.4f}".format(auc))
    # print('Decision Tree Confusion Matrix:','\n', matrix)
    return {"predict":y_pred,"accuracy":accuracy,"precision":precision,"recall":recall,"auc":auc,"matrix":matrix}

"""
RoC커브 시각화
분류모델에서 matrix만 있다면 그릴 수 있다.
"""
def RoC(matrix):
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    plt.rc('font',family='Malgun Gothic') # 한국어 표시
    import matplotlib as mpl
    mpl.rcParams['axes.unicode_minus']=False # -표시
    
    # x축과 y축을 위한 공식
    [[tn,fp],[fn,tp]]=matrix
    fallout=fp/(fp+tn)
    recall=tp/(tp+fn)

    fpr, tpr, thresholds = roc_curve(y_test, dt_clf.predict_proba(X_test)[:, 1])
    print(f'{fpr=},\n{tpr=},\n{thresholds=}')

    plt.plot(fpr, tpr, 'o-', label="Logistic Regression")
    plt.plot([0, 1], [0, 1], 'k--', label="random guess")
    plt.plot([fallout], [recall], 'ro', ms=10)
    plt.xlabel('위양성률(Fall-Out)')
    plt.ylabel('재현률(Recall)')
    plt.title('Receiver operating characteristic example')
    plt.show()