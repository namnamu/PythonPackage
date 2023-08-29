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

