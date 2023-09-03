# 사이킷런의 mean_square_error() 를 이용하여 RMSE 계산
def rmse(y,pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y,pred))

# MAE 는 scikit learn의 mean_absolute_error() 로 계산
def mae(y,pred):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y,pred)

# 선형회귀
def linear_regression_model(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # 모델생성
    lr_reg = LinearRegression()
    # 학습
    lr_reg.fit(X_train, y_train)
    # 예측
    pred = lr_reg.predict(X_test)
    # 평가
    print("mae:",mae(y_test ,pred),"\nrmse:",rmse(y_test ,pred))
