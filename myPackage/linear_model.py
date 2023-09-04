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
    import time
    start_time = time.time() # 걸리는 시간 측정

    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # 모델생성
    lr_reg = LinearRegression()
    # 학습
    lr_reg.fit(X_train, y_train)
    # 예측
    pred = lr_reg.predict(X_test)
    
    return {"predict":pred,"RMSE":rmse(y_test,pred), "y-intercept":lr_reg.intercept_, "coefficient":lr_reg.coef_,
            'y_test':y_test,'model':lr_reg,'X_test':X_test, "X_train": X_train , "y_train":y_train,
            "mae":mae(y_test ,pred),"rmse":rmse(y_test ,pred), #평가
            "time":str(time.time() - start_time)}


# 다항회귀
def poly_model(X,y,degree):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    import time
    start_time = time.time() # 걸리는 시간 측정

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 다차항 생성
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly=poly_features.fit_transform(X_train.to_numpy())

    # 선형회귀로 펼쳐줌.
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y_train)
    # print(lin_reg.intercept_) # y절편
    # print(lin_reg.coef_)# 계수  
    X_test_poly = poly_features.transform(X_test.to_numpy())      # x1**2 추가
    y_pred = lin_reg.predict(X_test_poly)
    

    return {"predict":y_pred,"RMSE":rmse(y_test,y_pred), "y-intercept":lin_reg.intercept_, "coefficient":lin_reg.coef_,
            'y_test':y_test,'model':lin_reg,'X_test':X_test, "X_train": X_train , "y_train":y_train,
            "mae":mae(y_test ,y_pred),"rmse":rmse(y_test ,y_pred), 
            "time":str(time.time() - start_time)}