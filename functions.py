def stacking(model1, model2, model3, X_train, y_train, X_test, y_test):
    pred1 = model1.predict(X_train)
    pred2 = model2.predict(X_train)
    pred3 = model3.predict(X_train)

    pred1_test = model1.predict(X_test)
    pred2_test = model2.predict(X_test)
    pred3_test = model3.predict(X_test)

    print("model1: ", accuracy_score(y_test, pred1_test),
          "model2: ", accuracy_score(y_test, pred2_test),
          "model3", accuracy_score(y_test, pred3_test))

    stacked_X_train = np.column_stack((pred1, pred2, pred3))
    stacked_X_test = np.column_stack((pred1_test, pred2_test, pred3_test))

    meta_model = RandomForestClassifier(random_state=42)
    meta_model.fit(stacked_X_train, y_train)

    stacked_pred = meta_model.predict(stacked_X_test)
    accuracy = accuracy_score(y_test, stacked_pred)
    return (accuracy)
