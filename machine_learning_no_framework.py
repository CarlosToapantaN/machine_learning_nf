import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# columns has the name of each column.
df = pd.read_csv("Salary.csv")
print(df.head())

pears_corr_coef = np.corrcoef(df.YearsExperience, df.Salary)
print(pears_corr_coef)

X = df['YearsExperience']
Y = df['Salary']

X_train=X.sample(frac=0.8,random_state=200)
X_test=X.drop(X_train.index)
Y_train = Y.sample(frac=0.8,random_state=200)
Y_test = Y.drop(Y_train.index)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

def linear_regression(X_train,Y_train,X_test,Y_test):
        # Calculate X and Y means
        X_mean = X_train.mean()
        Y_mean = Y_train.mean()

        # Calculate slope b
        b = sum((X_train-X_mean)*(Y_train-Y_mean)) / sum(((X_train - X_mean)**2))

        # Calculate intercept a
        a = (Y_mean - (b*X_mean))

        # Prediction on training data + MSE
        Y_pred_train = a + (b * X_train)
        training_error = sum((Y_train-Y_pred_train)**2)/len(Y_train)

        # Prediction on test data + MSE
        Y_pred_test = a + (b * X_test)
        test_error = sum((Y_test-Y_pred_test)**2)/len(Y_test)

        # Calculate R squared
        R2 = sum((Y_pred_train-Y_mean)**2)/sum((Y_train-Y_mean)**2)

        return a, b, training_error, test_error, R2

def salary_prediction(X):
        y = a + (b*X)
        return y

a, b, training_error, test_error, R2 = linear_regression(X_train,Y_train,X_test,Y_test)

print(f"Your LR equation is: Y = {a} + {b}*X")

print(f"Training MSE: {training_error}, Test MSE: {test_error}")

print(f"R squared: {R2}")

prediction1 = salary_prediction(5)
prediction2 = salary_prediction(7)

print("Predictions:")
print("Salary(5): ", prediction1)
print("Salary(7): ", prediction2)