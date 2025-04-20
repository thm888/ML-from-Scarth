from decision_tree import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np 


def main():
    data = fetch_california_housing()
    X,y = data.data, data.target
    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size = 0.3
    )
    model = DecisionTree(criterion='mse')
    model.fit(X_train,y_train)
    loss = np.mean((y_test- model.predict(X_test))**2)
    mse = np.sqrt(loss)
    print(mse)

if __name__ == "__main__":
    main()
    