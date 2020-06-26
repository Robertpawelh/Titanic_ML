from data_preparation import *
from decision_tree import *
from submissions import prepare_solution
from logistic_regression import *

def data_analysis(data):
    bar_chart(data, 'Pclass')
    bar_chart(data, 'Sex')
    bar_chart(data, 'SibSp')
    bar_chart(data, 'Parch')
    bar_chart(data, 'Cabin')
    bar_chart(data, 'Embarked')
    data.hist(bins=10,figsize=(10,6),grid=False)
    print(data.columns[data.isnull().any()])
    print(data['Pclass'].value_counts())
    line_chart(data, 'Age')
    line_chart(data, 'Age', 0, 25)
    line_chart(data, 'Age', 25, 50)
    line_chart(data, 'Age', 50, 80)
    line_chart(data, 'Fare', 0, 200)
    line_chart(data, 'Fare', 0, 25)
    line_chart(data, 'Fare', 25, 50)
    line_chart(data, 'Fare', 50, 80)
    print(data.info())
    print(data.info())
    print(data.describe())


def classification_correctness(pred, y):
    N = y.shape[0]
    count = 0
    for n in range(N):
        if pred[n] == y[n]:
            count += 1
    return count/N


def run():
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    #data_analysis(train_data)
    train_data = prepare_data(train_data)
    x_train = train_data.drop(['Survived'], axis=1)
    y_train = train_data['Survived']

    ids = test_data['PassengerId']
    test_data = prepare_data(test_data)

    LRModel = LogisticRegression()
    LRModel.fit(x_train, y_train)
    prepare_solution(ids, LRModel.predict(test_data), 'regression')
    print(classification_correctness(LRModel.predict(x_train), y_train))

    TreeModel = DecisionTree()
    TreeModel.fit(train_data, 'Survived')
    prepare_solution(ids, TreeModel.predict(test_data), 'tree')
    print(classification_correctness(TreeModel.predict(train_data), y_train))


if __name__ == "__main__":
    run()