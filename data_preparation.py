import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

def title_categories(title):
    titles = ["Mr", "Master", "Miss", "Mrs"]
    if title not in titles:
        return "Other"
    else:
        return title


def add_title(data):
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.')
    data['Title'] = data['Title'].map(title_categories)
    data = pd.concat([data, pd.get_dummies(data['Title'])], axis=1)
    return data


def prepare_sex(data):
    sex_mapping = {'female': 0, 'male': 1}
    data['Sex'] = data['Sex'].map(sex_mapping)
    return data


def age_categories(age):
    if age <= 14:
        return "Children"
    elif 14 < age <= 20:
        return "Young"
    elif 20 < age <= 32:
        return "Young adult"
    elif 32 < age <= 41:
        return "Adult"
    elif 41 < age <= 58:
        return "Old adult"
    else:
        return "Elderly"

def prepare_age(data):
    data['Age'].fillna(data.groupby(['Title', 'Pclass', 'Sex'])['Age'].transform('median'), inplace=True)
    data['Age'] = data['Age'].map(age_categories)
    data = pd.concat([data, pd.get_dummies(data['Age'])], axis=1)
    return data


def prepare_embarked(data):
    data['Embarked'].fillna(data['Embarked'].value_counts().index[0], inplace=True)
    data = pd.concat([data, pd.get_dummies(data['Embarked'])], axis=1)
    return data


def fare_categories(fare):
    if fare <= 10:
        return "VeryLowF"
    elif 10 < fare <= 30:
        return "LowF"
    elif 30 < fare <= 300:
        return "HighF"
    else:
        return "VeryHighF"


def prepare_fare(data):
    data['Fare'].fillna(data.groupby('Pclass')['Fare'].transform('median'), inplace=True)
    data['Fare'] = data['Fare'].map(fare_categories)
    data = pd.concat([data, pd.get_dummies(data['Fare'])], axis=1)
    return data

def prepare_family_size(data):
    data['FamilySize'] = data['SibSp'] + data['Parch']
    data['Alone'] = data['FamilySize'].map(lambda x: 1 if x == 0 else 0)
    return data

def delete_useless_data(data):
    useless_data = ['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Cabin', 'Title', 'Age', 'Embarked', 'Fare']
    data = data.drop(useless_data, axis=1)
    return data

def prepare_data(data):
    data = add_title(data)
    data = prepare_sex(data)
    data = prepare_age(data)
    data = prepare_embarked(data)
    data = prepare_fare(data)
    data = prepare_family_size(data)
    data = delete_useless_data(data)
    return data

def bar_chart(data, feature):
    survived = data[data['Survived']==1][feature].value_counts()
    dead = data[data['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,6), title=feature)
    plt.show()

def line_chart(data, feature, lim_beg=None, lim_end=None):
    for x in [0,1]:
        data[feature][data.Survived == x].plot(kind="kde", figsize=(10,6))
    plt.title(feature)
    plt.legend(("Dead", "Survived"))
    plt.xlim(lim_beg, lim_end)
    plt.show()
