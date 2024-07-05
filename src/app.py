import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import pickle
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

try:
    with open('./data/raw/wbm.dat', 'rb') as file:
        data = pickle.load(file)
except FileNotFoundError:
    data = 'https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv'
    data = pd.read_csv(data)
    data.drop_duplicates(inplace=True)
    with open('./data/raw/wbm.dat', 'wb') as file:
        pickle.dump(data, file)
finally:
    print(data.head(10))

x = data.drop(columns='Outcome')
y = data['Outcome']

xtr, xte, ytr, yte = train_test_split(x, y, train_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtr, ytr)
pred = model.predict(xte)
acc_check = accuracy_score(yte, pred)
print(f'Raw data DT accuracy: {acc_check}')

fig, axs = plt.subplots(4, 1, figsize=(5, 7))
tree_plot = plot_tree(model, feature_names=x.columns, filled=True, fontsize=10, ax=axs[0])

corr = data.corr()
refined_x = corr.loc[corr['Outcome'] > .2]
refined_x = data[refined_x.index.to_list()]
refined_x.drop(columns='Outcome', inplace=True)
rxtr, rxte, rytr, ryte = train_test_split(refined_x, y, train_size=0.2, random_state=42)
model2 = DecisionTreeClassifier()
model2.fit(rxtr, rytr)
pred = model2.predict(rxte)
acc_check = accuracy_score(ryte, pred)
print(f"Refined data DT accuracy: {acc_check}")

tree_plot = plot_tree(model2, feature_names=refined_x.columns, filled=True, fontsize=10, ax=axs[1])
pd.plotting.parallel_coordinates(data, 'Outcome', ax=axs[2])
axs[3].scatter(data['Insulin'], data['Outcome'])
plt.show()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20, 25, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8, 16]
}
grid = GridSearchCV(model2, param_grid=param_grid, cv=5, scoring='accuracy')
grid.fit(rxtr, rytr)
print(f'Best parameters (Refined X): {grid.best_params_}\n'
      f'Best score (Refined X): {grid.best_score_}')

grid2 = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='accuracy')
grid2.fit(xtr, ytr)
print(f'Best parameters: {grid2.best_params_}\nBest score: {grid2.best_score_}')

with open('./models/diabeetus.dat', 'wb') as file:
    pickle.dump(grid.best_estimator_, file)
