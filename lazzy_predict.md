# Predições com lazypredict

**Alvaro J. Lopez** 


```lazypredict``` é uma libreria que nos permite testar diversos algoritmos de machine learning
usando apenas poucas linhas de código. Assim, graças a essa libreria podemos explorar diversos 
modelos em pouco tempo.

Atualmente ```lazypredict``` conta com algoritmos de machine learning para resolver problemas
de classificação (usando o módulo ```LazyClassifier```) e regressão (usando o módulo ```LazyRegressor```).

Neste tutorial vamos a mostrar como usar a libreria ```lazypredict``` para resolver 
problemas de classificação e regressão.

Como primeiro passo devemos instalar a libreria ```lazypredict```

* Instalamos as librerias

```python
pip install lazypredict
```


## Problema de Classificação:  ```LazyClassifier```

* Importamos as librerias

```python
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets
```

* Carregamos os dados e definimos as variáveis dependentes (X) e independentes (y).

```python
# Carregar os dados
data = datasets.load_breast_cancer()
X, y = data.data, data.target
```

* Dividimos os dados em dados de treino e dados de teste

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
```

* Realizar o fit dos modelos usando ```LazyClassifier```

```python
# Fit dos modelos
clf = LazyClassifier(predictions=True)
modelos, predicoes = clf.fit(X_train, X_test, y_train, y_test)
```

A variável ```modelos``` é um dataframe que contém as informações sobre a performance 
de cada modelo testado usando o ```LazyClassifier```,

* Imprimir os resultados

Para imprimir os resultados usaremos a seguinte linha de código. 

```python
from tabulate import tabulate

print(tabulate(modelos, headers='keys', tablefmt='psql'))
```


```
Output:

+-------------------------------+------------+---------------------+-----------+------------+--------------+
| Model                         |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time Taken |
|-------------------------------+------------+---------------------+-----------+------------+--------------|
| BernoulliNB                   |   0.982456 |            0.98133  |  0.98133  |   0.982456 |    0.0207658 |
| PassiveAggressiveClassifier   |   0.982456 |            0.98133  |  0.98133  |   0.982456 |    0.0287631 |
| SVC                           |   0.982456 |            0.976744 |  0.976744 |   0.982369 |    0.0244675 |
| Perceptron                    |   0.973684 |            0.974288 |  0.974288 |   0.973742 |    0.0205691 |
| AdaBoostClassifier            |   0.973684 |            0.969702 |  0.969702 |   0.973621 |    0.17231   |
| LogisticRegression            |   0.973684 |            0.969702 |  0.969702 |   0.973621 |    0.0676136 |
| ExtraTreeClassifier           |   0.964912 |            0.967245 |  0.967245 |   0.96506  |    0.0172093 |
| SGDClassifier                 |   0.964912 |            0.967245 |  0.967245 |   0.96506  |    0.0165768 |
| CalibratedClassifierCV        |   0.973684 |            0.965116 |  0.965116 |   0.973481 |    0.0757906 |
| XGBClassifier                 |   0.964912 |            0.958074 |  0.958074 |   0.964738 |    0.177755  |
| RandomForestClassifier        |   0.964912 |            0.958074 |  0.958074 |   0.964738 |    0.256239  |
| LGBMClassifier                |   0.964912 |            0.958074 |  0.958074 |   0.964738 |    0.202984  |
| GaussianNB                    |   0.964912 |            0.958074 |  0.958074 |   0.964738 |    0.0206826 |
| ExtraTreesClassifier          |   0.964912 |            0.958074 |  0.958074 |   0.964738 |    0.286708  |
| QuadraticDiscriminantAnalysis |   0.95614  |            0.955617 |  0.955617 |   0.956237 |    0.0282714 |
| LinearSVC                     |   0.95614  |            0.955617 |  0.955617 |   0.956237 |    0.0428808 |
| BaggingClassifier             |   0.95614  |            0.951032 |  0.951032 |   0.956036 |    0.0930951 |
| LinearDiscriminantAnalysis    |   0.95614  |            0.946446 |  0.946446 |   0.955801 |    0.0354171 |
| NearestCentroid               |   0.95614  |            0.946446 |  0.946446 |   0.955801 |    0.0150883 |
| NuSVC                         |   0.95614  |            0.946446 |  0.946446 |   0.955801 |    0.0371542 |
| RidgeClassifier               |   0.95614  |            0.946446 |  0.946446 |   0.955801 |    0.0181067 |
| RidgeClassifierCV             |   0.95614  |            0.946446 |  0.946446 |   0.955801 |    0.0213528 |
| KNeighborsClassifier          |   0.947368 |            0.94399  |  0.94399  |   0.947368 |    0.0362985 |
| DecisionTreeClassifier        |   0.947368 |            0.94399  |  0.94399  |   0.947368 |    0.0387599 |
| LabelSpreading                |   0.938596 |            0.932362 |  0.932362 |   0.93845  |    0.0418503 |
| LabelPropagation              |   0.938596 |            0.932362 |  0.932362 |   0.93845  |    0.0694857 |
| DummyClassifier               |   0.622807 |            0.5      |  0.5      |   0.478046 |    0.0195985 |
+-------------------------------+------------+---------------------+-----------+------------+--------------+
```


`
## Problema de Regressão:  ```LazyRegressor```

* Importamos as librerias

```python
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets
```

* Carregamos os dados e definimos as variáveis dependentes (X) e independentes (y).

```python
# Carregar os dados
data = datasets.load_boston()
X, y = data.data, data.target
```

* Dividimos os dados em dados de treino e dados de teste

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
```

* Realizar o fit dos modelos usando ```LazyClassifier```

```python
# Fit dos modelos
clf = LazyRegressor(predictions=True)
modelos, predicoes = clf.fit(X_train, X_test, y_train, y_test)
```

A variável ```modelos``` é um dataframe que contém as informações sobre a performance 
de cada modelo testado usando o ```LazyClassifier```,

* Imprimir os resultados

Para imprimir os resultados usaremos a seguinte linha de código. 

```python
from tabulate import tabulate

print(tabulate(modelos, headers='keys', tablefmt='psql'))
```


```
Output:

+-------------------------------+----------------------+--------------+----------+--------------+
| Model                         |   Adjusted R-Squared |    R-Squared |     RMSE |   Time Taken |
|-------------------------------+----------------------+--------------+----------+--------------|
| GradientBoostingRegressor     |             0.902837 |  0.915344    |  2.49162 |   0.17058    |
| XGBRegressor                  |             0.88627  |  0.900909    |  2.69569 |   0.0512927  |
| RandomForestRegressor         |             0.87616  |  0.8921      |  2.81296 |   0.301073   |
| LGBMRegressor                 |             0.874857 |  0.890964    |  2.82772 |   0.0585206  |
| ExtraTreesRegressor           |             0.847273 |  0.866931    |  3.12385 |   0.192918   |
| BaggingRegressor              |             0.844298 |  0.864339    |  3.15413 |   0.0432482  |
| HistGradientBoostingRegressor |             0.841431 |  0.861841    |  3.18303 |   0.236061   |
| DecisionTreeRegressor         |             0.836981 |  0.857963    |  3.22739 |   0.014631   |
| AdaBoostRegressor             |             0.81424  |  0.83815     |  3.44516 |   0.103376   |
| PoissonRegressor              |             0.729439 |  0.764264    |  4.15782 |   0.017607   |
| ExtraTreeRegressor            |             0.711302 |  0.748461    |  4.29491 |   0.01176    |
| KNeighborsRegressor           |             0.677508 |  0.719017    |  4.53933 |   0.00904202 |
| RANSACRegressor               |             0.671047 |  0.713387    |  4.58458 |   0.129608   |
| Lars                          |             0.619826 |  0.668759    |  4.9286  |   0.0299439  |
| LinearRegression              |             0.619826 |  0.668759    |  4.9286  |   0.00763869 |
| LassoLarsCV                   |             0.619826 |  0.668759    |  4.9286  |   0.021666   |
| TransformedTargetRegressor    |             0.619826 |  0.668759    |  4.9286  |   0.00824976 |
| Ridge                         |             0.619485 |  0.668462    |  4.93081 |   0.00920606 |
| LassoCV                       |             0.6194   |  0.668388    |  4.93136 |   0.0618937  |
| ElasticNetCV                  |             0.618357 |  0.667479    |  4.93812 |   0.0548029  |
| BayesianRidge                 |             0.617852 |  0.667039    |  4.94139 |   0.0133696  |
| LassoLarsIC                   |             0.616922 |  0.666229    |  4.94739 |   0.0132451  |
| RidgeCV                       |             0.616622 |  0.665968    |  4.94933 |   0.00777245 |
| LarsCV                        |             0.615146 |  0.664682    |  4.95885 |   0.0649781  |
| SGDRegressor                  |             0.610687 |  0.660796    |  4.98749 |   0.00913835 |
| SVR                           |             0.598269 |  0.649977    |  5.06641 |   0.0218942  |
| GammaRegressor                |             0.590472 |  0.643184    |  5.11534 |   0.00929403 |
| NuSVR                         |             0.581029 |  0.634956    |  5.17398 |   0.031327   |
| MLPRegressor                  |             0.574835 |  0.62956     |  5.21208 |   0.578158   |
| Lasso                         |             0.568389 |  0.623943    |  5.25145 |   0.0217998  |
| HuberRegressor                |             0.557795 |  0.614712    |  5.31551 |   0.0175295  |
| ElasticNet                    |             0.556166 |  0.613294    |  5.32528 |   0.00791574 |
| TweedieRegressor              |             0.556031 |  0.613176    |  5.32609 |   0.00910378 |
| OrthogonalMatchingPursuitCV   |             0.547919 |  0.606108    |  5.37453 |   0.0190825  |
| LinearSVR                     |             0.527978 |  0.588733    |  5.49179 |   0.00854158 |
| OrthogonalMatchingPursuit     |             0.475395 |  0.542918    |  5.78961 |   0.0150115  |
| GaussianProcessRegressor      |             0.232243 |  0.331063    |  7.00397 |   0.0353706  |
| QuantileRegressor             |            -0.147923 | -0.000170336 |  8.56424 |   2.94266    |
| DummyRegressor                |            -0.174516 | -0.0233405   |  8.66288 |   0.00723863 |
| LassoLars                     |            -0.174516 | -0.0233405   |  8.66288 |   0.00749469 |
| PassiveAggressiveRegressor    |            -0.325398 | -0.154802    |  9.2025  |   0.0126204  |
| KernelRidge                   |            -7.68369  | -6.56599     | 23.5551  |   0.0271327  |
+-------------------------------+----------------------+--------------+----------+--------------+

```
