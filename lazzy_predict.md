# Predições com lazypredict

**Alvaro J. Lopez** 


```lazypredict``` é uma libreria que nos permite testar diversos algoritmos de machine learning
usando apenas poucas linhas de código. Assim graças a essa libreria podemos explorar diversos 
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
