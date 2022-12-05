# Predições com Lazy Predict

**Alvaro J. Lopez** 


A libreria chamada de ```lazypredict``` permite a utilização de uma diversos 
modelos de machine learning usando apenas uma línea de código.

A libreria ```lazypredict``` pode ser usada para resolver problemas de 
regressão (usando o módulo ```LazyRegressor```) e classificação (usando o módulo ```LazyClassifier```)

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

* Carregamos os dados e definimos as variáveis dependentes e independentes.

```python
# Carregar os dados
data = datasets.load_breast_cancer()
X, y = data.data, data.target
```

* Dividimos os dados em dados de treino e teste

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
```

* Realizar o fit dos modelos do ```LazyClassifier```

```python
# Fit dos modelos
clf = LazyClassifier(predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
```

* Imprimir os resultados

Os


`
## Problema de Regressão:  ```LazyRegressor```

