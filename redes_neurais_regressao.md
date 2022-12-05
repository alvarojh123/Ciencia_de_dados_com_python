# Redes neurais usando Scikit-Learn


## Redes neurais: Uma única arquitetura

* Carregamos as librerias

```python
# Train_test split
from sklearn.model_selection import train_test_split
# Redes neurais
from sklearn.neural_network import MLPRegressor
# Métricas
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import  mean_absolute_percentage_error

```

* Dividimos os dados em dados de treino e teste

```python

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
```

* Definimos a arquitertura, construimos o modelo de Redes neurais e corremos o modelo.

```python

# Definimos a arquitetura
arquitetura = (3,3,3)

# Rede neural 
model_nn = MLPRegressor(hidden_layer_sizes=arquitetura, activation='relu', solver='adam', max_iter=500)
model_nn.fit(X_train, y_train)


# Métricas: Coef de Pearson (dados de treino)
r_sq_treino = r2_score(y_train, y_train_pred)
mae_treino = mean_absolute_error(y_train, y_train_pred)
mape_treino = mean_absolute_percentage_error(y_train, y_train_pred)

# Predição (dados de teste)

# Cálculo do coeff de Pearson usando os dados de treino
r_sq_teste = r2_score(y_test, y_test_pred)
mae_teste = mean_absolute_error(y_test, y_test_pred)
mape_teste = mean_absolute_percentage_error(y_test, y_test_pred)

```

* Predições com os dados de treino e teste. 

```python
# Predição (dados de treino)
y_train_pred = model_nn.predict(X_train)
# Predição (dados de teste)
y_test_pred = model_nn.predict(X_test)
```

```python
# Predição (dados de treino)
y_train_pred = model_nn.predict(X_train)
y_test_pred = model_nn.predict(X_test)


