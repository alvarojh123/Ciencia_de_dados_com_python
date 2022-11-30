# Redes neurais usando Scikit-Learn

* Carregamos as librerias

```python
# Train_test split
from sklearn.model_selection import train_test_split
# Redes neurais
from sklearn.neural_network import MLPRegressor
# Métricas
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

```

* Dividimos os dados em dados de treino e teste

```python

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
```