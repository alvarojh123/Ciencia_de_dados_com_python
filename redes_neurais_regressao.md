# Redes neurais usando Scikit-Learn

**Alvaro J. Lopez**

Neste post vou a ensinar a usar modelos de redes neurais para modelar problemas
de regressão e classificação. 

Também vamos a aprender a salvar modelos treinados para seu posterior uso.

## Redes neurais: Regressão

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

* Carregamos os dados

```python
from sklearn import datasets
data = datasets.load_boston()
X, y = data.data, data.target
```

* Dividimos os dados em dados de treino e teste

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

* Definimos a arquitetura, construimos o modelo de Redes neurais e corremos o modelo.

```python
# Definimos a arquitetura
arquitetura = (3,3,3)

# Modelos de Rede neural 
model_nn = MLPRegressor(hidden_layer_sizes=arquitetura, activation='relu', solver='adam', max_iter=500)
model_nn.fit(X_train, y_train)
```

* Avaliar a performance do modelo usando métricas 

```python
# Métricas com os dados de treino
r_sq_treino = r2_score(y_train, y_train_pred)
mae_treino = mean_absolute_error(y_train, y_train_pred)
mape_treino = mean_absolute_percentage_error(y_train, y_train_pred)

# Métricas com os dados de teste
r_sq_teste = r2_score(y_test, y_test_pred)
mae_teste = mean_absolute_error(y_test, y_test_pred)
mape_teste = mean_absolute_percentage_error(y_test, y_test_pred)
```

* Predições 

```python
# Predição (dados de treino)
y_train_pred = model_nn.predict(X_train)
# Predição (dados de teste)
y_test_pred = model_nn.predict(X_test)
```


## Salvar o modelo treinado

Salvar os modelos treinados é necessário para evitar repetir o processo 
de treinamento. 
Também é necessário quando queremos compartilhar o nosso modelo treinado com alguém.

É necessário lembrar duas definições importantes:

* _Serialization_: Processo de salvar os dados (ou modelos)
* _Deserialization_: Processo de carregar os dados (ou modelos)



### Salvar modelos com Joblib

* Carregar as librerias

```python
# Forma antiga de importar joblib
#from sklearn.externals import joblib  
import joblib
```

* Salvar o modelo

Para salvar o modelo, precisamos passar como input o modelo
previamente treinado (neste caso, ```model_nn```)

```python
joblib_file = 'modelo_rede_neural.pkl'
joblib.dump(model_nn, joblib_file)

```

* Carregar o modelo

```python
joblib_modelo = joblib.load(joblib_file)
```

