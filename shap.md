# Explicando modelos de machine learning usando SHAP

**Alvaro J. Lopez** 

Neste artigo vou a ensinar como usar a biblioteca SHAP (SHapley Additive exPlanations) para interpretar os modelos de machine learning.

Os modelos de machine learning são muitas vezes usados como caixas pretas, ou seja após treinado o modelo 
é muito dificil extrair informações simples do modelo com a finalidade de explicar como cada preditor
contribui para o modelo.

Uma forma de fazer acessível o modelo de machine learning é usando o algoritmo SHAP. 

Aqui vamos a usar um modelo previsamente treinado usando redes neurais [link](./redes_neurais_regressão.md). Esse modelo esta armazenado no objeto  ``` model.nn```

A continuação vou mostrar o passo a passo para implementar o SHAP em python.

* Instalamos as librerias

```python
pip install shap
```


* Importamos as librerias


