# Criar um dataframe apartir de outro (usando for loop)

* Criamos o dataframe com dados fake.

```python
import pandas as pd

dados_fake = {'idade': [1,2,3,4,5,6],
			  'sexo': ['M','M','F','F', 'F', 'F'],
			  'cor': ['A', 'B', 'A', 'B', 'A', 'B']}

dataframe = pd.DataFrame(dados_fake) 

```

* Criamos um novo dataframe (o qual será populado)

```python

def loop_dataframe_column(df_column):
	'''
	Esta função 

	Inputs:
			df_column é

	Output:

	'''
	
	d = []

	for p in df_column:
		d.append(p)

	return d 

```

* Criamos um diccionário


```python
lista_idade = loop_dataframe_column(dataframe['idade'])
lista_sexo  = loop_dataframe_column(dataframe['sexo'])
lista_cor   = loop_dataframe_column(dataframe['cor'])

df_dict = {
			'idade': lista_idade, 
			'sexo': lista_sexo,
			'cor' : lista_cor
}

new_df = pd.DataFrame(df_dict)
```


### Casos especiais: Filtrar por 'Cor'

Agora criaremos um novo dataframe apartir de um existente, mas o novo dataframe
apenas terá cores específicas. 