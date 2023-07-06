# Salvar um dataframe no drive

**Alvaro J. Lopez**

Agora vamos a aprender como salvar um dataframe (df) no drive usando o colab. 


* Carregamos as librerias

```python
import pandas as pd
```

* Definir o nome do ariquivo e o path (lugar onde vai ser salvo o df)


```python
nome_archivo = 'meu_df_quero_exportar.csv'
path = '/content/drive/My Drive/lugar_para_salvar/' + nome_archivo
```

* Salvar o df

```python
df.to_csv(path_2019, index=False)

```