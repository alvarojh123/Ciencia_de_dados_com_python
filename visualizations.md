### Box-plot

```python


df = pd.DataFrame({'idade': [34,34,35,35,35,35,55,42,35,56,34,23,2,98,23, 3],
				   'UF': ['sp','sp','sp','sp','sp','sp','sp','sp','rj','rj','rj','rj','rj','rj','rj','rj']})
```


* Usando Seaborn

```python
import seaborn as sns


sns.set(rc={'figure.figsize':(15,8.27)})
ax = sns.boxplot(x='UF', y='idade', data=df, color='#99c2a2')
#ax.tick_params(axis='x', rotation=90)
ax.set(xlabel='UF', ylabel='Idade (anos)')
plt.show()


```