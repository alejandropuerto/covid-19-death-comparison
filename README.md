```python
# Libraries

import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dateutil.parser import parse
from pandas import Series
```

## Data from contralacorrupcion.mx


```python
# Reading the dataset
covid19 = pd.read_csv("actas-defuncion-covid-19-cdmx1.csv",parse_dates=['FECHA'], index_col = "FECHA")
covid19.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EDAD</th>
      <th>SEMANA</th>
      <th>MES</th>
      <th>RAZON</th>
      <th>ACTA</th>
    </tr>
    <tr>
      <th>FECHA</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-03-18</th>
      <td>41 AÑOS</td>
      <td>12</td>
      <td>3</td>
      <td>CHOQUE SEPTICO, NEUMONIA POR COVID 19 POR SARS...</td>
      <td>7129</td>
    </tr>
    <tr>
      <th>2020-03-23</th>
      <td>61 AÑOS</td>
      <td>13</td>
      <td>3</td>
      <td>INSUFICIENCIA RESPIRATORIA AGUDA, NEUMONIA VIR...</td>
      <td>4459</td>
    </tr>
    <tr>
      <th>2020-03-26</th>
      <td>60 AÑOS</td>
      <td>13</td>
      <td>3</td>
      <td>SINDROME DE INSUFICIENCIA RESPIRATORIA AGUDA, ...</td>
      <td>4591</td>
    </tr>
    <tr>
      <th>2020-03-26</th>
      <td>37 AÑOS</td>
      <td>13</td>
      <td>3</td>
      <td>NEUMONIA POR CORONAVIRUS</td>
      <td>7879</td>
    </tr>
    <tr>
      <th>2020-03-26</th>
      <td>63 AÑOS</td>
      <td>13</td>
      <td>3</td>
      <td>CERVICOVAGINITIS PURULENTA, CARCINOMA EPIDERMO...</td>
      <td>7829</td>
    </tr>
  </tbody>
</table>
</div>




```python
number_by_date = covid19.drop(['EDAD', 'SEMANA', 'MES', 'RAZON', 'ACTA'], axis=1)
number_by_date
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
    </tr>
    <tr>
      <th>FECHA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-03-18</th>
    </tr>
    <tr>
      <th>2020-03-23</th>
    </tr>
    <tr>
      <th>2020-03-26</th>
    </tr>
    <tr>
      <th>2020-03-26</th>
    </tr>
    <tr>
      <th>2020-03-26</th>
    </tr>
    <tr>
      <th>...</th>
    </tr>
    <tr>
      <th>2020-05-12</th>
    </tr>
    <tr>
      <th>2020-05-12</th>
    </tr>
    <tr>
      <th>2020-05-12</th>
    </tr>
    <tr>
      <th>2020-05-12</th>
    </tr>
    <tr>
      <th>2020-05-12</th>
    </tr>
  </tbody>
</table>
<p>4579 rows × 0 columns</p>
</div>




```python
number_by_date['MUERTES'] = 0
```


```python
number_by_date
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MUERTES</th>
    </tr>
    <tr>
      <th>FECHA</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-03-18</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-03-23</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-03-26</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-03-26</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-03-26</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-05-12</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-05-12</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-05-12</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-05-12</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-05-12</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4579 rows × 1 columns</p>
</div>




```python
number_by_date.drop(number_by_date.head(2).index, inplace=True)
```


```python
number_by_date = number_by_date.groupby('FECHA').count()
number_by_date.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MUERTES</th>
    </tr>
    <tr>
      <th>FECHA</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-03-26</th>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-03-27</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2020-03-28</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-03-29</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2020-03-30</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



## Data from gob.mx


```python
covid19_oficial = pd.read_csv("200521COVID19MEXICO.csv", sep = ",",parse_dates = ["FECHA_DEF"], encoding ='latin1')
covid19_oficial.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FECHA_ACTUALIZACION</th>
      <th>ID_REGISTRO</th>
      <th>ORIGEN</th>
      <th>SECTOR</th>
      <th>ENTIDAD_UM</th>
      <th>SEXO</th>
      <th>ENTIDAD_NAC</th>
      <th>ENTIDAD_RES</th>
      <th>MUNICIPIO_RES</th>
      <th>TIPO_PACIENTE</th>
      <th>...</th>
      <th>CARDIOVASCULAR</th>
      <th>OBESIDAD</th>
      <th>RENAL_CRONICA</th>
      <th>TABAQUISMO</th>
      <th>OTRO_CASO</th>
      <th>RESULTADO</th>
      <th>MIGRANTE</th>
      <th>PAIS_NACIONALIDAD</th>
      <th>PAIS_ORIGEN</th>
      <th>UCI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-05-21</td>
      <td>11e989</td>
      <td>2</td>
      <td>3</td>
      <td>27</td>
      <td>2</td>
      <td>27</td>
      <td>27</td>
      <td>4</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>99</td>
      <td>MÃ©xico</td>
      <td>99</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-05-21</td>
      <td>1aad65</td>
      <td>2</td>
      <td>4</td>
      <td>19</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>18</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>99</td>
      <td>1</td>
      <td>99</td>
      <td>MÃ©xico</td>
      <td>99</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-05-21</td>
      <td>04f631</td>
      <td>2</td>
      <td>4</td>
      <td>14</td>
      <td>1</td>
      <td>14</td>
      <td>14</td>
      <td>67</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>99</td>
      <td>1</td>
      <td>99</td>
      <td>MÃ©xico</td>
      <td>99</td>
      <td>97</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-05-21</td>
      <td>02556b</td>
      <td>2</td>
      <td>4</td>
      <td>15</td>
      <td>1</td>
      <td>15</td>
      <td>15</td>
      <td>110</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>99</td>
      <td>1</td>
      <td>99</td>
      <td>MÃ©xico</td>
      <td>99</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-05-21</td>
      <td>0356d5</td>
      <td>2</td>
      <td>4</td>
      <td>9</td>
      <td>1</td>
      <td>9</td>
      <td>9</td>
      <td>5</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>99</td>
      <td>1</td>
      <td>99</td>
      <td>MÃ©xico</td>
      <td>99</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-05-21</td>
      <td>1d2dfb</td>
      <td>2</td>
      <td>4</td>
      <td>25</td>
      <td>2</td>
      <td>14</td>
      <td>25</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>99</td>
      <td>1</td>
      <td>99</td>
      <td>MÃ©xico</td>
      <td>99</td>
      <td>97</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-05-21</td>
      <td>1b3e2b</td>
      <td>2</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>7</td>
      <td>9</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>99</td>
      <td>1</td>
      <td>99</td>
      <td>MÃ©xico</td>
      <td>99</td>
      <td>97</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-05-21</td>
      <td>0c0eef</td>
      <td>2</td>
      <td>4</td>
      <td>21</td>
      <td>1</td>
      <td>21</td>
      <td>21</td>
      <td>114</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>99</td>
      <td>1</td>
      <td>99</td>
      <td>MÃ©xico</td>
      <td>99</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2020-05-21</td>
      <td>043ea2</td>
      <td>2</td>
      <td>4</td>
      <td>27</td>
      <td>2</td>
      <td>27</td>
      <td>27</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>99</td>
      <td>1</td>
      <td>99</td>
      <td>MÃ©xico</td>
      <td>99</td>
      <td>97</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-05-21</td>
      <td>0bd39a</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>1</td>
      <td>8</td>
      <td>8</td>
      <td>17</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>99</td>
      <td>1</td>
      <td>99</td>
      <td>MÃ©xico</td>
      <td>99</td>
      <td>97</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 35 columns</p>
</div>




```python
temp = covid19_oficial[['FECHA_DEF', 'ENTIDAD_RES']]
```


```python
temp = temp.set_index('FECHA_DEF')
```


```python
temp
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENTIDAD_RES</th>
    </tr>
    <tr>
      <th>FECHA_DEF</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-04-27</th>
      <td>27</td>
    </tr>
    <tr>
      <th>2020-04-03</th>
      <td>5</td>
    </tr>
    <tr>
      <th>9999-99-99</th>
      <td>14</td>
    </tr>
    <tr>
      <th>2020-04-20</th>
      <td>15</td>
    </tr>
    <tr>
      <th>9999-99-99</th>
      <td>9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-05-14</th>
      <td>15</td>
    </tr>
    <tr>
      <th>9999-99-99</th>
      <td>26</td>
    </tr>
    <tr>
      <th>9999-99-99</th>
      <td>26</td>
    </tr>
    <tr>
      <th>9999-99-99</th>
      <td>15</td>
    </tr>
    <tr>
      <th>9999-99-99</th>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>201838 rows × 1 columns</p>
</div>




```python
from_cdmx = temp.loc[(temp['ENTIDAD_RES'] == 9)] #Number nine corresponds to CDMX as stated in the data dictionary
```


```python
from_cdmx
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENTIDAD_RES</th>
    </tr>
    <tr>
      <th>FECHA_DEF</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9999-99-99</th>
      <td>9</td>
    </tr>
    <tr>
      <th>9999-99-99</th>
      <td>9</td>
    </tr>
    <tr>
      <th>9999-99-99</th>
      <td>9</td>
    </tr>
    <tr>
      <th>9999-99-99</th>
      <td>9</td>
    </tr>
    <tr>
      <th>2020-03-22</th>
      <td>9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-05-05</th>
      <td>9</td>
    </tr>
    <tr>
      <th>9999-99-99</th>
      <td>9</td>
    </tr>
    <tr>
      <th>9999-99-99</th>
      <td>9</td>
    </tr>
    <tr>
      <th>9999-99-99</th>
      <td>9</td>
    </tr>
    <tr>
      <th>9999-99-99</th>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>46594 rows × 1 columns</p>
</div>




```python
from_cdmx.drop(['ENTIDAD_RES'], axis=1, inplace=True, errors='ignore')
```

    C:\Users\User\Anaconda3\lib\site-packages\pandas\core\frame.py:3997: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      errors=errors,



```python
from_cdmx
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
    </tr>
    <tr>
      <th>FECHA_DEF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9999-99-99</th>
    </tr>
    <tr>
      <th>9999-99-99</th>
    </tr>
    <tr>
      <th>9999-99-99</th>
    </tr>
    <tr>
      <th>9999-99-99</th>
    </tr>
    <tr>
      <th>2020-03-22</th>
    </tr>
    <tr>
      <th>...</th>
    </tr>
    <tr>
      <th>2020-05-05</th>
    </tr>
    <tr>
      <th>9999-99-99</th>
    </tr>
    <tr>
      <th>9999-99-99</th>
    </tr>
    <tr>
      <th>9999-99-99</th>
    </tr>
    <tr>
      <th>9999-99-99</th>
    </tr>
  </tbody>
</table>
<p>46594 rows × 0 columns</p>
</div>




```python
from_cdmx['MUERTES'] = 0
```

    C:\Users\User\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
from_cdmx = from_cdmx.groupby('FECHA_DEF').count()
```


```python
from_cdmx.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MUERTES</th>
    </tr>
    <tr>
      <th>FECHA_DEF</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-03-16</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-03-22</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-03-23</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-03-25</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-03-26</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
from_cdmx.drop(from_cdmx.tail(1).index, inplace=True) #9999-99-99 date is dropped
```


```python
from_cdmx = from_cdmx.reset_index()
```


```python
from_cdmx = from_cdmx.loc[(from_cdmx['FECHA_DEF'] >= '2020-03-26') &  (from_cdmx['FECHA_DEF'] <= '2020-05-12')]
```


```python
from_cdmx = from_cdmx.set_index('FECHA_DEF')
```


```python
print(from_cdmx.head(1))
print(from_cdmx.tail(1))
```

                MUERTES
    FECHA_DEF          
    2020-03-26        3
                MUERTES
    FECHA_DEF          
    2020-05-12       60



```python
from_cdmx = from_cdmx.reset_index()
from_cdmx['FECHA_DEF'] = pd.to_datetime(from_cdmx['FECHA_DEF'])
```


```python
from_cdmx = from_cdmx.set_index('FECHA_DEF')
```

## Visualization


```python
sns.set(rc={'figure.figsize':(14, 7)})
```


```python
start, end = '2020-03', '2020-05'
```


```python
fig, ax = plt.subplots()
ax.plot(from_cdmx.loc[start:end],
marker='o', markersize=8, linestyle='-', label='gob.mx')
ax.plot(number_by_date.loc[start:end],
marker='o', markersize=8, linestyle='-', label='contralacorrupcion.mx')
ax.set_xlabel('Date')
ax.set_ylabel('Number of Deaths')
ax.set_title('COVID-19 death comparison in CDMX')
ax.legend()
```




    <matplotlib.legend.Legend at 0x1946b7e5a58>




![death comparison](https://github.com/alejandropuerto/covid-19-death-comparison/blob/master/plot.png)



```python



```
