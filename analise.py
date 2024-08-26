import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# Importando um CSV do Google Drive
url = 'https://drive.google.com/file/d/1ptToITq-hU03kcv0K3-vkgcAkLt4sNj5/view?usp=sharing'
# Pegando o ID do arquivo a partir da URL
file_id = url.split('/')[-2]
# Criando a URL para download direto no ambiente
pogo = 'https://drive.google.com/uc?id=' + file_id

data = pd.read_csv(pogo)


# análise exploratoria 
data.head()
data.info()
data.describe()


# verificar se tem valores nulos 
data.isnull().sum()
data.isna().sum()


#Criando uma coluna "Grade" para servir como parâmetro em vez das notas dos 3 trimestres
#A nova coluna será a média das notas dos 3 trimestres
data["Grade"] = (data["G1"] + data["G2"] + data["G3"])/3
data = data.drop(columns = ['G1', 'G2', 'G3'])
data.head()

data['resultado'] = data['Grade'].apply(lambda x: 0 if x < 2.5 else 2 if x > 10 else 1)
data.head()
data['resultado'].value_counts()


#Convertendo variáveis categóricas em formato numérico
label_encoder = LabelEncoder()
# Loop through each column in the DataFrame
for column in data.columns:
    # Check if the column is of categorical type
    if data[column].dtype == 'object':
        # Apply label encoding to the column
        data[column] = label_encoder.fit_transform(data[column])

data


#visualização de dados 
#matriz de correlacao
matriz = data.corr(numeric_only=True)
plt.figure(figsize = (15, 20));
sns.heatmap(matriz, annot = True, fmt = ".1f")

data = data.drop(columns = ['school', 'Pstatus', 'activities', 'nursery', 'freetime', 'Dalc','absences'])


# APLICAÇÃO DE ML 
# divisão dos dados em treino e teste
variaveis  = data.drop(columns=["resultado", "Grade"])
resultado = data["resultado"]

variaveis_treino, variaveis_teste, result_treino, result_teste = train_test_split(
    variaveis, resultado, shuffle=True)

# aplicando modelo de classificação 
classificador = RandomForestClassifier()
classificador.fit(variaveis_treino, result_treino)

predicao = classificador.predict(variaveis_teste)
precisao = accuracy_score(result_teste, predicao)

#usando todos os critérios
print("A precisão é de: {:.2f}%".format(precisao * 100))
#63,64%
#63,64%