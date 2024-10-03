import pandas as pd
import numpy as np
import seaborn as sns
import joblib
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
data["Grade"] = (data["G1"] + data["G2"] + data["G3"])/6
data = data.drop(columns = ['G1', 'G2', 'G3'])
data.head()

data['resultado'] = data['Grade'].apply(lambda x: 0 if x <= 2 else 2 if x >= 5 else 1)
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
plt.figure(figsize = (15, 20))
sns.heatmap(matriz, annot = True, fmt = ".1f")

data = data.drop(columns = ['school', 'Pstatus', 'activities', 'nursery', 'freetime', 'Dalc','absences', 'address', 'Mjob', 'Fjob',
       'reason', 'guardian', 'failures','schoolsup', 'famsup', 'paid', 'higher', 'internet', 'romantic',
       'famrel', 'goout', 'Walc', 'health'])

# APLICAÇÃO DE ML 
# divisão dos dados em treino e teste
variaveis  = data.drop(columns=["resultado", "Grade"])
resultado = data["resultado"]

variaveis_treino, variaveis_teste, result_treino, result_teste = train_test_split(
    variaveis, resultado, shuffle=True)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Lista de classificadores a serem testados
classificadores = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB()
}

# Função para treinar e avaliar diferentes classificadores
def testar_classificadores(classificadores, variaveis_treino, result_treino, variaveis_teste, result_teste):
    resultados = []
    
    for nome, clf in classificadores.items():
        clf.fit(variaveis_treino, result_treino)
        predicao = clf.predict(variaveis_teste)
        
        acuracia = accuracy_score(result_teste, predicao)
        f1 = f1_score(result_teste, predicao, average='weighted')
        precisao = precision_score(result_teste, predicao, average='weighted')
        recall = recall_score(result_teste, predicao, average='weighted')
        
        resultados.append({
            "Classificador": nome,
            "Acurácia": acuracia,
            "F1-Score": f1,
            "Precisão": precisao,
            "Recall": recall
        })
    
    return pd.DataFrame(resultados)

# Testar e exibir os resultados
resultados = testar_classificadores(classificadores, variaveis_treino, result_treino, variaveis_teste, result_teste)
print(resultados)

# aplicando modelo de classificação 
classificador = LogisticRegression(max_iter=200) #65,65%
classificador.fit(variaveis_treino, result_treino)
variaveis_treino.columns
predicao = classificador.predict(variaveis_teste)

joblib.dump(classificador, 'modelo.pkl') 
model = joblib.load('modelo.pkl')

def prever(input_dados):
    input_dados = np.array(input_dados).reshape(1, -1)  
    resultado_previsto = model.predict(input_dados)
    return resultado_previsto