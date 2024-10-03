import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder
from analise import prever

# Função para codificar variáveis categóricas
def encode_labels(value, categories):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(categories)[value]
# Mapeamento do resultado previsto para uma descrição textual
def traduzir_previsao(valor):
    if valor == 0:
        return "Reprovado :("
    elif valor == 1:
        return "De recuperação..."
    elif valor == 2:
        return "Aprovado!"
    else:
        return "Resultado desconhecido"

# Título do aplicativo
st.title("Previsão de nota de estudante do ensino médio.")
st.markdown(
    """
    **A previsão de sua situação escolar é baseada na média das notas dos trimestres. A classificação funciona da seguinte maneira:**

    - **Reprovado:** Nota final inferior a 2
    - **Recuperação:** Nota final entre 2 e 5
    - **Aprovado:** Nota final superior a 5
    """
)

st.subheader("Insira seus dados e descubra sua situação ao final do ano letivo!")
# Input dos dados do usuário para as variáveis utilizadas no modelo
sexo = st.selectbox('Sexo', ['Masculino', 'Feminino'])
idade = st.slider('Idade', 15, 22, 18)
familia = st.selectbox('Tamanho da família (maior ou menor que 3 membros)', ['maior', 'menor'])
educacao_mae = st.selectbox('Educação da Mãe', ['não tem educação', 'fundamental completo', 'ensino médio completo', 'ensino médio técnico completo', 'ensino superior completo'])
educacao_pai = st.selectbox('Educação do Pai', ['não tem educação', 'fundamental completo', 'ensino médio completo', 'ensino médio técnico completo', 'ensino superior completo'])
tempo_viagem = st.selectbox('Tempo de viagem para a escola', ['viagem curta', 'viagem média', 'viagem longa', 'viagem muito longa'])
estudo_semana = st.selectbox('Horas de estudo por semana', ['menos de 2h', 'entre 2h a 5h', 'entre 5h a 10h', 'mais de 10h'])

# Codificando as variáveis categóricas
sexo_encoded = encode_labels(['Masculino', 'Feminino'].index(sexo), ['Masculino', 'Feminino'])
familia_encoded = encode_labels(['maior', 'menor'].index(familia), ['maior', 'menor'])
educacao_mae_encoded = encode_labels(['não tem educação', 'fundamental completo', 'ensino médio completo', 'ensino médio técnico completo', 'ensino superior completo'].index(educacao_mae), ['não tem educação', 'fundamental completo', 'ensino médio completo', 'ensino médio técnico completo', 'ensino superior completo'])
educacao_pai_encoded = encode_labels(['não tem educação', 'fundamental completo', 'ensino médio completo', 'ensino médio técnico completo', 'ensino superior completo'].index(educacao_pai), ['não tem educação', 'fundamental completo', 'ensino médio completo', 'ensino médio técnico completo', 'ensino superior completo'])
tempo_viagem_encoded = encode_labels(['viagem curta', 'viagem média', 'viagem longa', 'viagem muito longa'].index(tempo_viagem), ['viagem curta', 'viagem média', 'viagem longa', 'viagem muito longa'])
estudo_semana_encoded = encode_labels(['menos de 2h', 'entre 2h a 5h', 'entre 5h a 10h', 'mais de 10h'].index(estudo_semana), ['menos de 2h', 'entre 2h a 5h', 'entre 5h a 10h', 'mais de 10h'])

# Criar o array com os dados codificados
input_dados = [
    sexo_encoded, 
    idade, 
    familia_encoded, 
    educacao_mae_encoded, 
    educacao_pai_encoded, 
    tempo_viagem_encoded, 
    estudo_semana_encoded
]

# Fazendo a previsão com a função do arquivo analise.py
if st.button('Prever'):
    # Fazendo a previsão com a função do arquivo analise.py
    resultado_previsto = prever(input_dados)
    # Traduzir o resultado numérico para uma descrição textual
    resultado_texto = traduzir_previsao(resultado_previsto[0])
    st.write(f'Resultado previsto: {resultado_texto}')
    
# Centralizando o botão
st.markdown(
    """
    <style>
    div.stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)