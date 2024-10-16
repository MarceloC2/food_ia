# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Função para carregar e traduzir os dados
def carregar_dados(caminho):
    data = pd.read_csv(caminho)
    translate_columns = {
        'store_sales(in millions)': 'Venda_Loja_milhoes',
        'unit_sales(in millions)': 'Vendas_unidades_milhões',
        'total_children': 'Total_crianca',
        'num_children_at_home': 'Crianca_por_familia',
        'avg_cars_at home(approx).1': 'Media_carros_aprox',
        'gross_weight': 'Peso_bruto_produtos',
        'recyclable_package': 'Embalagem_reciclavel_binario',
        'low_fat': 'Baixo_teor_gordura_binario',
        'units_per_case': 'Unidades_por_caixa',
        'store_sqft': 'Area_loja',
        'coffee_bar': 'Cafeteria_binario',
        'video_store': 'Locadora_video_binario',
        'salad_bar': 'Barra_saladas_binario',
        'prepared_food': 'Comida_preparada_binario',
        'florist': 'Florista_binario',
        'cost': 'Custo'
    }
    data.rename(columns=translate_columns, inplace=True)
    return data

# Carregar os dados
train_data = carregar_dados(r'dados//train_dataset.csv')
test_data = carregar_dados(r'dados//test_dataset.csv')

# Divisão de dados
X = train_data.drop(columns=['Custo'])
y = train_data['Custo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Função para treinar e avaliar modelos
def treinar_avaliar_modelo(modelo, X_train, y_train, X_test, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return rmse, mae

# Treinamento e avaliação dos modelos
modelos = {
    'Linear Regression': LinearRegression(),
    'SVM': SVR(),
    'Decision Tree': DecisionTreeRegressor()
}

resultados = {}
for nome, modelo in modelos.items():
    rmse, mae = treinar_avaliar_modelo(modelo, X_train_scaled, y_train, X_test_scaled, y_test)
    resultados[nome] = {'RMSE': rmse, 'MAE': mae}

# Exibir resultados
for nome, metricas in resultados.items():
    print(f'{nome} - RMSE: {metricas["RMSE"]:.4f}, MAE: {metricas["MAE"]:.4f}')

# Validação cruzada
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
print('Best parameters:', grid_search.best_params_)

# Análise dos coeficientes
coefficients = pd.DataFrame(modelos['Linear Regression'].coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Exemplo de previsão em tempo real
def prever_custo(novos_dados):
    novos_dados_scaled = scaler.transform(novos_dados)
    return modelos['Linear Regression'].predict(novos_dados_scaled)

# Função para o usuário testar
def prever_custo_usuario():
    print("Insira os valores para as seguintes características:")
    novos_dados = []
    for coluna in X.columns:
        while True:
            valor = input(f"{coluna}: ")
            try:
                valor = float(valor)
                novos_dados.append(valor)
                break
            except ValueError:
                print("Por favor, insira um número válido.")
    novos_dados = np.array(novos_dados).reshape(1, -1)
    custo_previsto = prever_custo(novos_dados)
    print(f"Custo previsto: {custo_previsto[0]:.2f}")

# Chamar a função para o usuário testar
prever_custo_usuario()
# 5.73,3.0,5.0,5.0,3.0,18.7,1.0,0.0,30.0,20319.0,0.0,0.0,0.0,0.0,0.0,118.36