df = pd.read_csv("/content/drive/MyDrive/BPA/TP/DataTPClean.csv", encoding='ISO-8859-1', sep=",")

df.isnull().sum().sort_values(ascending=False) * 100 / len(df)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^\w\s]", "", regex=True)
)

df.head(5)

import seaborn as sb
import sklearn

df.describe()

df.saleprice.hist(bins=100)
plt.title("Histograma de Sales Price")
plt.xlabel('Sales Price')
plt.ylabel('Ventas');

df.lot_area.hist(bins=100)
plt.title("Histograma de Lot Area")
plt.xlabel('Lot Area')
plt.ylabel('Numero de Lotes');

df['yr_sold'].value_counts().plot(kind='bar')
plt.title("Bar Chart del año de venta de la Casa")
plt.xlabel('Sale Year')
plt.ylabel('Number of Sales');

df['sale_condition'].value_counts().plot(kind='bar')
plt.title("Bar Chart of Condition of Home at Time of Sale")
plt.xlabel('Condition')
plt.ylabel('Number of Sales');

"""Chequeamos las correlaciones"""

df_num = df.select_dtypes(include='number')

plt.figure(figsize=(9, 9))

dataplot = sb.heatmap(df_num.corr(), xticklabels=df_num.corr().columns, yticklabels=df_num.corr().columns,  cmap="YlGnBu", annot=False)
plt.savefig('corr.png', bbox_inches="tight")
plt.show();

df = df.drop(columns=['order'])

# Total de metros cuadrados (primer piso + segundo piso + sótano)
df['total_sf'] = (
    df['1st_flr_sf'] +
    df['2nd_flr_sf'] +
    df['total_bsmt_sf']
)

# Metros cuadrados terminados (excluye sótano sin terminar)
df['finished_sf'] = (
    df['1st_flr_sf'] +
    df['2nd_flr_sf'] +
    df['total_bsmt_sf'] -
    df['bsmt_unf_sf']
)

# Metros cuadrados útiles de alta calidad (excluye low quality finish)
df['quality_sf'] = (
    df['1st_flr_sf'] +
    df['2nd_flr_sf'] +
    df['total_bsmt_sf'] -
    df['low_qual_fin_sf']
)

# Total de baños (completos y medios baños, incluyendo sótano)
df['total_bath'] = (
    df['full_bath'] +
    0.5 * df['half_bath'] +
    df['bsmt_full_bath'] +
    0.5 * df['bsmt_half_bath']
)

# Total de dormitorios sobre rasante
df['total_bedrooms'] = df['bedroom_abvgr']

dfe = df.copy()

dfe.total_sf.hist(bins=100)
plt.title("Histogram of Total Square Footage")
plt.xlabel('Sqr. Feet')
plt.ylabel('Number of Sales');

numeric_cols = dfe.select_dtypes(include='number')

plt.figure(figsize=(10, 9))
dataplot = sb.heatmap(numeric_cols.corr(), cmap="YlGnBu", annot=False)
plt.show()

pt = numeric_cols[['lot_frontage',  'lot_area', 'mo_sold', 'yr_sold', 'saleprice', 'total_sf', 'finished_sf', 'total_bath', 'quality_sf']]

dataplot3 = sb.heatmap(pt.corr(), cmap="YlGnBu", annot=False)
plt.savefig('baby_corr.png', bbox_inches="tight")
plt.show()

# Copia del df original para trabajar con limpieza completa
df_clean = df.copy()

#En esta parte se eliminan las columnas con 50% o mas de datos faltantes
coverage = df_clean.isna().sum() / len(df_clean)
low_coverage_cols = coverage[coverage > 0.5].index.tolist()

df_clean.drop(columns=low_coverage_cols, inplace=True)

# Numéricos: imputamos con mediana
num_cols = df_clean.select_dtypes(include='number').columns
for col in num_cols:
    if df_clean[col].isna().sum() > 0:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

# Categóricos: imputamos con "None"
cat_cols = df_clean.select_dtypes(include='object').columns
for col in cat_cols:
    if df_clean[col].isna().sum() > 0:
        df_clean[col].fillna("None", inplace=True)

df_clean.isnull().sum().sort_values(ascending=False) * 100 / len(df)

set(df.columns) - set(df_clean.columns)

X = df_clean.drop(columns=['saleprice'])
y = df_clean['saleprice']

# Ignorar 'id' si la tienes
numeric_features = X.select_dtypes(include='number').columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Pipeline para numéricos
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Pipeline para categóricos
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Salida como DataFrame (en sklearn >= 1.2)
preprocessor.set_output(transform="pandas")

# Ajustar y transformar los datos
X_processed = preprocessor.fit_transform(X)

# Añadir 'sale_price' al resultado si deseas unirlos para modelar
df_model = X_processed.copy()
df_model['sale_price'] = y.values

df_model.head(5)

# 1. Verificar nulos
print("🔍 Nulos totales:", df_model.isna().sum().sum())

# 2. Tipos de datos
print("📌 Tipos de datos únicos:", df_model.dtypes.unique())

# 3. Columnas no numéricas
non_numeric_cols = df_model.select_dtypes(exclude=['number']).columns
print("🧱 Columnas NO numéricas:", list(non_numeric_cols))

# 4. ¿SalePrice está como target?
print("'sale_price' en columnas:", 'sale_price' in df_model.columns)

# 5. Dimensiones
print("📐 Shape:", df_model.shape)

"""ENTRENAMOS EL MODELO RANDONM FOREST."""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from math import sqrt

# Fijar semilla para reproducibilidad
np.random.seed(42)

# Separar X e y desde df_model
X = df_model.drop('sale_price', axis=1)
y = df_model['sale_price']

# División en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo Random Forest
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)

# Predecir sobre el set de validación
rf_preds = reg.predict(X_val)

# Evaluación
rmse = sqrt(mean_squared_error(y_val, rf_preds))
r2 = r2_score(y_val, rf_preds)

print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ R²: {r2:.4f}")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Fijar semilla para reproducibilidad
np.random.seed(42)

# Entrenar el modelo Gradient Boosting
gbr = GradientBoostingRegressor(loss='squared_error', random_state=42)
gbr.fit(X_train, y_train)

# Predecir sobre el set de validación
gbr_preds = gbr.predict(X_val)

# Evaluación
gbr_rmse = sqrt(mean_squared_error(y_val, gbr_preds))
gbr_r2 = r2_score(y_val, gbr_preds)

print(f"🌲 Gradient Boosting RMSE: {gbr_rmse:.2f}")
print(f"🌲 Gradient Boosting R²: {gbr_r2:.4f}")

from sklearn.linear_model import LinearRegression

# Entrenar modelo lineal
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predecir sobre validación
lr_preds = lr.predict(X_val)

# Evaluación
lr_rmse = sqrt(mean_squared_error(y_val, lr_preds))
lr_r2 = r2_score(y_val, lr_preds)

print(f"📈 Linear Regression RMSE: {lr_rmse:.2f}")
print(f"📈 Linear Regression R²: {lr_r2:.4f}")

import seaborn as sns
# Crear un DataFrame con los resultados
results = pd.DataFrame({
    'Modelo': ['Random Forest', 'Gradient Boosting', 'Linear Regression'],
    'R²': [r2, gbr_r2, lr_r2]
})

# Gráfico de barras
plt.figure(figsize=(8, 5))
sns.barplot(data=results, x='Modelo', y='R²', palette='viridis')
plt.title('Comparación de R² entre modelos')
plt.ylim(0, 1)
plt.ylabel('R² (Coeficiente de determinación)')
plt.xlabel('')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Crear DataFrame con los RMSE
rmse_results = pd.DataFrame({
    'Modelo': ['Random Forest', 'Gradient Boosting', 'Linear Regression'],
    'RMSE': [rmse, gbr_rmse, lr_rmse]
})

# Gráfico de barras
plt.figure(figsize=(8, 5))
sns.barplot(data=rmse_results, x='Modelo', y='RMSE', palette='magma')
plt.title('Comparación de RMSE entre modelos')
plt.ylabel('RMSE (Error cuadrático medio)')
plt.xlabel('')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

from sklearn.model_selection import GridSearchCV



# Definimos la grilla de hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200, 400],
    'learning_rate': [0.1, 0.5, 0.75, 1],
    'subsample': [0.75, 1]
}

# Modelo base
gbr_model = GradientBoostingRegressor(loss='squared_error', random_state=42)

# Grid Search con validación cruzada
grid_search = GridSearchCV(estimator=gbr_model, param_grid=param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)

# Entrenamiento
grid_search.fit(X_train, y_train)

# Mejores hiperparámetros encontrados
print("✅ Mejores parámetros:", grid_search.best_params_)

# Predecir sobre validación con el mejor modelo
gbr_best = grid_search.best_estimator_
gbr_cv_preds = gbr_best.predict(X_val)

# Evaluación
gbr_cv_rmse = sqrt(mean_squared_error(y_val, gbr_cv_preds))
gbr_cv_r2 = r2_score(y_val, gbr_cv_preds)

print(f"🎯 Gradient Boosted Tuned - RMSE: {gbr_cv_rmse:.2f}")
print(f"🎯 Gradient Boosted Tuned - R²: {gbr_cv_r2:.4f}")