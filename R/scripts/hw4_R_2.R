# 2. Causal Forest (10 puntos)
# Traducción aproximada del notebook original en Python a R (RStudio).
# Comentarios: He dividido el script en secciones para que sea más fácil de seguir.

# -----------------------------
# 1) Cargar librerías y datos
# -----------------------------
library(tidyverse)    # data wrangling + ggplot2
library(data.table)   # fread si el archivo es grande
library(grf)          # causal forest
library(rpart)        # árbol representativo
library(rpart.plot)   # para graficar el árbol
library(grf)


# Ajusta la ruta a tu archivo local
# En el notebook original la ruta apuntaba a: C:\Users\Dafne\...\processed.cleveland.data
ruta <- "C:/Users/FERNANDO/Downloads/processed.cleveland.data"

# Leer datos (si es CSV sin encabezado)
df <- fread(ruta, header = FALSE, data.table = FALSE)

# Asignar nombres de columnas según el notebook
colnames(df) <- c('age','sex','cp','restbp','chol','fbs','restecg','thalach',
                  'exang','oldpeak','slope','ca','thal','hd')

# Revisar dimensiones y primeras filas
dim(df)
head(df)

# -----------------------------
# 2) Preprocesamiento / limpieza
# -----------------------------
# En el notebook original parece que la variable objetivo se llama `hd`.
# Convertir variables categóricas a factores según sea necesario
factor_vars <- c('sex','cp','fbs','restecg','exang','slope','ca','thal','hd')
df[factor_vars] <- lapply(df[factor_vars], function(x) as.factor(x))

# Revisar valores únicos en hd
unique(df$hd)

# Si hd tiene clases 0/1 o similar, asegurarse de binaria
# (Aquí dejamos tal cual; adaptar si hd es multiclase)

# Tratar valores faltantes si existen (ejemplo simple)
# df <- df %>% mutate_all(~ ifelse(. == "?", NA, .))
# df <- df %>% drop_na()

# -----------------------------
# 3) Definir tratamiento, outcome y covariables
# -----------------------------
# En causal inference necesitamos una variable de tratamiento T y un outcome Y.
# El notebook original mezclaba modelos supervisados; aquí proponemos un ejemplo
# en el que: T = exang (ejemplo: presencia de angina con ejercicio),
# Y = thalach (frecuencia cardiaca máxima) o hd (presencia de enfermedad) — ajustar según objetivo.

# Ejemplo 1: Tratamiento binario T (exang), Outcome continuo Y (thalach)
# Convertir T a numérica 0/1
df$T <- as.numeric(as.character(df$exang))
# Si exang es factor con niveles no numéricos, hacerlo así:
if(any(is.na(df$T))) df$T <- as.numeric(df$exang) - 1

# Outcome: usar thalach (numérico)
df$Y <- as.numeric(df$thalach)

# Covariables X: todas las demás columnas excepto Y, T
covariables <- setdiff(names(df), c('Y','T','exang'))
X <- df %>% select(all_of(covariables))

# Convertir factores a dummies para grf (necesita matriz numérica)
X_model <- model.matrix(~ . -1, data = X) %>% as.matrix()

# Asegurarse que T y Y sean vectores numéricos
T_vec <- as.numeric(df$T)
Y_vec <- as.numeric(df$Y)

# -----------------------------
# 4) Entrenar un Causal Forest (grf)
# -----------------------------
set.seed(123)
# Ajuste básico de causal_forest
cf <- causal_forest(X_model, Y_vec, T_vec, num.trees = 2000)

# Estimar efectos individuales de tratamiento (CATE)
cate_hat <- predict(cf)$pred

# Resumen de efectos
summary(cate_hat)
mean(cate_hat)

# Estimar ATE (efecto promedio)
ate_est <- average_treatment_effect(cf, target.sample = "all")
ate_est

# -----------------------------
# 5) Diagnósticos y visualizaciones
# -----------------------------
# Distribución de las estimaciones CATE
library(ggplot2)
qplot(cate_hat, bins = 40) +
  labs(title = "Distribución de CATE estimados por causal_forest",
       x = "CATE estimado", y = "Frecuencia")

# Ver importancia de variables
var_imp <- variable_importance(cf)
var_imp_df <- data.frame(variable = colnames(X_model), importance = var_imp)
var_imp_df <- var_imp_df %>% arrange(desc(importance))
var_imp_df

# Mostrar top 10 variables
head(var_imp_df, 10)

# -----------------------------
# 6) Árbol representativo (ejemplo alternativo)
# -----------------------------
# En el notebook se graficó un árbol con profundidad 2. Aquí entrenamos un árbol
# simple para ver reglas que separan subgrupos con distinto CATE.

# Añadir la estimación CATE al dataframe original
df$cate_hat <- cate_hat

# Construir un árbol que prediga cate_hat usando las covariables (ejemplo)
# Convertir datos a un data.frame con variables predictoras limpias
tree_data <- cbind(df %>% select(all_of(covariables)), data.frame(cate_hat = df$cate_hat))
# Si hay factores en tree_data, rpart los maneja bien.

tree <- rpart(cate_hat ~ ., data = tree_data, control = rpart.control(maxdepth = 2))

# Graficar el árbol
rpart.plot(tree, main = "Árbol representativo (maxdepth = 2)")

# -----------------------------
# 7) Estimaciones por subgrupos (ejemplo)
# -----------------------------
# Podemos comparar promedio de cate_hat por un factor, por ejemplo 'sex'
cate_by_sex <- df %>% group_by(sex) %>% summarize(mean_cate = mean(cate_hat, na.rm = TRUE), n = n())
print(cate_by_sex)

