# <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 
#########################################################
# 1. Predicting Heart Disease Using a Classification Tree
#########################################################
# <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 

#########################################################
# 1.1 Data Cleaning
#########################################################

# Upload data
library(readr)
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df <- read_csv(file = url, col_names = FALSE, na = "?")
dim(df)

# Rename the variables in the following order: ['age', 'sex', 'cp', 'restbp', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'hd'], and remove missing values.
colnames(df) <- c('age', 'sex', 'cp', 'restbp', 'chol', 'fbs',
                  'restecg', 'thalach', 'exang', 'oldpeak',
                  'slope', 'ca', 'thal', 'hd')

df <- na.omit(df)
df <- as.data.frame(lapply(df, as.numeric))

# Convert all categorical variables into dummy variables.
install.packages("fastDummies")
library(fastDummies)
categorical_vars <- c('sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal')
df_dummies <- dummy_cols(df, select_columns = categorical_vars, remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# Create a binary variable y that equals 1 if the person has heart disease and 0 otherwise.
y <- as.integer(df$hd > 0)
table(y)
df$y <- y
head(df[, c("hd", "y")])

#########################################################
# 1.2 Data Analysis
#########################################################

# Split the data into training and test sets, and plot a classification tree (random_state = 123).
install.packages("rpart.plot")
library(rpart)      # Para árboles de decisión
library(rpart.plot) # Para visualizar árboles de decisión
library(caret)      # Para partición de datos y evaluación
set.seed(123)       # Para reproducibilidad

X <- df[, !(names(df) %in% c("hd", "y"))]   # todas las columnas excepto 'hd' y 'y'
y <- df$y                                   # variable binaria objetivo

train_index <- createDataPartition(y, p = 0.7, list = FALSE) # Dividir en conjuntos de entrenamiento (70%) y prueba (30%)
X_train <- X[train_index, ]
X_test  <- X[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]

tree <- rpart( #Entrenar el árbol de decisión SIN PODA (cp = 0)
  y_train ~ ., 
  data = data.frame(X_train, y_train),
  method = "class",
  control = rpart.control(
    cp = 0,        # <---- sin poda inicial
    minsplit = 2,  # permitir que crezca bastante
    minbucket = 1,
    maxdepth = 30  # profundidad máxima
  )
)

rpart.plot(tree,               # Graficar el árbol
           type = 2,           # muestra los nombres de variables
           extra = 104,        # muestra conteo y porcentaje
           under = TRUE,       
           faclen = 0,         # no acorta nombres
           box.palette = "RdYlGn", 
           shadow.col = "gray", 
           main = "Classification Tree for Heart Disease")

summary(tree)

# Plot the confusion matrix using the labels ["Does not have HD", "Has HD"] and interpret the results.
y_pred <- predict(tree, X_test, type = "class") # Predicciones sobre el conjunto de prueba
cm <- confusionMatrix(as.factor(y_pred), as.factor(y_test), # Crear la matriz de confusión
                      positive = "1")   # '1' es "Has HD"
print(cm)

# Interpretation:
# The classification tree predicts whether someone has heart disease with 76% accuracy — meaning it gets about 3 out of 4 people right.
# 36 people were correctly predicted as healthy.
# 32 people were correctly predicted as having heart disease.
# 10 people were predicted to have the disease but actually don’t (false alarms).
# 11 people were predicted as healthy but actually have the disease (missed cases).
# So, the model does a good job overall, balancing sensitivity (74%) and specificity (78%), but it still misses some real cases and sometimes raises unnecessary alerts.

# Fix the overfitting problem using ross-validation.
# Generate 50 values of α equally spaced on a logarithmic scale between e⁻¹⁰ and 0.05.
# Use 4-fold cross-validation to select the optimal alpha (random_state = 123).
alphas <- 10^(seq(log10(exp(-10)), log10(0.05), length.out = 50)) # Generar 50 valores de alpha en una escala logarítmica entre e^-10 y 0.05
alphas

y_train <- as.factor(y_train) # Make sure y_train is a factor for classification

cv_scores <- numeric(length(alphas)) # Vector to store mean CV accuracies

for (i in seq_along(alphas)) {
  a <- alphas[i]
  
    model <- train( # Train a decision tree with given alpha (cp)
    y_train ~ ., 
    data = data.frame(X_train, y_train),
    method = "rpart",
    trControl = trainControl(method = "cv", number = 4),
    tuneGrid = data.frame(cp = a),
    metric = "Accuracy"
  )
  
  cv_scores[i] <- model$results$Accuracy # Store mean accuracy
}

cv_results <- data.frame(alpha = alphas, mean_CV_accuracy = cv_scores) # Combine results
print(cv_results)

best_index <- which.max(cv_scores)
best_alpha <- alphas[best_index]
cat("Best alpha:", best_alpha, "\n")
cat("Best mean CV accuracy:", max(cv_scores), "\n")

# Plot the Inaccuracy Rate (1 − Accuracy) against alpha.
inaccuracy <- 1 - cv_scores

plot(alphas, inaccuracy, 
     log = "x",          # escala logarítmica en el eje X
     type = "b",         # puntos conectados con líneas
     pch = 19,           # tipo de punto sólido
     col = "blue",
     xlab = "Alpha (cp)",
     ylab = "Inaccuracy Rate (1 - Accuracy)",
     main = "Inaccuracy Rate vs Alpha")

grid()

# Plot again the classification tree and the confusion matrix using the optimal α. Interpret and briefly discuss the results.
pruned_tree <- rpart( # Entrenar el árbol podado con el alpha óptimo (best_alpha)
  y_train ~ ., 
  data = data.frame(X_train, y_train),
  method = "class",
  control = rpart.control(cp = best_alpha)
)

rpart.plot( # Graficar el arbol podado
  pruned_tree,
  type = 2,             # mostrar nombres de variables
  extra = 104,          # muestra conteo y porcentaje
  under = TRUE,
  box.palette = "RdYlGn",
  shadow.col = "gray",
  main = paste("Pruned Classification Tree (Alpha =", round(best_alpha, 5), ")")
)

y_pred <- predict(pruned_tree, X_test, type = "class") # Evaluar desempeño en el conjunto de prueba
cm <- confusionMatrix(as.factor(y_pred), as.factor(y_test), positive = "1")
print(cm)

cat("Test Accuracy:", round(cm$overall["Accuracy"], 3), "\n") # Mostrar exactitud final

# Interpretation
# After pruning, the classification tree became smaller and easier to interpret.
# The test accuracy became 76%.
# The pruned model likely generalizes better and avoids overfitting, even if it sacrifices a few correct predictions on this sample.