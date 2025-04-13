
data <- read.csv("train_mod1.csv")
View(data)
summary(data)

library(lubridate)
library(dplyr)
library(ggplot2)
library(reshape2)
library(e1071)
library(caret)
library(caTools)
library(readr)
library(naivebayes)


missing_values <- colSums(is.na(data))
print("Missing Values:")
print(missing_values)

empty_string <- colSums(data == "" | data == " ")
print("Empty string:")
print(empty_string)

unique_counts <- sapply(data, function(x) length(unique(x)))
unique_counts

numeric_columns <- sapply(data, is.numeric)
for (col in names(data)[numeric_columns]) {
  col_mean <- mean(data[[col]], na.rm = TRUE) # Calculate mean for the column
  data[[col]][is.na(data[[col]])] <- col_mean # Replace NAs with mean
}

data <- data %>%
  mutate_if(is.numeric, ~round(., 2))




data$RainTomorrow <- as.factor(data$RainTomorrow)
p_values <- vector()
for (col in names(data)[-which(names(data) == "RainTomorrow")]) {
  data[[col]] <- as.factor(data[[col]])
  test_result <- chisq.test(table(data[[col]], data$RainTomorrow))
  p_values[col] <- test_result$p.value
}
print(p_values)
significant_attributes <- names(p_values)[p_values < 0.05]
print(significant_attributes)

  for(col in names(data)){
    if(is.factor(data[[col]]) || is.character(data[[col]])){
      data[[col]] <- as.integer(factor(data[[col]]))
    }
  }

get_mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
columns_to_replace <- c("WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow")
for (col in columns_to_replace) {
  mode_value <- get_mode(data[[col]][!is.na(data[[col]])])
  data[[col]][is.na(data[[col]])] <- mode_value
}


num_col <- sapply(data, is.numeric)
for (col in names(data)[num_col]) {
  boxplot(data[[col]], main = paste(col, "Box Plot"), ylab = col)
}

evaporation_column <- data$Evaporation

replace_outliers_with_mean <- function(x) {
  Q1 <- quantile(x, 0.31, na.rm = TRUE)
  Q3 <- quantile(x, 0.69, na.rm = TRUE)
  IQR <- Q3 - Q1
  
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  
  mean_value <- mean(x[x >= lower_bound & x <= upper_bound], na.rm = TRUE)
  
  x[x < lower_bound | x > upper_bound] <- mean_value
  return(x)
}
data$Evaporation <- replace_outliers_with_mean(evaporation_column)



MaxTemp_column <- data$MaxTemp
replace_outliers_with_mean <- function(x) {
  Q1 <- quantile(x, 0.3, na.rm = TRUE)
  Q3 <- quantile(x, 0.7, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  mean_value <- mean(x[x >= lower_bound & x <= upper_bound], na.rm = TRUE)
  x[x < lower_bound | x > upper_bound] <- mean_value
  return(x)
}
data$MaxTemp <- replace_outliers_with_mean(MaxTemp_column)

Sunshine_column <- data$Sunshine
replace_outliers_with_mean <- function(x) {
  Q1 <- quantile(x, 0.3, na.rm = TRUE)
  Q3 <- quantile(x, 0.7, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  mean_value <- mean(x[x >= lower_bound & x <= upper_bound], na.rm = TRUE)
  x[x < lower_bound | x > upper_bound] <- mean_value
  return(x)
}
data$Sunshine <- replace_outliers_with_mean(Sunshine_column)



set.seed(123)  # for reproducibility
split <- sample.split(data$RainTomorrow, SplitRatio = 0.7)
train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)

train_data$RainTomorrow <- as.factor(train_data$RainTomorrow)
test_data$RainTomorrow <- as.factor(test_data$RainTomorrow)

control <- trainControl(method = "cv", number = 10)
model <- train(RainTomorrow~ ., data = train_data, method = "naive_bayes", trControl = control)

predictions <- predict(model, test_data)
confusionMatrix <- confusionMatrix(predictions, test_data$RainTomorrow)

recall <- confusionMatrix$byClass["Sensitivity"]
precision <- confusionMatrix$byClass["Pos Pred Value"]
F1 <- 2 * (precision * recall) / (precision + recall)
accuracy <- sum(predictions == test_data$RainTomorrow) / nrow(test_data)
print(paste("Accuracy on Test Set:", accuracy))

print(paste("Recall:", recall))
print(paste("Precision:", precision))
print(paste("F1 Score:", F1))
print(accuracy)
print(confusionMatrix)



confusionMatrix <- confusionMatrix(predictions, test_data$RainTomorrow)
confusionMatrixTable <- as.table(confusionMatrix$table)
ggplot(data = as.data.frame(confusionMatrixTable), aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "black") +
  geom_text(aes(label = sprintf("%0.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "yellow", high = "steelblue") +
  theme_minimal() +
  labs(fill = "Count")

