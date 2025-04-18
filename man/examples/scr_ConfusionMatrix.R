# 1) recode Iris
# to binary classification
# problem
iris$species_num <- as.numeric(
  iris$Species == "virginica"
)

# 2) fit the logistic
# regression
model <- glm(
  formula = species_num ~ Sepal.Length + Sepal.Width,
  data    = iris,
  family  = binomial(
    link = "logit"
  )
)

# 3) generate predicted
# classes
predicted <- factor(
  as.numeric(
    predict(model, type = "response") > 0.5
  ),
  levels = c(1,0),
  labels = c("Virginica", "Others")
)

# 3.1) generate actual
# classes
actual <- factor(
  x = iris$species_num,
  levels = c(1,0),
  labels = c("Virginica", "Others")
)

# 4) summarise performance
# in a confusion matrix

# 4.1) unweighted matrix
confusion_matrix <- cmatrix(
  actual    = actual,
  predicted = predicted
)

# 4.1.1) summarise matrix
summary(
  confusion_matrix
)

# 4.1.2) plot confusion
# matrix
plot(
  confusion_matrix
)

# 4.2) weighted matrix
confusion_matrix <- weighted.cmatrix(
  actual    = actual,
  predicted = predicted,
  w         = iris$Petal.Length/mean(iris$Petal.Length)
)

# 4.2.1) summarise matrix
summary(
  confusion_matrix
)

# 4.2.1) plot confusion
# matrix
plot(
  confusion_matrix
)