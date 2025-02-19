## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment  = "#>",
  message  = FALSE
)

## ----setup--------------------------------------------------------------------
# load libraries
library(SLmetrics)

## ----data---------------------------------------------------------------------
# 1) load data
# from {mlbench}
data("Glass", package = "mlbench")

## -----------------------------------------------------------------------------
# 1.1) define the features
# and outcomes
outcome  <- c("Type")
features <- setdiff(x = colnames(Glass), y = outcome)

# 2) split data in training
# and test

# 2.1) set seed for 
# for reproducibility
set.seed(1903)

# 2.2) exttract
# indices with a simple
# 80/20 split
index <- sample(1:nrow(Glass), size = 0.8 * nrow(Glass))

# 1.1) extract training
# data and construct
# as lgb.Dataset
train <- Glass[index,]
dtrain <- lightgbm::lgb.Dataset(
    data  = data.matrix(train[,features]),
    label = train$Type
)
# 1.2) extract test
# data
test <- Glass[-index,]


# 1.2.1) extract actual
# values and constuct
# as.factor for {SLmetrics}
# methods
actual <- as.factor(
    test$Type
)

# 1.2.2) construct as data.matrix
# for predict method
test <- data.matrix(
    test[,features]
)

## ----parameters---------------------------------------------------------------
# 1) define parameters
# across the vignette
parameters <- list(
    objective     = "multiclass",
    num_leaves    = 4L,
    learning_rate = 0.1,
    num_class     = 8
)

## -----------------------------------------------------------------------------
# 1) define the custom
# evaluation metric
eval_fbeta <- function(
    dtrain, 
    preds) {

        # 1) extract values
        actual    <- as.factor(dtrain)
        predicted <- lightgbm::get_field(preds, "label")
        value     <- fbeta(
            actual    = actual,
            predicted = predicted,
            beta      = 2,
            # Use micro-averaging to account
            # for class imbalances
            micro     = TRUE
        )

        # 2) construnct output
        # list
        list(
            name          = "fbeta",
            value         = value,
            higher_better = TRUE 
        )
    
}

## -----------------------------------------------------------------------------
model <- lightgbm::lgb.train(
    params  = parameters,
    data    = dtrain,
    nrounds = 10L,
    eval    = eval_fbeta,
    verbose = -1
)

## ----forecasts----------------------------------------------------------------
# 1) prediction
# from the model
predicted <- as.factor(
    predict(
        model,
        newdata = test,
        type = "class"
    )
)

## ----cmatrix------------------------------------------------------------------
# 1) construct confusion
# matrix
confusion_matrix <- cmatrix(
    actual = actual,
    predicted = predicted
)

# 2) visualize
plot(
    confusion_matrix
)

# 3) summarize
summary(
    confusion_matrix
)

## ----response-----------------------------------------------------------------
# 1) prediction
# from the model
response <- predict(
        model,
        newdata = test
    )


## -----------------------------------------------------------------------------
# 1) calculate the reciever
# operator characteristics
roc <- ROC(
    actual   = actual,
    response = response
)

# 2) print the roc
# object
print(roc)

## -----------------------------------------------------------------------------
# 1) plot roc
# object
plot(roc)

## ----custom thresholds--------------------------------------------------------
# 1) create custom
# thresholds
thresholds <- seq(
    from = 0.9,
    to   = 0.1,
    length.out = 10
)

# 2) pass the custom thresholds
# to the ROC()-function
roc <- ROC(
    actual     = actual,
    response   = response,
    thresholds = thresholds 
)

# 3) print the roc
# object
print(roc)

## ----ROC----------------------------------------------------------------------
# 1) viasualize
# ROC
plot(roc)

## ----summary of ROC-----------------------------------------------------------
# 1) summarise ROC
summary(roc)

