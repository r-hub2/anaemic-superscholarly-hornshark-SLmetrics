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
data("BostonHousing", package = "mlbench")

## -----------------------------------------------------------------------------
# 1.1) define the features
# and outcomes
outcome  <- c("medv")
features <- setdiff(
    x = colnames(BostonHousing), 
    y = outcome
    )

# 2) split data in training
# and test

# 2.1) set seed for 
# for reproducibility
set.seed(1903)

# 2.2) exttract
# indices with a simple
# 90/10 split
index <- sample(1:nrow(BostonHousing), size = 0.9 * nrow(BostonHousing))

# 1.1) extract training
# data and construct
# as lgb.Dataset
train <- BostonHousing[index,]

# 1.1.1) convert
# to DMatrix
dtrain <- xgboost::xgb.DMatrix(
    data = data.matrix(train[, features]),
    label = data.matrix(train[, outcome])
)


# 1.2) extract test
# data
test <- BostonHousing[-index,]

# 1.2.1) convert to DMatrix
dtest <-  xgboost::xgb.DMatrix(
    data = data.matrix(test[, features]),
    label = data.matrix(test[, outcome])
)

# 1.2.2) extract actual
# outcome
actual <- test$medv

## ----parameters---------------------------------------------------------------
# 1) define parameters
# across the vignette
parameters <- list(
    max_depth = 10, 
    eta = 0.1
)

## -----------------------------------------------------------------------------
# 1) define the custom
# evaluation metric
eval_rrse <- function(
    preds, 
    dtrain) {

        # 1) extract values
        actual    <- xgboost::getinfo(dtrain, "label")
        predicted <- preds
        value     <- rrse(
            actual    = actual,
            predicted = predicted
        )

        # 2) construnct output
        # list
        list(
            metric = "RRMSE",
            value  = value
        )
    
}

## -----------------------------------------------------------------------------
# 1) model training
model <- xgboost::xgb.train(
    params  = parameters,
    data    = dtrain,
    nrounds = 10L,
    verbose = 0,
    feval   = eval_rrse,
    watchlist = list(
        train = dtrain,
        test  = dtest
    ),
    maximize = FALSE
)

## -----------------------------------------------------------------------------
# 1) out of sample
# prediction
predicted <- predict(
    model,
    newdata = dtest
)

## -----------------------------------------------------------------------------
# 1) summarize all
# performance measures 
# in data.frame
data.frame(
    RRMSE  = rrse(actual, predicted), 
    RMSE   = rmse(actual, predicted),
    CCC    = ccc(actual, predicted)
)

