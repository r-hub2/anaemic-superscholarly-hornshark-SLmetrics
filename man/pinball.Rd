% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R, R/S3_PinballLoss.R
\name{pinball.numeric}
\alias{pinball.numeric}
\alias{weighted.pinball.numeric}
\alias{pinball}
\alias{weighted.pinball}
\title{Pinball Loss}
\usage{
\method{pinball}{numeric}(actual, predicted, alpha = 0.5, deviance = FALSE, ...)

\method{weighted.pinball}{numeric}(actual, predicted, w, alpha = 0.5, deviance = FALSE, ...)

## Generic S3 method
pinball(
 actual,
 predicted,
 alpha    = 0.5,
 deviance = FALSE,
 ...
)

## Generic S3 method
weighted.pinball(
 actual,
 predicted,
 w,
 alpha    = 0.5,
 deviance = FALSE,
 ...
)
}
\arguments{
\item{actual}{A <\link{numeric}>-vector of \link{length} \eqn{n}. The observed (continuous) response variable.}

\item{predicted}{A <\link{numeric}>-vector of \link{length} \eqn{n}. The estimated (continuous) response variable.}

\item{alpha}{A <\link{numeric}>-value of \link{length} \eqn{1} (default: \eqn{0.5}). The slope of the pinball loss function.}

\item{deviance}{A <\link{logical}>-value of \link{length} 1 (default: \link{FALSE}). If \link{TRUE} the function returns the \eqn{D^2} loss.}

\item{...}{Arguments passed into other methods.}

\item{w}{A <\link{numeric}>-vector of \link{length} \eqn{n}. The weight assigned to each observation in the data.}
}
\value{
A <\link{numeric}> vector of \link{length} 1.
}
\description{
The \code{\link[=pinball]{pinball()}}-function computes the \href{https://en.wikipedia.org/wiki/Quantile_regression}{pinball loss} between
the observed and predicted <\link{numeric}> vectors. The \code{\link[=weighted.pinball]{weighted.pinball()}} function computes the weighted Pinball Loss.
}
\section{Definition}{


The metric is calculated as,

\deqn{\text{PinballLoss}_{\text{unweighted}} = \frac{1}{n} \sum_{i=1}^{n} \left[ \alpha \cdot \max(0, y_i - \hat{y}_i) - (1 - \alpha) \cdot \max(0, \hat{y}_i - y_i) \right]}

where \eqn{y_i} is the actual value, \eqn{\hat{y}_i} is the predicted value and \eqn{\alpha} is the quantile level.
}

\examples{
# 1) fit a linear
# regression
model <- lm(
  mpg ~ .,
  data = mtcars
)

# 1.1) define actual
# and predicted values
# to measure performance
actual    <- mtcars$mpg
predicted <- fitted(model)

# 2) evaluate in-sample model
# performance using Pinball Loss
cat(
  "Pinball Loss", pinball(
    actual    = actual,
    predicted = predicted,
  ),
  "Pinball Loss (weighted)", weighted.pinball(
    actual    = actual,
    predicted = predicted,
    w         = mtcars$mpg/mean(mtcars$mpg)
  ),
  sep = "\n"
)
}
\seealso{
Other Regression: 
\code{\link{ccc.numeric}()},
\code{\link{huberloss.numeric}()},
\code{\link{mae.numeric}()},
\code{\link{mape.numeric}()},
\code{\link{mpe.numeric}()},
\code{\link{mse.numeric}()},
\code{\link{rae.numeric}()},
\code{\link{rmse.numeric}()},
\code{\link{rmsle.numeric}()},
\code{\link{rrmse.numeric}()},
\code{\link{rrse.numeric}()},
\code{\link{rsq.numeric}()},
\code{\link{smape.numeric}()}

Other Supervised Learning: 
\code{\link{ROC.factor}()},
\code{\link{accuracy.factor}()},
\code{\link{baccuracy.factor}()},
\code{\link{ccc.numeric}()},
\code{\link{ckappa.factor}()},
\code{\link{cmatrix.factor}()},
\code{\link{dor.factor}()},
\code{\link{entropy.matrix}()},
\code{\link{fbeta.factor}()},
\code{\link{fdr.factor}()},
\code{\link{fer.factor}()},
\code{\link{fpr.factor}()},
\code{\link{huberloss.numeric}()},
\code{\link{jaccard.factor}()},
\code{\link{logloss.factor}()},
\code{\link{mae.numeric}()},
\code{\link{mape.numeric}()},
\code{\link{mcc.factor}()},
\code{\link{mpe.numeric}()},
\code{\link{mse.numeric}()},
\code{\link{nlr.factor}()},
\code{\link{npv.factor}()},
\code{\link{plr.factor}()},
\code{\link{prROC.factor}()},
\code{\link{precision.factor}()},
\code{\link{rae.numeric}()},
\code{\link{recall.factor}()},
\code{\link{rmse.numeric}()},
\code{\link{rmsle.numeric}()},
\code{\link{rrmse.numeric}()},
\code{\link{rrse.numeric}()},
\code{\link{rsq.numeric}()},
\code{\link{smape.numeric}()},
\code{\link{specificity.factor}()},
\code{\link{zerooneloss.factor}()}
}
\concept{Regression}
\concept{Supervised Learning}
