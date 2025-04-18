% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R,
%   R/S3_MeanAbsolutePercentageError.R
\name{mape.numeric}
\alias{mape.numeric}
\alias{weighted.mape.numeric}
\alias{mape}
\alias{weighted.mape}
\title{Mean Absolute Percentage Error}
\usage{
\method{mape}{numeric}(actual, predicted, ...)

\method{weighted.mape}{numeric}(actual, predicted, w, ...)

## Generic S3 method
mape(
 actual,
 predicted,
 ...
)

## Generic S3 method
weighted.mape(
 actual,
 predicted,
 w,
 ...
)
}
\arguments{
\item{actual}{A <\link{numeric}>-vector of \link{length} \eqn{n}. The observed (continuous) response variable.}

\item{predicted}{A <\link{numeric}>-vector of \link{length} \eqn{n}. The estimated (continuous) response variable.}

\item{...}{Arguments passed into other methods.}

\item{w}{A <\link{numeric}>-vector of \link{length} \eqn{n}. The weight assigned to each observation in the data.}
}
\value{
A <\link{numeric}> vector of \link{length} 1.
}
\description{
The \code{\link[=mape]{mape()}}-function computes the \href{https://en.wikipedia.org/wiki/Mean_absolute_percentage_error}{mean absolute percentage error} between
the observed and predicted <\link{numeric}> vectors. The \code{\link[=weighted.mape]{weighted.mape()}} function computes the weighted mean absolute percentage error.
}
\section{Definition}{


The metric is calculated as,

\deqn{
  \frac{1}{n} \sum_i^n \frac{|y_i - \upsilon_i|}{|y_i|}
}
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
# performance using Mean Absolute Percentage Error (MAPE)
cat(
  "Mean Absolute Percentage Error", mape(
    actual    = actual,
    predicted = predicted,
  ),
  "Mean Absolute Percentage Error (weighted)", weighted.mape(
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
\code{\link{mpe.numeric}()},
\code{\link{mse.numeric}()},
\code{\link{pinball.numeric}()},
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
\code{\link{mcc.factor}()},
\code{\link{mpe.numeric}()},
\code{\link{mse.numeric}()},
\code{\link{nlr.factor}()},
\code{\link{npv.factor}()},
\code{\link{pinball.numeric}()},
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
