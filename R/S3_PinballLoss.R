# script: Pinball Loss
# date: 2024-10-13
# author: Serkan Korkmaz, serkor1@duck.com
# objective: Generate Methods
# script start;

#' @inherit huberloss
#' 
#' @title Pinball Loss
#'
#' @description
#' The [pinball()]-function computes the [pinball loss](https://en.wikipedia.org/wiki/Quantile_regression) between
#' the observed and predicted <[numeric]> vectors. The [weighted.pinball()] function computes the weighted Pinball Loss.
#'
#' @usage
#' ## Generic S3 method
#' pinball(
#'  actual,
#'  predicted,
#'  alpha    = 0.5,
#'  deviance = FALSE,
#'  ...
#' )
#' 
#' @param alpha A <[numeric]>-value of [length] \eqn{1} (default: \eqn{0.5}). The slope of the pinball loss function.
#' @param deviance A <[logical]>-value of [length] 1 (default: [FALSE]). If [TRUE] the function returns the \eqn{D^2} loss.
#' 
#' @section Definition:
#' 
#' The metric is calculated as,
#' 
#' \deqn{\text{PinballLoss}_{\text{unweighted}} = \frac{1}{n} \sum_{i=1}^{n} \left[ \alpha \cdot \max(0, y_i - \hat{y}_i) - (1 - \alpha) \cdot \max(0, \hat{y}_i - y_i) \right]}
#' 
#' where \eqn{y_i} is the actual value, \eqn{\hat{y}_i} is the predicted value and \eqn{\alpha} is the quantile level.
#'
#' @example man/examples/scr_PinballLoss.R
#' 
#' @family Regression
#' @family Supervised Learning
#' 
#' @export
pinball <- function(
  actual,
  predicted,
  alpha    = 0.5,
  deviance = FALSE,
  ...) {
  UseMethod(
    generic = "pinball"
  )
}

#' @rdname pinball
#' @usage
#' ## Generic S3 method
#' weighted.pinball(
#'  actual,
#'  predicted,
#'  w,
#'  alpha    = 0.5,
#'  deviance = FALSE,
#'  ...
#' )
#' @export
weighted.pinball <- function(
  actual,
  predicted,
  w,
  alpha    = 0.5,
  deviance = FALSE,
  ...) {
  UseMethod(
    generic = "weighted.pinball"
  )
}

# script end;
