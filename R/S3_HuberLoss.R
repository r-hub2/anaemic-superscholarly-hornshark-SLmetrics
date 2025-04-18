# script: Huber Loss
# date: 2024-10-09
# author: Serkan Korkmaz, serkor1@duck.com
# objective: Generate Methods
# script start;

#' @title Huber Loss
#'
#' @description
#' The [huberloss()]-function computes the simple and weighted [huber loss](https://en.wikipedia.org/wiki/Huber_loss) between
#' the predicted and observed <[numeric]> vectors. The [weighted.huberloss()] function computes the weighted Huber Loss.
#'
#' @usage
#' ## Generic S3 method
#' huberloss(
#'  actual,
#'  predicted,
#'  delta = 1,
#'  ...
#' )
#' 
#' @param actual A <[numeric]>-vector of [length] \eqn{n}. The observed (continuous) response variable.
#' @param predicted A <[numeric]>-vector of [length] \eqn{n}. The estimated (continuous) response variable.
#' @param w A <[numeric]>-vector of [length] \eqn{n}. The weight assigned to each observation in the data.
#' @param delta A <[numeric]>-vector of [length] \eqn{1} (default: \eqn{1}). The threshold value for switch between functions (see calculation).
#' @param ... Arguments passed into other methods.
#'
#' @section Definition:
#'
#' The metric is calculated as follows,
#'
#' \deqn{
#'  \frac{1}{2} (y - \upsilon)^2 ~for~ |y - \upsilon| \leq \delta
#' }
#'
#' and
#'
#' \deqn{
#'   \delta |y-\upsilon|-\frac{1}{2} \delta^2 ~for~ \text{otherwise}
#' }
#'
#' where \eqn{y} and \eqn{\upsilon} are the `actual` and `predicted` values respectively. If `w` is not [NULL], then all values
#' are aggregated using the weights.
#'
#' @returns A <[numeric]> vector of [length] 1.
#'
#' @example man/examples/scr_HuberLoss.R
#'
#' @family Regression
#' @family Supervised Learning
#' 
#' @export
huberloss <- function(
  actual, 
  predicted, 
  delta = 1,
   ...) {
  UseMethod(
    generic = "huberloss"
  )
}

#' @rdname huberloss
#' @usage
#' ## Generic S3 method
#' weighted.huberloss(
#'  actual,
#'  predicted,
#'  w,
#'  delta = 1,
#'  ...
#' )
#' @export
weighted.huberloss <- function(
  actual, 
  predicted, 
  w, 
  delta = 1,
   ...) {
  UseMethod(
    generic = "weighted.huberloss"
  )
}

# script end;
