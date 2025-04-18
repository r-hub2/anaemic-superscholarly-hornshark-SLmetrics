# script: specificity
# author: Serkan Korkmaz, serkor1@duck.com
# date: 2025-19-01
# objective: Generate method
# script start;

#' @aliases logloss
#' @inherit accuracy
#' 
#' @title Log Loss
#'
#' @description
#' The [logloss()] function computes the **Log Loss** between observed classes (as a <[factor]>) and their predicted probability distributions (a <[numeric]> matrix). The [weighted.logloss()] function is the weighted version, applying observation-specific weights.
#' 
#' @usage
#' ## Generic S3 method
#' logloss(
#'  actual,
#'  response,
#'  normalize = TRUE,
#'  ...
#' )
#' 
#' @param response A \eqn{n \times k} <[numeric]>-matrix of predicted probabilities.
#'   The \eqn{i}-th row should sum to 1 (i.e., a valid probability distribution
#'   over the \eqn{k} classes). The first column corresponds to the first factor
#'   level in \code{actual}, the second column to the second factor level, and so on.
#' @param normalize A <[logical]>-value (default: [TRUE]). If [TRUE],
#'   the mean cross-entropy across all observations is returned; otherwise, the
#'   sum of cross-entropies is returned.
#'
#' @section Definition:
#' 
#' \deqn{H(p, response) = -\sum_{i} \sum_{j} y_{ij} \log_2(response_{ij})}{
#' H(p, response) = -\sum_{i} \sum_{j} y_ij log(response_ij)}
#' where:
#' \itemize{
#'   \item \eqn{y_{ij}} is the `actual`-values, where \eqn{y_{ij}} = 1 if the `i`-th sample belongs to class `j`, and 0 otherwise.
#'   \item \eqn{response_{ij}} is the estimated probability for the `i`-th sample belonging to class `j`.
#' }
#' 
#' @example man/examples/scr_LogLoss.R
#'
#' @family Classification
#' @family Supervised Learning
#'

#' @export
logloss <- function(
  actual, 
  response,
  normalize = TRUE, 
  ...) {
  UseMethod(
    generic = "logloss"
  )
}

#' @rdname logloss
#' @usage
#' ## Generic S3 method
#' weighted.logloss(
#'  actual,
#'  response,
#'  w,
#'  normalize = TRUE,
#'  ...
#' )
#' @export
weighted.logloss <- function(
  actual, 
  response,
  w,
  normalize = TRUE, 
  ...) {
  UseMethod(
    generic = "weighted.logloss"
  )
}

# script end;
