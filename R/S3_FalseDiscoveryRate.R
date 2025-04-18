# script: False Discovery Rate
# date: 2024-10-07
# author: Serkan Korkmaz, serkor1@duck.com
# objective: Generate Methods
# script start;

#' @inherit specificity
#' 
#' @title false discovery rate
#'
#' @description
#' A generic function for the [False Discovery Rate](https://en.wikipedia.org/wiki/False_discovery_rate). Use [weighted.fdr()] for the weighted [False Discovery Rate](https://en.wikipedia.org/wiki/False_discovery_rate).
#' 
#' @usage 
#' ## Generic S3 method
#' fdr(
#'  ...,
#'  micro = NULL,
#'  na.rm = TRUE
#' )
#' 
#' @section Definition:
#' Let \eqn{\hat{\alpha} \in [0, 1]} be the proportion of false positives among the preditced positives. The false discovery rate of the classifier is calculated as,
#'
#' \deqn{
#'   \hat{\alpha} = \frac{\#FP_k}{\#TP_k+\#FP_k}
#' }
#' 
#' Where:
#' 
#' - \eqn{\#TP_k} is the number of true positives, and
#' - \eqn{\#FP_k} is the number of false positives
#'
#' @example man/examples/scr_FalseDiscoveryRate.R
#' 
#' @family Classification
#' @family Supervised Learning
#' 
#' @export
fdr <- function(
  ...,
  micro = NULL, 
  na.rm = TRUE) {
  UseMethod(
    generic = "fdr",
    object   = ..1
  )
}

#' @rdname fdr
#' @usage 
#' ## Generic S3 method
#' weighted.fdr(
#'  ...,
#'  w,
#'  micro = NULL,
#'  na.rm = TRUE
#' )
#' @export
weighted.fdr <- function(
  ..., 
  w, 
  micro = NULL, 
  na.rm = TRUE) {
  UseMethod(
    generic = "weighted.fdr",
    object   = ..1
  )
}

# script end;
