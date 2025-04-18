# script: Reciever Operator Characteristics
# date: 2024-10-25
# author: Serkan Korkmaz, serkor1@duck.com
# objective: Generate Methods
# script start;

#' @inherit specificity
#'
#' @title Reciever Operator Characteristics
#'
#' @description
#' The [ROC()]-function computes the [tpr()] and [fpr()] at thresholds provided by the \eqn{response}- or \eqn{thresholds}-vector. The function
#' constructs a [data.frame()] grouped by \eqn{k}-classes where each class is treated as a binary classification problem.
#' 
#' @usage
#' ## Generic S3 method
#' ROC(
#'  actual,
#'  response,
#'  thresholds,
#'  ...
#' )
#' 
#' @param response A <[numeric]>-vector of [length] \eqn{n}. The estimated response probabilities.
#' @param thresholds An optional <[numeric]>-vector of non-zero [length] (default: [NULL]).
#' @param ... Arguments passed into other methods.
#'
#' @returns A [data.frame] on the following form,
#'
#' \item{threshold}{<[numeric]> Thresholds used to determine [tpr()] and [fpr()]}
#' \item{level}{<[character]> The level of the actual <[factor]>}
#' \item{label}{<[character]> The levels of the actual <[factor]>}
#' \item{fpr}{<[numeric]> The false positive rate}
#' \item{tpr}{<[numeric]> The true positve rate}
#'
#' @example man/examples/scr_RecieverOperatorCurve.R
#'
#' @family Classification
#' @family Supervised Learning
#'
#' @export
ROC <- function(
  actual,
  response, 
  thresholds, 
  ...) {
  UseMethod(
    generic = "ROC"
  )
}

#' @rdname ROC
#' @usage
#' ## Generic S3 method
#' weighted.ROC(
#'  actual,
#'  response,
#'  w,
#'  thresholds,
#'  ...
#' )
#' @export
weighted.ROC <- function(
  actual,
  response,
  w,
  thresholds,
  ...) {
  UseMethod(
    generic = "weighted.ROC"
  )
}

#' @export
print.ROC <- function(x, ...) {

  print.data.frame(
    x,
    ...,
    digits = 3,
    max = sum(
      rep(
        10,
        ncol(x)
      )
    )
  )

}

#' @export
summary.ROC <- function(
  object,
  ...) {
  
  # 1) calculate area 
  # under the curve

  # 1.1) extract list
  # of labels
  x_list <- split(
    x = object,
    f = object$label
  )

  # 1.2) calculate AUC
  # for each label
  metric <- vapply(
    x_list, 
    function(x) {
      auc(
        y = x$tpr,
        x = x$fpr
      )
    }, 
    FUN.VALUE = numeric(1),
    USE.NAMES = TRUE
  )

  names(metric) <- names(x_list)
  
  structure(
    .Data = {
      list(
        auc = metric
      )
    },
    class = "summary.ROC"
  )

}

#' @export
print.summary.ROC <- function(
  x, 
  ...) {

  cat("Reciever Operator Characteristics", "\n")
  full_line()
  cat(
    "AUC",
    paste0(" - ", names(x$auc),": " , round(x$auc, 3)),
    sep = "\n"
  )

  invisible(x)

}


#' @export
plot.ROC <- function(
    x,
    panels = TRUE,
    ...) {
  
    # 0) exract the finite
    # data.frame
    x <- x[is.finite(x$threshold), ]

    # 1) Plot options
    #
    # All common options for the
    # plot goes her
    pformula <- tpr ~ fpr
    groups   <- x$label
    xlab     <- "False Positive Rate (FPR)"
    ylab     <- "True Positive Rate (TPR)"
    main     <- "Reciever Operator Characteristics"

    # 1.1) conditional plotting
    # statements
    if (panels) {

      # 1.2) grouped by
      # label.
      pformula <- tpr ~ fpr | factor(label, labels = unique(label))

      # 1.3) disable grouping
      # if panelwise
      groups  <- NULL

    }


    roc_plot(
      formula  = pformula,
      groups   = groups,
      xlab     = xlab,
      ylab     = ylab,
      main     = main,
      DT       = x,
      add_poly = panels,
      ...  
    )
  
}

# script end;
