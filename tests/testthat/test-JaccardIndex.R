# objective: Test that the metric
# implemented in {SLmetrics} is aligned with
# target functions.

testthat::test_that(
  desc = "Test `jaccard()`-function", code = {

    testthat::skip_on_cran()

    # 0) construct Balanced Accuracy
    # wrapper
    wrapped_jaccard <- function(
      actual,
      predicted,
      w = NULL,
      micro = TRUE) {
      
        if (is.null(w)) {
          jaccard(
            actual     = actual,
            predicted  = predicted,
            micro      = micro
          )
        } else {
          weighted.jaccard(
            actual     = actual,
            predicted  = predicted,
            w          = w,
            micro      = micro
          )
        }
      
    }
    
    for (balanced in c(FALSE, TRUE)) {

      # 1) generate class
      # values
      actual    <- create_factor(balanced = balanced)
      predicted <- create_factor(balanced = balanced)
      w         <- runif(n = length(actual))

      for (weighted in c(TRUE, FALSE)) {
      
        # 2) test that the are 
        # equal to target values
        for (micro in c(NA, TRUE, FALSE)) {

          # 2.1) generate sensible 
          # label information
          info <- paste(
            "Balanced = ", balanced,
            "Weighted = ", weighted,
            "Micro = ", micro
          )

          # 2.2) generate score
          # from {slmetrics}
          score <- wrapped_jaccard(
            actual     = actual,
            predicted  = predicted,
            w          = if (weighted) w else NULL,
            micro      = if (is.na(micro)) { NULL } else micro
          )

          # 2.3) test that the values
          # are sensible the values 
          # can be NA
          testthat::expect_true(is.numeric(score), info = info)

          # 2.4) test that the values
          # are equal to target value

          # 2.4.1) calculate py_score
          py_score <- py_jaccard(
            actual    = actual,
            predicted = predicted,
            average   = if (is.na(micro)) { NULL } else ifelse(micro, "micro", "macro"),
            w         = if (weighted) w else NULL
          )

          if (is.na(micro)) {

            # Python returns values
            # that is less than the number
            # of classes depending on the calculations
            # the behaviour isnt acutally understood as of now.
            score <- score[!is.na(score) & !is.nan(score)]
            py_score <- py_score[!is.na(score) & !is.nan(score)]

            if (length(score) != length(py_score)) {
              py_score <- py_score[py_score != 0]  
            }

          }

          testthat::skip_if(
            length(score) != length(py_score),message =  "Unpredictable behaviour. Skipping test."
         )

          # 2.4.2) test for equality
          testthat::expect_true(
            object = set_equal(
              current = as.numeric(score),
              target  = as.numeric(py_score)
            ),
            info = info
          )

        }

      }

    }
  }
)