

> Version 0.3-3 is considered pre-release of {SLmetrics}. We do not
> expect any breaking changes, unless a major bug/issue is reported and
> its nature forces breaking changes.

# Version 0.3-3

## :rocket: Improvements

- **S3 signatures:** All S3-methods now have a generic signature, the
  functions should now be easier to navigate in argument-wise.

## :fire: New features

### :hammer: Metric tools

A new family of `Tools`-functions are introduced with this update. This
addition introduces unexported functions for constructing fast and
memory efficient proprietary metrics. These functions are rewritten
built-in functions from {stats} and family.

- **Covariance Matrix:** A re-written `stats::cov.wt()`, using `Rcpp`.
  Example usage:

``` r
## generate values
actual    <- c(1.2,  0.3, 0.56, 0.11, 1.01)
predicted <- c(0.9, 0.22, 0.76, 0.21, 1.1) 

## generate covariance matrix
SLmetrics:::cov.wt(
    cbind(
        actual,
        predicted
    )
)
#> $cov
#>             actual predicted
#> actual    0.213330  0.169215
#> predicted 0.169215  0.163720
#> 
#> $center
#>    actual predicted 
#>     0.636     0.638 
#> 
#> $n.obs
#> [1] 5
```

## :warning: Breaking changes

- **Logloss:** The argument `pk` has been replaced by `response`.

# Version 0.3-2

## :rocket: Improvements

- **Regression metrics (See PR
  https://github.com/serkor1/SLmetrics/pull/64):** All regression
  metrics have had their back-end optimized and are now 2-10 times
  faster than prior versions.
- **LAPACK/BLAS Support
  (https://github.com/serkor1/SLmetrics/pull/65):** Added LAPACK/BLAS
  support for efficient matrix-operations.
- **OpenMP:** Enabling/disabling OpenMP is now handled on the `R`-side
  and obeys `suppressMessages()`. See below:

``` r
## suppress OpenMP messages
suppressMessages(
  SLmetrics::openmp.off()
)
```

## :fire: New features

- **Available threads:** The available number of threads can be
  retrieved using the `openmp.threads()`. See below:

``` r
## number of available
## threads
SLmetrics::openmp.threads()
#> [1] 16
```

## :bug: Bug-fixes

- **Diagnostic Odds Ratio:** The `dor()` is now returning a single
  `<[numeric]>`-value instead of `k` number of identical
  `<[numeric]>`-values.

## :warning: Breaking Changes

- **OpenMP Interface:** The interface to enabling/disabling OpenMP
  support has been reworked and has a more natural flow. The new
  interface is described below:

``` r
## enable OpenMP
SLmetrics::openmp.on()
#> OpenMP enabled!
```

``` r
## disable OpenMP
SLmetrics::openmp.off()
#> OpenMP disabled!
```

To set the number of threads use the `openmp.threads()` as follows:

``` r
## set number of threads
SLmetrics::openmp.threads(3)
#> Using 3 threads.
```

# Version 0.3-1

## :rocket: Improvements

- **OpenMP Support (PR https://github.com/serkor1/SLmetrics/pull/40):**
  {SLmetrics} now supports parallelization through OpenMP. The OpenMP
  can be utilized as follows:

``` r
# 1) probability distribution
# generator
rand.sum <- function(n){
    x <- sort(runif(n-1))
    c(x,1) - c(0,x)
  }

# 2) generate probability
# matrix
set.seed(1903)
pk <- t(replicate(100,rand.sum(1e3)))

# 3) Enable OpenMP
SLmetrics::setUseOpenMP(TRUE)
system.time(SLmetrics::entropy(pk))

# 3) Disable OpenMP
SLmetrics::setUseOpenMP(FALSE)
system.time(SLmetrics::entropy(pk))
```

- **Entropy with soft labels
  (https://github.com/serkor1/SLmetrics/issues/37):** `entropy()`,
  `cross.entropy()` and `relative.entropy()` have been introduced. These
  functions are heavily inspired by {scipy}. The functions can be used
  as follows:

``` r
# 1) Define actual
# and observed probabilities

# 1.1) actual probabilies
pk <- matrix(
  cbind(1/2, 1/2),
  ncol = 2
)

# 1.2) observed (estimated) probabilites
qk <- matrix(
  cbind(9/10, 1/10), 
  ncol = 2
)

# 2) calculate
# Entropy
cat(
  "Entropy", SLmetrics::entropy(
    pk
  ),
  "Relative Entropy", SLmetrics::relative.entropy(
    pk,
    qk
  ),
  "Cross Entropy", SLmetrics::cross.entropy(
    pk,
    qk
  ),
  sep = "\n"
)
#> Entropy
#> 0.6931472
#> Relative Entropy
#> 0.5108256
#> Cross Entropy
#> 1.203973
```

## :warning: Breaking changes

- **logloss:** The argument `response` have ben renamed to `qk` as in
  the `entropy()`-family to maintain some degree of consistency.
- **entropy.factor():** The function have been deleted and is no more.
  This was mainly due to avoid the documentation from being too large.
  The `logloss()`-function replaces it.

## :bug: Bug-fixes

- **Plot-method in ROC and prROC
  (https://github.com/serkor1/SLmetrics/issues/36):** Fixed a bug in
  `plot.ROC()` and `plot.prROC()` where if `panels = FALSE` additional
  lines would be added to the plot.

# Version 0.3-0

## Improvements

## New Feature

- **Relative Root Mean Squared Error:** The function normalizes the Root
  Mean Squared Error by a factor. There is no official way of
  normalizing it - and in {SLmetrics} the RMSE can be normalized using
  three options; mean-, range- and IQR-normalization. It can be used as
  follows,

``` r
# 1) define values
actual <- rnorm(1e3)
predicted <- actual + rnorm(1e3)

# 2) calculate Relative Root Mean Squared Error
cat(
  "Mean Relative Root Mean Squared Error", SLmetrics::rrmse(
    actual        = actual,
    predicted     = predicted,
    normalization = 0
  ),
  "Range Relative Root Mean Squared Error", SLmetrics::rrmse(
    actual        = actual,
    predicted     = predicted,
    normalization = 1
  ),
  "IQR Relative Root Mean Squared Error", SLmetrics::rrmse(
    actual        = actual,
    predicted     = predicted,
    normalization = 2
  ),
  sep = "\n"
)
#> Mean Relative Root Mean Squared Error
#> 23.57525
#> Range Relative Root Mean Squared Error
#> 0.1413768
#> IQR Relative Root Mean Squared Error
#> 0.6665152
```

- **Log Loss:** Weighted and unweighted Log Loss, with and without
  normalization. The function can be used as follows,

``` r
# Create factors and response probabilities (qk)
actual   <- factor(c("Class A", "Class B", "Class A"))
weights  <- c(0.3,0.9,1) 
response <- matrix(cbind(
    0.2, 0.8,
    0.8, 0.2,
    0.7, 0.3
),nrow = 3, ncol = 2)

cat(
    "Unweighted Log Loss:",
    SLmetrics::logloss(
        actual,
        response
    ),
    "Weighted log Loss:",
    SLmetrics::weighted.logloss(
        actual   = actual,
        response       = response,
        w        = weights
    ),
    sep = "\n"
)
#> Unweighted Log Loss:
#> 0.7297521
#> Weighted log Loss:
#> 0.4668102
```

- **Weighted Receiver Operator Characteristics:** `weighted.ROC()`, the
  function calculates the weighted True Positive and False Positive
  Rates for each threshold.

- **Weighted Precision-Recall Curve:** `weighted.prROC()`, the function
  calculates the weighted Recall and Precision for each threshold.

## Breaking Changes

- **Weighted Confusion Matrix:** The `w`-argument in `cmatrix()` has
  been removed in favor of the more verbose weighted confusion matrix
  call `weighted.cmatrix()`-function. See below,

Prior to version `0.3-0` the weighted confusion matrix were a part of
the `cmatrix()`-function and were called as follows,

``` r
SLmetrics::cmatrix(
    actual    = actual,
    predicted = predicted,
    w         = weights
)
```

This solution, although simple, were inconsistent with the remaining
implementation of weighted metrics in {SLmetrics}. To regain consistency
and simplicity the weighted confusion matrix are now retrieved as
follows,

``` r
# 1) define factors
actual    <- factor(sample(letters[1:3], 100, replace = TRUE))
predicted <- factor(sample(letters[1:3], 100, replace = TRUE))
weights   <- runif(length(actual))

# 2) without weights
SLmetrics::cmatrix(
    actual    = actual,
    predicted = predicted
)
#>    a  b  c
#> a  7 10 11
#> b 15  8  8
#> c  7 23 11

# 2) with weights
SLmetrics::weighted.cmatrix(
    actual    = actual,
    predicted = predicted,
    w         = weights
)
#>           a         b         c
#> a  3.720489  5.048507  5.829939
#> b  7.962346  3.438800  2.198618
#> c  3.473506 10.423952  7.394876
```

## :bug: Bug-fixes

- **Return named vectors:** The classification metrics when
  `micro == NULL` were not returning named vectors. This has been fixed.

# Version 0.2-0

## Improvements

- **documentation:** The documentation has gotten some extra love, and
  now all functions have their formulas embedded, the details section
  have been freed from a general description of \[factor\] creation.
  This will make room for future expansions on the various functions
  where more details are required.

- **weighted classification metrics:** The `cmatrix()`-function now
  accepts the argument `w` which is the sample weights; if passed the
  respective method will return the weighted metric. Below is an example
  using sample weights for the confusion matrix,

``` r
# 1) define factors
actual    <- factor(sample(letters[1:3], 100, replace = TRUE))
predicted <- factor(sample(letters[1:3], 100, replace = TRUE))
weights   <- runif(length(actual))

# 2) without weights
SLmetrics::cmatrix(
    actual    = actual,
    predicted = predicted
)
#>    a  b  c
#> a  5 11  9
#> b 16 13 10
#> c 13 13 10

# 2) with weights
SLmetrics::weighted.cmatrix(
    actual    = actual,
    predicted = predicted,
    w         = weights
)
#>          a        b        c
#> a 3.055945 5.979979 4.421233
#> b 7.673914 5.250443 4.319918
#> c 5.864924 7.383163 4.517976
```

Calculating weighted metrics manually or by using
`foo.cmatrix()`-method,

``` r
# 1) weigthed confusion matrix
# and weighted accuray
confusion_matrix <- SLmetrics::cmatrix(
    actual    = actual,
    predicted = predicted,
    w         = weights
)

# 2) pass into accuracy
# function
SLmetrics::accuracy(
    confusion_matrix
)
#> [1] 0.28

# 3) calculate the weighted
# accuracy manually
SLmetrics::weighted.accuracy(
    actual    = actual,
    predicted = predicted,
    w         = weights
)
#> [1] 0.2645972
```

Please note, however, that it is not possible to pass `cmatrix()`-into
`weighted.accuracy()`,

- **Unit-testing:** All functions are now being tested for edge-cases in
  balanced and imbalanced classification problems, and regression
  problems, individually. This will enable a more robust development
  process and prevent avoidable bugs.

``` r
try(
    SLmetrics::weighted.accuracy(
        confusion_matrix
    )
)
#> Error in UseMethod(generic = "weighted.accuracy", object = ..1) : 
#>   no applicable method for 'weighted.accuracy' applied to an object of class "cmatrix"
```

## :bug: Bug-fixes

- **Floating precision:** Metrics would give different results based on
  the method used. This means that `foo.cmatrix()` and `foo.factor()`
  would produce different results (See Issue
  https://github.com/serkor1/SLmetrics/issues/16). This has been fixed
  by using higher precision `Rcpp::NumericMatrix` instead of
  `Rcpp::IntegerMatrix`.

- **Miscalculation of Confusion Matrix elements:** An error in how `FN`,
  `TN`, `FP` and `TP` were calculated have been fixed. No issue has been
  raised for this bug. This was not something that was caught by the
  unit-tests, as the total samples were too high to spot this error. It
  has, however, been fixed now. This means that all metrics that uses
  these explicitly are now stable, and produces the desired output.

- **Calculation Error in Fowlks Mallows Index:** A bug in the
  calculation of the `fmi()`-function has been fixed. The
  `fmi()`-function now correctly calculates the measure.

- **Calculation Error in Pinball Deviance and Concordance Correlation
  Coefficient:** See issue
  https://github.com/serkor1/SLmetrics/issues/19. Switched to unbiased
  variance calculation in `ccc()`-function. The `pinball()`-function
  were missing a weighted quantile function. The issue is now fixed.

- **Calculation Error in Balanced Accuracy:** See issue
  https://github.com/serkor1/SLmetrics/issues/24. The function now
  correctly adjusts for random chance, and the result matches that of
  {scikit-learn}

- **Calculation Error in F-beta Score:** See issue
  https://github.com/serkor1/SLmetrics/issues/23. The function werent
  respecting `na.rm` and `micro`, this has been fixed accordingly.

- **Calculation Error in Relative Absolute Error:** The function was
  incorrectly calculating means, instead of sums. This has been fixed.

## Breaking changes

- All regression metrics have had `na.rm`- and `w`-arguments removed.
  All weighted regression metrics have a separate function on the
  `weighted.foo()` to increase consistency across all metrics. See
  example below,

``` r
# 1) define regression problem
actual    <- rnorm(n = 1e3)
predicted <- actual + rnorm(n = 1e3)
w         <- runif(n = 1e3)

# 2) unweighted metrics
SLmetrics::rmse(actual, predicted)
#> [1] 1.012171

# 3) weighted metrics
SLmetrics::weighted.rmse(actual, predicted, w = w)
#> [1] 1.016224
```

- The `rrmse()`-function have been removed in favor of the
  `rrse()`-function. This function was incorrectly specified and
  described in the package.

# Version 0.1-1

## General

- **Backend changes:** All pair-wise metrics are moved from {Rcpp} to
  C++, this have reduced execution time by half. All pair-wise metrics
  are now faster.

## Improvements

- **NA-controls:** All pair-wise metrics that doesn’t have a
  `micro`-argument were handling missing values as according to C++ and
  {Rcpp} internals. See
  [Issue](https://github.com/serkor1/SLmetrics/issues/8). Thank you
  @EmilHvitfeldt for pointing this out. This has now been fixed so
  functions uses an `na.rm`-argument to explicitly control for this. See
  below,

``` r
# 1) define factors
actual    <- factor(c("no", "yes"))
predicted <- factor(c(NA, "no"))

# 2) accuracy with na.rm = TRUE
SLmetrics::accuracy(
    actual    = actual,
    predicted = predicted,
    na.rm     = TRUE
)

# 2) accuracy with na.rm = FALSE
SLmetrics::accuracy(
    actual    = actual,
    predicted = predicted,
    na.rm     = FALSE
)
```

## :bug: Bug-fixes

- The `plot.prROC()`- and `plot.ROC()`-functions now adds a line to the
  plot when `panels = FALSE`. See Issue
  https://github.com/serkor1/SLmetrics/issues/9.

``` r
# 1) define actual
# classes
actual <- factor(
  sample(letters[1:2], size = 100, replace = TRUE)
)

# 2) define response
# probabilities
response <- runif(100)

# 3) calculate
# ROC and prROC

# 3.1) ROC
roc <- SLmetrics::ROC(
    actual,
    response
)

# 3.2) prROC
prroc <- SLmetrics::prROC(
    actual,
    response
)

# 4) plot with panels
# FALSE
par(mfrow = c(1,2))
plot(
  roc,
  panels = FALSE
)
```

<img src="meta/NEWS_files/figure-commonmark/unnamed-chunk-18-1.png"
style="width:100.0%" />

``` r

plot(
    prroc,
    panels = FALSE
)
```

<img src="meta/NEWS_files/figure-commonmark/unnamed-chunk-18-2.png"
style="width:100.0%" />

# Version 0.1-0

## General

- {SLmetrics} is a collection of Machine Learning performance evaluation
  functions for supervised learning. Visit the online documentation on
  [GitHub Pages](https://serkor1.github.io/SLmetrics/).

## Examples

### Supervised classification metrics

``` r
# 1) actual classes
print(
    actual <- factor(
        sample(letters[1:3], size = 10, replace = TRUE)
    )
)
#>  [1] c b a b c b c b c c
#> Levels: a b c

# 2) predicted classes
print(
    predicted <- factor(
        sample(letters[1:3], size = 10, replace = TRUE)
    )
)
#>  [1] c c c b b c b a a a
#> Levels: a b c
```

``` r
# 1) calculate confusion
# matrix and summarise
# it
summary(
    confusion_matrix <- SLmetrics::cmatrix(
        actual    = actual,
        predicted = predicted
    )
)
#> Confusion Matrix (3 x 3) 
#> ================================================================================
#>   a b c
#> a 0 0 1
#> b 1 1 2
#> c 2 2 1
#> ================================================================================
#> Overall Statistics (micro average)
#>  - Accuracy:          0.20
#>  - Balanced Accuracy: 0.15
#>  - Sensitivity:       0.20
#>  - Specificity:       0.60
#>  - Precision:         0.20

# 2) calculate false positive
# rate using micro average
SLmetrics::fpr(
    confusion_matrix
)
#>         a         b         c 
#> 0.3333333 0.3333333 0.6000000
```

### Supervised regression metrics

``` r
# 1) actual values
actual <- rnorm(n = 100)

# 2) predicted values
predicted <- actual + rnorm(n = 100)
```

``` r
# 1) calculate
# huber loss
SLmetrics::huberloss(
    actual    = actual,
    predicted = predicted
)
#> [1] 0.4261266
```
