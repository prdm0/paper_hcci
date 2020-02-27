data("marketing", package = "datarium")
head(marketing, 4)
formula <-
  lm(sales ~ youtube + facebook + newspaper, data = marketing)
summary(formula)



hc <- function(formula, hc = 0L, k = 0.7) {
  if (class(formula) != "lm")
    stop("\'formula\' must be an object of the \'lm\' class.")
  
  data <- formula$model
  
  X <- as.matrix(cbind(intercept = 1, data[,-1]))
  
  result_qr <- qr(X, LAPACK = TRUE)
  
  R <- qr.R(result_qr)
  Q <- qr.Q(result_qr)
  
  invr <- solve(R)
  
  h <- hatvalues(formula)
  
  hc <- as.character(hc)
  switch (hc,
          "0" = {
            omega <- diag(formula$residuals ^ 2)
            bread <- tcrossprod(invr, Q)
            result <- bread %*% tcrossprod(omega, bread)
            return(result)
          },
          "2" = {
            omega <- diag(formula$residuals ^ 2 / (1 - h))
            bread <- tcrossprod(invr, Q)
            result <- bread %*% tcrossprod(omega, bread)
            return(result)
          },
          "3" = {
            omega <- diag(formula$residuals ^ 2 / (1 - h) ^ 2)
            bread <- tcrossprod(invr, Q)
            result <- bread %*% tcrossprod(omega, bread)
            return(result)
          },
          "4" = {
            delta <- pmin(4, h / mean(h))
            omega <- diag(formula$residuals ^ 2 / (1 - h) ^ delta)
            bread <- tcrossprod(invr, Q)
            result <- bread %*% tcrossprod(omega, bread)
            return(result)
          },
          "5" = {
            alpha <- pmin(h / mean(h), max(4, k * max(h) / mean(h)))
            omega <- diag(formula$residuals ^ 2 / sqrt((1 - h) ^ alpha))
            bread <- tcrossprod(invr, Q)
            result <- bread %*% tcrossprod(omega, bread)
            return(result)
          }
  )
}

microbenchmark::microbenchmark(
  hcci::HC(formula, method = 5),
  hc(formula = formula, hc = 5),
  times = 1e3L
)



  