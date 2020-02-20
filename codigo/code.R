# y <- c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
# x <- c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
# result <- lm(ctl ~ trt)
# X <- cbind(1, trt)
# matrix_qr <- qr(X)
# R <- qr.R(matrix_qr)
# Q <- qr.Q(matrix_qr)
# as.vector(solve(R)%*%t(Q)%*%y) # Estimadores.

library(data.table)

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
  #y <- as.matrix(data[, 1])
  
  
  result_qr <- qr(X, LAPACK = TRUE)
  
  R <- qr.R(result_qr)
  Q <- qr.Q(result_qr)
  
  invr <- solve(R)
  
  hc <- as.character(hc)
  switch (hc,
          "0" = {
            omega <- diag(formula$residuals ^ 2)
            bread <- tcrossprod(invr, Q)
            result <- bread %*% tcrossprod(omega, bread)
            return(result)
          },
          "2" = {
            omega <- diag(formula$residuals ^ 2 / (1 - hatvalues(formula)))
            bread <- tcrossprod(invr, Q)
            result <- bread %*% tcrossprod(omega, bread)
            return(result)
          },
          "3" = {
            omega <- diag(formula$residuals ^ 2 / (1 - hatvalues(formula)) ^ 2)
            bread <- tcrossprod(invr, Q)
            result <- bread %*% tcrossprod(omega, bread)
            return(result)
          },
          "4" = {
            delta <- pmin(4, h/mean(h))
            omega <- diag(formula$residuals ^ 2 / (1 - hatvalues(formula)) ^ delta)
            bread <- tcrossprod(invr, Q)
            result <- bread %*% tcrossprod(omega, bread)
            return(result)
          },
          "5" = {
            alpha <- pmin(h/mean(h), max(4, k * max(h)/mean(h)))
            omega <- diag(formula$residuals ^ 2 / sqrt((1 - hatvalues(formula)) ^ alpha))
            bread <- tcrossprod(invr, Q)
            result <- bread %*% tcrossprod(omega, bread)
            return(result)
          }
  )
}

microbenchmark::microbenchmark(
  hcci::HC(formula, method = 0),
  hc(formula = formula, hc = 5),
  times = 1e3L
)



  