data("marketing", package = "datarium")
head(marketing, 4)
formula <-
  lm(sales ~ youtube + facebook + newspaper, data = marketing)
summary(formula)

create_x <- function(formula) {
  data <- formula$model
  as.matrix(cbind(intercept = 1, data[,-1]))
}

hc <- function(formula, hc = 0L, k = 0.7) {
  if (class(formula) != "lm")
    stop("\'formula\' must be an object of the \'lm\' class.")
  
  X <- create_x(formula = formula)
  
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

t_value <- function(n = 1L, normal = FALSE) {
  if (!normal) {
    sample(x = c(-1, 1), size = n, replace = TRUE, prob = c(0.5, 0.5))
  } else {
    rnorm(n = n, mean = 0, sd = 1)
  }
}


sample_boot <- function(formula, X, H, error){
  
    
  X %*% formula$coefficients
}

microbenchmark::microbenchmark(
  hcci::HC(formula, method = 5),
  hc(formula = formula, hc = 5),
  times = 1e3L
)



  