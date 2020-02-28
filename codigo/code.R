data("marketing", package = "datarium")
head(marketing, 4)
formula <-
  lm(sales ~ youtube + facebook + newspaper, data = marketing)


# Generating original sample ----------------------------------------------

original_sample <- function(n = 20L, npar = 1L, lambda = 1, balanced = TRUE, error = "weibull", args) {
  error <- do.call(what = paste0("r", error), args = args)
  error <- (error - mean(error)) / sd(error)
  
  
}

# Constant information ----------------------------------------------------
X <- create_x(formula = formula)
y <- marketing
result_qr <- qr(X, LAPACK = TRUE)
R <- qr.R(result_qr)
Q <- qr.Q(result_qr)
H <- X %*% solve(t(X) %*% X) %*% t(X)
invr <- solve(R)

olse <- function(X, y)
  solve(R) %*%  t(Q) %*% y

residuals <- function(H, y) 
  as.vector((diag(1, nrow = nrow(H), ncol = nrow(H)) - H) %*% y) 
    
hc <- function(y, X, H, hc = 0L, k = 0.7) {
  hc <- as.character(hc)
  r <- residuals(H = H, y = y) 
  h <- diag(H)
  
  switch (hc,
          "0" = {
            omega <- diag(r ^ 2)
            bread <- tcrossprod(invr, Q)
            result <- bread %*% tcrossprod(omega, bread)
            return(result)
          },
          "2" = {
            omega <- diag(r ^ 2 / (1 - h))
            bread <- tcrossprod(invr, Q)
            result <- bread %*% tcrossprod(omega, bread)
            return(result)
          },
          "3" = {
            omega <- diag(r ^ 2 / (1 - h) ^ 2)
            bread <- tcrossprod(invr, Q)
            result <- bread %*% tcrossprod(omega, bread)
            return(result)
          },
          "4" = {
            delta <- pmin(4, h / mean(h))
            omega <- diag(r ^ 2 / (1 - h) ^ delta)
            bread <- tcrossprod(invr, Q)
            result <- bread %*% tcrossprod(omega, bread)
            return(result)
          },
          "5" = {
            alpha <- pmin(h / mean(h), max(4, k * max(h) / mean(h)))
            omega <- diag(r ^ 2 / sqrt((1 - h) ^ alpha))
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


sample_boot <- function(formula){
  
  X <- create_x(formula)
  
  y <-  X %*% formula$coefficients * t_value(n = dim(X)[1L]) * formula$residuals / sqrt(1 - h)
  y
}

microbenchmark::microbenchmark(
    hcci::HC(formula, method = 5),
  hc(y = y, X = X, H = H, hc = 5),
  times = 1e3L
)



  