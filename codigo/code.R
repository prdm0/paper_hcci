# y <- c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
# x <- c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
# result <- lm(ctl ~ trt)
# X <- cbind(1, trt)
# matrix_qr <- qr(X)
# R <- qr.R(matrix_qr)
# Q <- qr.Q(matrix_qr)
# as.vector(solve(R)%*%t(Q)%*%y) # Estimadores.

data("marketing", package = "datarium")
head(marketing, 4)
model <- lm(sales ~ youtube + facebook + newspaper, data = marketing)
summary(model)

hc <- function(formula, ...) {
  if (class(formula) == "lm")
    stop("\'formula\' must be an object of the \'lm\' class.")
  
  
}