# Simulations, Codes, and Other Materials

### General Information

Supplementary materials for the paper **Theory and Computational Tool for Estimating Confidence Intervals of Heteroscedastic Linear Regression Model Parameters of Unknown Form Using Double Bootstrap** to be published in [**Communications in Statistics – Theory and Methods**](https://www.tandfonline.com/journals/lsta20)

**Authors:**

1. Prof. Dr. Pedro Rafael Diniz Marinho, see the [website](https://prdm0.rbind.io/);
2. Prof. Dr. Francisco Cribari Neto, see the [website](https://www.cribari.com.br/);
3. Prof. Dra. Vera Tomazella, see the [website](https://www.servidores.ufscar.br/vera/).

### Abstract

In several applications where the true regression can be well estimated by a linear regression model, the use of linear regression models is quite common, characterized by their simplicity in theoretical understanding, implementation, and application. One of the assumptions made is that the error variations are constant across all observations. This assumption, known as homoscedasticity, is often violated in practice. A commonly used strategy is to estimate the regression parameters by ordinary least squares and to calculate standard errors that provide asymptotically valid inference under both homoscedasticity and heteroscedasticity of an unknown form. However, since errors are heteroscedastic, and we have no knowledge of their distribution, there are difficulties in obtaining interval estimates and conducting hypothesis tests for the parameters that index the linear regression model. One of the methods used to overcome this problem is the bootstrap method. 

In this context, this work proposes two bootstrap algorithms, double percentile bootstrap and double bootstrap-$t$, to obtain interval estimates in linear regression models with heteroscedasticity of unknown form. Both algorithms use two levels of bootstrapping and employ consistent estimators of the covariance structure of the $\beta$ estimators obtained by the ordinary least squares method. Additionally, a computational library is provided, using the R language, which will allow easy use of the algorithms in applied problems. The \href{https://CRAN.R-project.org/package=hcci}{{\bf hcci}} package can calculate consistent estimates of the covariance matrix of the parameters of linear regression models with heteroscedasticity of unknown form. Our findings regarding bootstrap consistency are exemplified with real-world data.

Repository Materials

1. **thesis.pdf** file containing the simulations conducted in the thesis by the author Pedro Rafael Diniz Marinho. Details about the simulations can be found in Chapter 4, pages 57 to 93;

2. Directory **codes** containing the codes used in the simulations.