###   APLICACAO PARA OS DADOS DO ALASKA  ###
# Orientando: Pedro Rafael Diniz Marinho
# Orientador: Francisco Cribari Neto
###########################################
rm(list=ls(all=TRUE))
library(lmtest)


setwd("~/Downloads/")

dados = read.table(file="http://www.de.ufpe.br/~cribari/educacao.dat", header=TRUE)


dados = as.data.frame(dados[-50,],ncol=3)
#dados = dados[-48,]
#dados = dados[-2,]
y = dados$Expenditure # Variável gastos
x = dados$Income/10000 # Variável renda
x2 = x^2
ajuste1 = lm(y ~ x) 
ajuste2 = lm(y ~ x+x2) 

curva = function(x){
  ajuste2$coefficients[1]+ajuste2$coefficients[2]*x+ajuste2$coefficients[3]*x^2 
}

reta = function(x){
  ajuste1$coefficients[1]+ajuste1$coefficients[2]*x
}

x_dominio = seq(0.5,1.2,length.out=200)
y_curva = curva(x_dominio)
y_reta = reta(x_dominio)

# SALVANDO GRAFICO RENDA PER CAPITA  VS DESPESAS PER CAPITA EM ESCOLAS PUBLICAS.
pdf(file="aplicacao1.pdf",width=9,height=9, paper="special",
    family="Bookman",pointsize=20)
plot(x,y, xlab=expression(paste("Per capita income/",10^4)), ylab = "Per capita expenditure on public schools",pch=1, lwd=2)
grid(lwd=2)
lines(x_dominio,y_reta, lwd = 2.2, lty = 1)
arrows(1, 750, 1.07, 810, col = "black")
text(1, 750, "Alaska", pos = 1, cex=1.1)
#points(1.0851,821, col="white", pch=16, lwd=30)
#points(1.0851,821, col="red", pch=16, lwd=2)
#points(1.0022,428, col="red", pch=16, lwd=2)
arrows(1.002, 300, 1.002, 410, col = "black")
text(1, 300, "Washington D.C.", pos = 1, cex=1.1)
#lines(x_dominio,y_curva, lwd = 2)
dev.off()

# SALVANDO GRAFICO RENDA PER CAPITA  VS DESPESAS PER CAPITA EM ESCOLAS PUBLICAS.
pdf(file="aplicacao2.pdf",width=9,height=9, paper="special",
    family="Bookman",pointsize=20)
plot(x,y, xlab=expression(paste("Per capita income/",10^4)),
     ylab = "Per capita expenditure on public schools",
     pch=1, lwd=2, ylim=c(150,900))
grid(lwd=2)
#lines(x_dominio,y_reta, lwd = 2.2, lty = 1)
arrows(1, 750, 1.07, 810, col = "black")
text(1, 750, "Alaska", pos = 1, cex=1.1)
#points(1.0851,821, col="white", pch=16, lwd=30)
#points(1.0851,821, col="red", pch=16, lwd=2)
#points(1.0022,428, col="red", pch=16, lwd=2)
arrows(1.002, 300, 1.002, 410, col = "black")
text(1, 300, "Washington D.C.", pos = 1, cex=1.1)
text(0.62, 200, "Mississippi", pos = 1, cex=1.1)
lines(x_dominio,y_curva, lwd = 2)
arrows(0.60, 180, 0.579, 240, col = "black")
dev.off()


# Elementos da diagonal da matriz H.
h = hatvalues.lm(ajuste2)
which(h>4*3/dim(dados)[1]) # As observacoes 3p/n sao pontos de alavanca.

resettest(ajuste1) # Teste Reset (Test para Linearidade)
# resettest(ajuste2) # Teste Reset (Test para Linearidade)

# TESTE DE HETEROSCEDASTICIDADE - Breusch-Pagan.
breusch_pagan = bptest(ajuste1)

# TESTE QUASE T

# Modelo 1:
HC4_modelo1 = vcovHC(ajuste1,type="HC4")
tau_beta1_modelo1 = ajuste1$coefficients[1]/sqrt(HC4_modelo1[1,1]) 
tau_beta2_mdelo1 = ajuste1$coefficients[2]/sqrt(HC4_modelo1[2,2]) 

# Modelo 2:
HC4_modelo2 = vcovHC(ajuste2,type="HC4")
tau_beta1_modelo2 = as.vector(ajuste2$coefficients[1])/sqrt(HC4_modelo2[1,1]) 
tau_beta2_modelo2 = as.vector(ajuste2$coefficients[2])/sqrt(HC4_modelo2[2,2]) 
tau_beta3_modelo2 = as.vector(ajuste2$coefficients[3])/sqrt(HC4_modelo2[3,3]) 

# TESTE QUASI-F 

# Modelo1:
X = cbind(1,x)
HC4_modelo1 = vcovHC(ajuste1,type="HC4")
R = matrix(c(1,0,0,1), byrow=TRUE, nrow=2, ncol=2)
m_chapeu = R%*%ajuste1$coefficients
W = t(m_chapeu)%*%R%*%HC4_modelo1%*%t(R)%*%m_chapeu


# Modelo2 :
R = matrix(c(1,0,0,0,1,0,0,0,1), byrow=TRUE, nrow=3, ncol=3)
m_chapeu = R%*%ajuste2$coefficients
W = t(m_chapeu)%*%R%*%HC4_modelo2%*%t(R)%*%m_chapeu

############ BOOTSTRAP TESTE DE HIPOTESE - TESTE QUASI-T #################### 
library(sandwich) # Calcula os HC.
ajuste2 = lm(y ~ x+x2) 
beta_chapeu = as.vector(ajuste2$coefficients) # beta_chapeu da amostra original.
X = cbind(1, x, x2) # Matriz de regressores.
h = as.vector(hatvalues.lm(ajuste2)) # h_i (grau de alavancagem).

HC4 = vcovHC(ajuste2,type="HC4")

tau_1 = beta_chapeu[1]/sqrt(HC4[1,1]) 
tau_2 = beta_chapeu[2]/sqrt(HC4[2,2])
tau_3 = beta_chapeu[3]/sqrt(HC4[3,3])

tau_1_estrela = vector()
tau_2_estrela = vector()
tau_3_estrela = vector()


# H_0 : \beta_2 = 0;
ajuste2_restrito = lm(y ~ x2) 
B = 1000 # Numero de replicas do bootstrap.
for(i in 1:B){
  t_estrela = sample(c(-1,1),size=50,replace = TRUE)
  y_estrela = X%*%c(ajuste2_restrito$coefficients[1],0,ajuste2_restrito$coefficients[2])
  + t_estrela*ajuste2_restrito$residuals/(1-h)
  ajuste_estrela = lm(y_estrela ~ x + x2)
  beta_chapeu_estrela = as.vector(ajuste_estrela$coefficients)
  HC4_estrela = vcovHC(ajuste_estrela,type="HC4")
  tau_2_estrela[i] = beta_chapeu_estrela[2]/sqrt(HC4_estrela[2,2])
}

tau_2_estrela = abs(tau_2_estrela)
p_valor_beta_2 = (1+length(tau_2_estrela[tau_2_estrela>=abs(tau_2)]))/(B+1)


# H_0 : \beta_3 = 0;
ajuste2_restrito = lm(y ~ x) 
B = 1000 # Numero de replicas do bootstrap.
for(i in 1:B){
  t_estrela = sample(c(-1,1),size=50,replace = TRUE)
  y_estrela = X%*%c(ajuste2_restrito$coefficients[1],ajuste2_restrito$coefficients[2],0)
  + t_estrela*ajuste2_restrito$residuals/(1-h)
  ajuste_estrela = lm(y_estrela ~ x + x2)
  beta_chapeu_estrela = as.vector(ajuste_estrela$coefficients)
  HC4_estrela = vcovHC(ajuste_estrela,type="HC4")
  tau_3_estrela[i] = beta_chapeu_estrela[3]/sqrt(HC4_estrela[3,3])
}

tau_3_estrela = abs(tau_3_estrela)
p_valor_beta_3 = (1+length(tau_3_estrela[tau_3_estrela>=abs(tau_3)]))/(B+1)
