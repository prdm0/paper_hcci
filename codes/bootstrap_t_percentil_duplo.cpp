/* *********************************************************************
                 Modelos de Regressão Heterocedástico
                  Intervalos de Confiança Bootstrap
                   Bootstrap t-percentil corrigido    
========================================================================
Orientador: Francisco Cribari Neto.
E-mail: cribari@de.ufpe.br 
Orientando: Pedro Rafael Diniz Marinho.
E-mail - pedro.rafael.marinho@gmail.com
Mestrado em Estatística - UFPE.
************************************************************************ */

// NOTAS SOBRE O PROGRAMA:

// Este programa refere-se ao metodo boostrap t-percentil
// com correcao da quantidade do denominador da variavel z^{*}. Essa quantidade
// refere-se a raiz quadrada de \hat{var}(\hat{\beta_j}^*}. É utilizado o segundo
// nivel de bootstrap para corregir essa quantidade no bootstrap exterior.

// Esse programa faz uso de duas correcoes. Uma correcao na quantidade do denominador
// da variavel z^* e a outra correcao corrige o desvio que entra no calculo dos 
// limites do intervalo de confinaca.

/* Versões das biliotecas utilizadas
   Armadillo - versão 3.2.4
   GSL - versão 4.6.3
*/

/*Compilando o código usando a biblioteca armadillo e gsl

Comando para compilação: g++ -O3 -o bootstrap_t_percentil_duplo bootstrap_t_percentil_duplo.cpp -lgsl -lopenblas -larmadillo

*/

/*Instalação da biblioteca armadillo no GNU/Linux
Ubuntu: apt-get install libarmadillo2 && libarmadillo-dev
Fedora: yum install armadillo
Site: http://arma.sourceforge.net/

Para instalar a biblioteca pelo código fonte siga os seguintes passos:
(1) apt-get install cmake
(2) Na pasta da biblioteca execute:
  	(2.1) ./configure
    (2.2) make
    (2.3) make install
*/

//#define ARMA_DONT_USE_BLAS	/*A compilação da biblioteca blas pode ser 32-bits. Logo, não será eficiente a sua utilização. */
//#define ARMA_USE_LAPACK

#include <iostream>
//#include <omp.h>
#include <sstream> 
#include <string.h>
#include <fstream> /* Biblioteca para leitura e escrita em arquivos */
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <time.h>
#include "armadillo"		/* Biblioteca de Algebra Linear para C++ */

using namespace arma;
using namespace std;

namespace myfunctions
{
  double quantil (vec dados, double p, int n)
  {
    vec xx = sort (dados);
    double x[n];
    for (int i = 0; i < n; i++)
      {
	x[i] = xx (i);
      }
    return gsl_stats_quantile_from_sorted_data (x, 1, n, p);
  }
}

// (1) Esse namespace tata-se de funções gerais que não estão implementadas na biblioteca armadillo.
// (2) Apesar de algumas das funções implementadas nesse namespace estarem implementadas em bibliotecas
//     como por exemplo a GSL, as funções aqui implementadas podem trabalhar diretamente com os tipos de dados
//     suportados pela biblioteca armadillo.
// (3) As funções buscam ser de fácil uso.
// (4) Informações para o uso das funções implementadas nesse namespace podem ser encontradas nos comentários 
//       destas funções.

// Para rodar o programa apenas mude os valores das variáveis definidas no
// painel de controle definido logo abaixo:


int nrep = 5000;	     // NUMERO DE REPLICAS DE MONTE CARLO.
int nrep_boot = 500;         // NUMERO DE REPLICAS DO BOOTSTRAP T-PERCENTIL.
int nrep_boot_duplo = 250;   // NUMERO DE REPLICAS DO BOOTSTRAP DUPLO T-PERCENTIL. 
int samplesize = 1;	     // NUMERO DE REPLICACOES DA MATRIZ X. A MATRIZ X SERA REPLICADAS samplisize VEZES.      
int nobs = 20;		     // NUMERO DE OBSERVACOES. SE esquema = 1, A MATRIZ X TERA nobs LINHAS. NO CASO EM QUE esquema = 2 

		             // A MATRIZ X TERA nobs*samplesize LINHAS.      
int esquema = 2;             // SE esquema = 1 A OPCAO samplesize SERA DESCONSIDERADA. DESSA FORMA, A SEGUNDA COLUNA DA MATRIX X
			     // SERA GERADA DIRETAMENTE DE UMA DISTRIBUICAO T COM 3 GRAUS DE LIBERDADE. CASO A ESCOLHA SEJA
			     // esquema = 2 GERAMOS INICIALMENTE UMA MATRIZ COM nobs LINHAS E POSTERIORMENTE REPLICAMOS ESSA 
			     // MATRIZ samplesize VEZES. 

double lambda = 49;           // BASTA FIXAR O VALOR DE LAMBDA QUE O VALOR DA CONSTANTE "a" É ESCOLHIDO AUTOMATICAMENTE.
                             // ASSIM O VALOR DE  LAMBDA TRABALHADO SERA MUITO PROXIMO AO VALOR DE LAMBIDA ESCOLHIDO.
                             // POR EXEMPLO, PARA "lambda = 9" O LAMBIDA ESCOLHIDO É IGUAL A 9.00017. 
int hc = 4;                  // MÉTODO HC UTILIZADO NA SIMULAÇÃO.  
int dist_erro = 1;	     // ESCOLHA DA DISTRIBUICAO DOS ERROS: 1: normal; 2: t(3); 3: chi-squared(2)
int dist_t = 2;		     // 1: rademacher; 2: normal padrao
int ncorrecoes = 2;          // NUMERO DE CORRECOES UTILIZADAS. SE ncorrecoes = 1 APENAS O ERRO PADRAO (QUANTIDADE NO DENOMINADOR DA VARIÁVEL
                             // z^{*}) SERÁ CORRIGIDO. PARA ISSO, É UTILIZADO O BOOTSTRAP INTERIOR PARA ESTIMATIVA DO VIÉS. SE ncorrecoes = 2,
                             // TAMBÉM SERÁ CORRIGIDO O DESVIO QUE ENTRA NO CÁLCULO DO INTERVALO DE CONFIANÇA. PARA ISSO, É UTILIZADO O 			     //	BOOTSTAP EXTERIOR.

int
main ()
{
  time_t rawtime;
  struct tm * timeinfo;
  time (&rawtime);
  timeinfo = localtime (&rawtime);

  const clock_t tempo_inicial = clock ();
  
  ofstream saida(" ultima_simulacao_RENOMEAR.txt");	
  double a;

  vec beta = ones < vec > (2);	// VETOR DE UNS. VETOR COM OS PARAMETROS VERDADEIROS.

  // Definição do gerador 
  gsl_rng *r;

  //GERADOR UTILZADO. 

  r = gsl_rng_alloc (gsl_rng_tt800);
  //r = gsl_rng_alloc (gsl_rng_mt19937);
  // DEFININDO SEMENTE DO GERADOR.
  
  gsl_rng_set (r, 0);		

  mat X (nobs, 2);

  // PRIMEIRO ESQUEMA PARA GERACAO DA MATRIZ X.
  
  if (esquema == 1)
    {
      X = ones < mat > (nobs, 2);
      for (int colunas = 1; colunas < 2; colunas++)
	{
	  for (int linhas = 0; linhas < nobs; linhas++)
	    {
	      X (linhas, colunas) = gsl_ran_tdist (r, 3);
	    }
	}

    }

  // SEGUNDO ESQUEMA PARA GERACAO DA MATRIZ X.

  if (esquema == 2)
    {
      X = ones < mat > (nobs, 2);
      for (int colunas = 1; colunas < 2; colunas++)
	{
	  for (int linhas = 0; linhas < nobs; linhas++)
	    {
	      X (linhas, colunas) = gsl_ran_tdist (r, 3);
	    }
	}

      mat X1;
      X1 = X;
      int l = 1;

      while (l < samplesize)
	{
	  X = join_cols (X, X1);
	  l++;
	}
      nobs = nobs * samplesize;
    }
   
  // PREDITOR LINEAR.

  mat eta = X * beta;
		
  // P = (X'X)^{-1}*X'

  mat P = inv (sympd (trans (X) * X)) * trans (X);
	
  // TRANSPOSTA DA MATRIZ P.

  mat Pt = trans (P);	
	
  // MATRIZ CHAPEU, H = X(X'X)^{-1}X'.

  mat H = X * P;		

  // VETOR DE MEDIDAS DE ALAVANCAGEM.

  mat h = diagvec (H);		

  mat hmax = max (h);

  mat g = nobs / 2 * h;

  // USADA EM HC0.
  mat weight0 =  1.0/ (1.0 - h);

  // USADA EM HC3.
  mat weight3 = 1.0 / (pow ((1.0 - h), 2.));	

  //Todos elementos do vetor h são maiores que 4? 
  mat vetor_4 = ones < mat > (X.n_rows) + 3;	//Vetor com 1 somado à 3, ou seja, vetor com elementos 4.
  uvec verificando_limite_inferior_g4 = find (g > vetor_4);
  mat g4 (nobs, 1), weight4 (nobs, 1);

  if (double (verificando_limite_inferior_g4.n_rows) == nobs)
    {
      weight4 = 1.0 / pow ((1.0 - h), 4.0);
    }

  if (double (verificando_limite_inferior_g4.n_rows) < nobs)
    {
      for (int i = 0; i < nobs; i++)
	{
	  weight4 (i, 0) = as_scalar (1.0 / pow ((1 - h (i, 0)), g (i, 0)));
	}
    }

  mat gtemp;
  if (as_scalar (nobs * 0.7 * hmax / 1) > 4.0)
    {
      gtemp = as_scalar (nobs * 0.7 * hmax / 1);
    }
  else
    {
      gtemp = 4.0;
    }

  mat vetor_gtemp = zeros < mat > (X.n_rows) + as_scalar (gtemp);
  uvec verificando_limite_inferior_g5 = find (g < vetor_gtemp);
  mat g5 (nobs, 1), weight5 (nobs, 1); // USADO EM HC5.

  if (verificando_limite_inferior_g5.n_rows < nobs)
    {
      double expoente;
      expoente = as_scalar (gtemp);
      weight5 = sqrt (1.0 / sqrt (pow ((1.0 - h), expoente)));
    }
  else if (verificando_limite_inferior_g5.n_rows == nobs)
    {
      for (int i = 0; i < nobs; i++)
	{
	  weight5 (i, 0) =
	    as_scalar (sqrt (1.0 / sqrt (pow ((1.0 - h (i, 0)), g (i, 0)))));
	}
    }
  
  //cout << weight5 << endl;
  // ARMAZENA OS PONTOS DE ALTA ALAVANCAGEM.
  vec contador (nobs);		

  // CONTANDO O NUMERO DE PONTOS DE ALAVANCA.

  for (int d = 0; d < nobs; d++)
    {
      if (h (d) > 4.0 / nobs)
	contador (d) = 1;
      else
	contador (d) = 0;
    }

  // "A" É UM VETOR COM POSSIVEIS CANDIDATOS A SER O VALOR DE "a" QUE NOS DARA UM LAMBDA PROXIMO
  // DO VALOR DE lambda ESCOLHIDO.

  vec A (4000000);
  A (0) = 0;
  for (int s = 1; s < 4000000; s++)
    {
      A (s) = A (s - 1) + 0.000001;
    }

  double lambda_utilizado, a_utilizado;

  if (lambda == 1)
    {
      a_utilizado = 0;
    }

  if (lambda != 1)
    {
      int s = 0;
      mat resultado;
      while (lambda_utilizado <= lambda - 0.000001)
	{
	  resultado = exp (A (s) * X.col (1));
	  lambda_utilizado = resultado.max () / resultado.min ();
	  s++;
	}
      a_utilizado = as_scalar (A (s));
    }

  vec sigma2 (nobs), sigma (nobs);

  // VETOR DE VARIANCIAS.

  sigma2 = exp (a_utilizado * X.col (1));

  // VETOR DE DESVIOS PADROES. 	 
  sigma = sqrt (sigma2);

  // RAZAO ENTRE O MAXIMO E O MINIMO DAS VARIANCIAS.	
  lambda = sigma2.max () / sigma2.min ();	     

  // DADOS PRELIMINARES. INFORMACOES SOBRE O NUMERO DE REPLICAS DE MONTE CARLO, BOOTSTRAP, BOOTSTRAP DUPLO.
  // TAMBEM E APRESENTADO INFORMACOES SOBRE O VALOR DE LAMBDA UTILIZADO E O VALOR DE "a" ESCOLHIDO, ASSIM COMO
  // O NUMERO DE PONTOS DE ALTA ALAVANCAGEM.

  saida << "\t \t DADOS DA SIMULACAO" << endl << endl;
  saida << ">> [*] nobs = " << nobs << endl;
  saida << ">> [*] lambda = " << lambda << endl;
  saida << ">> [*] hc = " << hc << endl; 
  saida << ">> [*] a = " << a_utilizado << endl;
  saida << ">> [*] nrep = " << nrep << endl;
  saida << ">> [*] nrep_boot = " << nrep_boot << endl;
  saida << ">> [*] nrep_boot_duplo = " << nrep_boot_duplo << endl;
  saida << ">> [*] ncorrecoes = " << ncorrecoes << endl;
  saida << ">> [*] Quant. de pontos de alavanca = " << arma::sum (contador) <<
    endl;

  if (dist_erro == 1)
    saida << ">> [*] Distribuicao do erro = normal" << endl;
  if (dist_erro == 2)
    saida << ">> [*] Distribuicao do erro = t(3)" << endl;
  if (dist_erro == 3)
    saida << ">> [*] Distribuicao do erro = qui-quadrado(2)" << endl;

  if (dist_t == 1)
    saida << ">> [*] Distribuicao de t^* = rademacher" << endl;
  if (dist_t == 2)
    saida << ">> [*] Distribuicao de t^* = normal padrao" << endl;

  saida << ">> [*] Gerador utilizado = " << "gsl_rng_tt800 da biblioteca GSL"
    << endl;
  saida << ">> [*] Semente do gerador = 0" << endl;
  saida << ">> [*] Horario de inico da simulacao: " <<  asctime(timeinfo); 
  mat betahat = zeros < mat > (2, nrep); // vector used to store the estimates

  //VARIAVEIS DO BOOTSTRAP.

  vec epsilon_chapeu(nobs), y_estrela(nobs), beta_chapeu_boot;
  vec beta2_chapeu_boot;
  vec beta2_chapeu_boot_temp (nrep_boot);
  vec z_estrela (nrep_boot), z_estrela_duplo(nrep_boot);
  vec betaj_estrela_menos_betaj (nrep_boot); // VARIAVEL UTILIZADA NO BOOTSTRAP PERCENTIL.
  vec beta2 (nrep_boot); // VARIAVEL UTILIZADA NO BOOTSTRAP PERCENTIL.

  vec cob95_percentil (nrep), cob99_percentil (nrep),
  cob90_percentil (nrep), ncobesq95_percentil (nrep),
  ncobdi95_percentil (nrep), ncobesq99_percentil (nrep),
  ncobdi99_percentil (nrep), ncobesq90_percentil (nrep),
  ncobdi90_percentil (nrep), ampl95_percentil (nrep),
  ampl90_percentil (nrep), ampl99_percentil (nrep);

  vec cob95_percentil_duplo (nrep), cob99_percentil_duplo (nrep),
  cob90_percentil_duplo (nrep), ncobesq95_percentil_duplo (nrep),
  ncobdi95_percentil_duplo (nrep), ncobesq99_percentil_duplo (nrep),
  ncobdi99_percentil_duplo (nrep), ncobesq90_percentil_duplo (nrep),
  ncobdi90_percentil_duplo (nrep), ampl95_percentil_duplo (nrep),
  ampl90_percentil_duplo (nrep), ampl99_percentil_duplo (nrep);

  vec cob95_t_percentil (nrep), cob99_t_percentil (nrep),
  cob90_t_percentil (nrep), ncobesq95_t_percentil (nrep),
  ncobdi95_t_percentil (nrep), ncobesq99_t_percentil (nrep),
  ncobdi99_t_percentil (nrep), ncobesq90_t_percentil (nrep),
  ncobdi90_t_percentil (nrep), ampl95_t_percentil (nrep),
  ampl90_t_percentil (nrep), ampl99_t_percentil (nrep);

  double li95, li90, ls90, ls95, li99, ls99,hat_ql95, hat_qu95;

  // VARIAVEIS DO BOOTSTRAP DUPLO.

  vec epsilon_chapeu_boot_duplo, beta_chapeu_boot_duplo,
    beta2_chapeu_boot_duplo, beta2_chapeu_boot_duplo_temp (nrep_boot_duplo),
    y_estrela_estrela, t_estrela_estrela;

  vec cob95_t_percentil_duplo (nrep),
    cob99_t_percentil_duplo (nrep),
    cob90_t_percentil_duplo (nrep),
    ncobesq95_t_percentil_duplo (nrep),
    ncobdi95_t_percentil_duplo (nrep),
    ncobesq99_t_percentil_duplo (nrep),
    ncobdi99_t_percentil_duplo (nrep),
    ncobesq90_t_percentil_duplo (nrep),
    ncobdi90_t_percentil_duplo (nrep),
    ampl95_t_percentil_duplo (nrep),
    ampl90_t_percentil_duplo (nrep),
    ampl99_t_percentil_duplo (nrep);

  // AQUI COMECA O LACO DE MONTE CARLO.

  vec Y = ones < vec > (nobs);
  mat HC0, HC3, HC4, HC5;
  
  mat weight_usado;

  if(hc == 0){
    weight_usado = weight0;
  }
  if(hc == 3){
    weight_usado = weight3;
  }
  if(hc == 4){
    weight_usado = weight4;
  }
  if(hc == 5){
    weight_usado = weight5;	
  }

  mat produtos = inv (sympd (trans (X) * X)) * trans (X);

  // UTILIZADO NA GERACAO DO VALOR DE t^*.
  double numero;
  //#pragma omp parallel
  //{
  //#pragma omp for
  for (int i = 0; i < nrep; i++)
    {
      if (dist_erro == 1)
	{
	  for (int v = 0; v < nobs; v++)
	    {
	      Y (v) = eta (v) + sigma (v) * gsl_ran_gaussian (r, 1.0);
	    }
	}
      if (dist_erro == 2)
	{
	  for (int v = 0; v < nobs; v++)
	    {
	      Y (v) =
		eta (v) + sigma (v) * (gsl_ran_tdist (r, 3) / sqrt (1.5));
	    }
	}

      if (dist_erro == 3)
	{
	  for (int v = 0; v < nobs; v++)
	    {
	      Y (v) =
		eta (v) + sigma (v) * (gsl_ran_chisq (r, 2) - 2.0) / 2.0;
	    }
	}

      mat invrXX = inv (sympd (trans (X) * X));
      mat temp = invrXX * trans (X) * Y;
      mat resid2 = arma::pow ((Y - X * temp), 2.0); // epsilon ao quadrado. 
      betahat.col (i) = temp;

      mat matrixtemp (nobs, 2);	// p=2

      resid2 = repmat (resid2, 1, 2);	// p=2
      matrixtemp = resid2 % Pt;

      weight4 = repmat (weight_usado, 1, 2);	// *****
      HC4 = P * (matrixtemp % weight4);
      weight4.resize (nobs, 1);

      epsilon_chapeu = Y - X * temp;	// ESTIMATIVAS DOS ERROS.
      mat Xtemp = X * temp;	// X*\hat{beta}.

      mat HC4_b;
      mat HC4_b_duplo;
      
      vec hc4_b(nrep_boot); 

      vec u_estrela (nrep_boot);
      u_estrela.zeros ();
      double u_estrela_numerador = 0;

      // AQUI COMECA O LACO BOOTSTRAP.
      for (int k = 0; k < nrep_boot; k++)
	{
          u_estrela_numerador = 0;
	  vec y_estrela (nobs);	// VARIAVEL RESPOSTA UTILIZADA NO BOOTSTRAP.
	  vec t_estrela (nobs);	// NUMERO ALEATORIO COM MEDIA ZERO E VARINCIA UM.

	  if (dist_t == 2)
	    {
	      for (int t = 0; t < nobs; t++)
		{
		  numero = gsl_ran_gaussian (r, 1.0);
		  t_estrela (t) = numero;
		}
	    }

	  if (dist_t == 1)
	    {
	      for (int t = 0; t < nobs; t++)
		{
		  numero = gsl_rng_uniform (r);
		  if (numero <= 0.5)
		    t_estrela (t) = -1;
		  if (numero > 0.5)
		    t_estrela (t) = 1;
		}
	    }

	  y_estrela = Xtemp + t_estrela % epsilon_chapeu / sqrt (1 - h);	// CONFERIDO.

	  // Aqui temos as estimativas de \hat{{\beta^{*}}_j}. Lembrando que nosso interesse eh \hat{{\beta^{*}}_2}

	  beta_chapeu_boot = produtos * y_estrela;	// Estimativa dos betas estrela (bootstrap). \hat{beta^{*}}
	  beta2_chapeu_boot_temp (k) = as_scalar (beta_chapeu_boot (1));	
	  mat matrixtemp_b (nobs, 2);
	  mat resid2_b = arma::pow ((y_estrela - X * beta_chapeu_boot), 2.0);
          
	  resid2_b = repmat (resid2_b, 1, 2);	// p=2

	  matrixtemp_b = resid2_b % Pt;
	  mat weight4_b = repmat (weight_usado, 1, 2); //************* 

	  HC4_b = P * (matrixtemp_b % weight4_b);
          hc4_b(k) = sqrt(as_scalar(HC4_b(1,1)));
	  z_estrela(k) =
	  (as_scalar (beta2_chapeu_boot_temp (k) - temp (1))) /
	  sqrt (HC4_b (1, 1));
          beta2 (k) = beta_chapeu_boot (1);	// BETA2 DA REPLICA DE BOOTSTRAP.
	  epsilon_chapeu_boot_duplo = y_estrela - X * beta_chapeu_boot;	// SERA UTILIZADO NO BOOTSTRAP DUPLO.

          // VETOR QUE IRA ARMAZENAR AS ESTIMATIVAS HC4 DO BOOTSTRAP DUPLO QUE
          // SERA UTILIZADO PARA CORRIGIR O ERRO PADRAO DO BOOTSTRAP EXTERIOR.
          vec hc4_duplo(nrep_boot_duplo); 

	  double desvio_b = sqrt(as_scalar(HC4_b (1, 1)));

	  // AQUI COMECA O BOOTSTRAP DUPLO.
	  for (int m = 0; m < nrep_boot_duplo; m++)
	    {

	      vec y_estrela_estrela (nobs);	// VARIAVEL RESPOSTA DENTRO DO BOOTSTRAP DUPLO.
	      vec t_estrela_estrela (nobs);	// NUMERO ALEATORIO COM MEDIA 0 E VARIANCIA 1.

	      if (dist_t == 2)
	      {
	         for (int t = 0; t < nobs; t++)
		 {
		     numero = gsl_ran_gaussian (r, 1.0);
		     t_estrela_estrela (t) = numero;
		 }
	      }

	      if (dist_t == 1)
	      {
	         for (int t = 0; t < nobs; t++)
		 {
		    numero = gsl_rng_uniform (r);
		    if (numero <= 0.5)
		       t_estrela_estrela (t) = -1;
		    if (numero > 0.5)
		       t_estrela_estrela (t) = 1;
		}
	      }

	      for (int t = 0; t < nobs; t++)
	      {
		  y_estrela_estrela (t) =
		    as_scalar ((X.row (t)) * beta_chapeu_boot +
			       t_estrela_estrela (t) *
			       epsilon_chapeu_boot_duplo (t) / sqrt (1 -
								     h (t)));
	      }
	      
	      // A VARIAVEL produtos REFERE-SE A (X'X)^-1
	      beta_chapeu_boot_duplo = produtos * y_estrela_estrela;	

	      beta2_chapeu_boot_duplo_temp (m) = as_scalar (beta_chapeu_boot_duplo (1));	 
	      
	      mat matrixtemp_b_duplo(nobs,2);                       

	      mat resid2_b_duplo = arma::pow((y_estrela_estrela - X*beta_chapeu_boot_duplo),2.0); 

	      resid2_b_duplo = repmat(resid2_b_duplo,1,2);

	      matrixtemp_b_duplo = resid2_b_duplo%Pt;

	      mat weight4_b_duplo = repmat(weight_usado,1,2); //************ antes era weight4

	      HC4_b_duplo = P * (matrixtemp_b_duplo%weight4_b_duplo);
              hc4_duplo(m) = sqrt(as_scalar(HC4_b_duplo(1,1))); 

	      if(beta_chapeu_boot_duplo (1) <= 2 * beta_chapeu_boot (1) - temp (1))
		u_estrela_numerador = 1 + u_estrela_numerador;

	    } // AQUI TERMINA O LACO DO BOOTSTRAP DUPLO.
          //cout << u_estrela_numerador / nrep_boot_duplo << endl;
	  u_estrela (k) = u_estrela_numerador / nrep_boot_duplo;

	  betaj_estrela_menos_betaj (k) = beta_chapeu_boot (1) - temp (1);
            z_estrela_duplo (k) =
	    (as_scalar (beta2_chapeu_boot_temp (k) - temp (1))) /
	    (2*arma::sum(hc4_duplo)/nrep_boot_duplo - desvio_b);
	} // AQUI TERMINA O LACO BOOTSTRAP.

      
      // CONTANDO CONVERGÊNCIAS PARA O BOOSTRAP T PERCENTIL DUPLO. (AQUI É O BOOTSTRAP DUPLO.)    
      // Confiança de 95%.
      double quantil_inferior95 =
	myfunctions::quantil (z_estrela_duplo, 0.975, nrep_boot);
      double quantil_superior95 =
	myfunctions::quantil (z_estrela_duplo, 0.025, nrep_boot);

      if(ncorrecoes == 2){
      // AQUI ESTAMOS CORRIGINDO O CALCULO DOS LIMITES INFERIORES E SUPERIORES DO INTERVALO DE CONFIANCA.
      // ESSA CORRECAO FAZ USO DO BOOTSTRAP EXTERIOR.
      li95 = temp (1, 0) - quantil_inferior95 * (2*arma::sum(hc4_b)/nrep_boot - sqrt (HC4 (1, 1)));
      ls95 = temp (1, 0) - quantil_superior95 * (2*arma::sum(hc4_b)/nrep_boot - sqrt (HC4 (1, 1)));
      }

      if(ncorrecoes == 1){ 
      	// AQUI ESTAMOS CONSTRUINDO OS LIMITES DOS INTERVALOS DE CONFIANCAS SEM USAR A CORRECAO DO DESVIO QUE
      	// ENTRA NO CALCULO DOS LIMITES.	
      	li95 = temp (1, 0) - quantil_inferior95 * sqrt (HC4 (1, 1));
      	ls95 = temp (1, 0) - quantil_superior95 * sqrt (HC4 (1, 1));
      }

      if (beta (1) >= li95 && beta (1) <= ls95)
	{
	  cob95_t_percentil_duplo (i) = 1;
	}
      else
	cob95_t_percentil_duplo (i) = 0;

      if (beta (1) < li95)
	{
	  ncobesq95_t_percentil_duplo (i) = 1;
	}
      else
	ncobesq95_t_percentil_duplo (i) = 0;

      if (beta (1) > ls95)
	{
	  ncobdi95_t_percentil_duplo (i) = 1;
	}
      else
	ncobdi95_t_percentil_duplo (i) = 0;

      ampl95_t_percentil_duplo (i) = ls95 - li95;

      // CONTANDO CONVERGÊNCIAS PARA O BOOTSTRAP T PERCENTIL (AQUI NÃO É O BOOSTRAP DUPLO.)
   
      quantil_inferior95 = myfunctions::quantil (z_estrela, 0.975, nrep_boot);
      quantil_superior95 = myfunctions::quantil (z_estrela, 0.025, nrep_boot);
      li95 = temp (1, 0) - quantil_inferior95 * sqrt (HC4 (1, 1));
      ls95 = temp (1, 0) - quantil_superior95 * sqrt (HC4 (1, 1));

      if (beta (1) >= li95 && beta (1) <= ls95)
	{
	  cob95_t_percentil (i) = 1;
	}
      else
	cob95_t_percentil (i) = 0;

      if (beta (1) < li95)
	{
	  ncobesq95_t_percentil (i) = 1;
	}
      else
	ncobesq95_t_percentil (i) = 0;

      if (beta (1) > ls95)
	{
	  ncobdi95_t_percentil (i) = 1;
	}
      else
	ncobdi95_t_percentil (i) = 0;

      ampl95_t_percentil (i) = ls95 - li95;

      // INTERVALO BOOTSTRAP PERCENTIL.
      li95 = myfunctions::quantil (beta2, 0.025, nrep_boot);
      ls95 = myfunctions::quantil (beta2, 0.975, nrep_boot);
      if (beta (1) >= li95 && beta (1) <= ls95)
	cob95_percentil (i) = 1;
      else
	cob95_percentil (i) = 0;

      ampl95_percentil (i) = ls95 - li95;

      if (beta (1) < li95)
	ncobesq95_percentil (i) = 1;
      else
	ncobesq95_percentil (i) = 0;

      if (beta (1) > ls95)
	ncobdi95_percentil (i) = 1;
      else
	ncobdi95_percentil (i) = 0;

      // INTERVALO PERCENTIL BOOTSTRAP DUPLO - 95% (BOOTSTRAP EXTERIOR).
      double hat_ql95 = myfunctions::quantil (u_estrela, 0.025, nrep_boot);
      double hat_qu95 = myfunctions::quantil (u_estrela, 0.975, nrep_boot);

      ls95 =
	temp (1) - myfunctions::quantil (betaj_estrela_menos_betaj, hat_ql95,
					 nrep_boot);
      //cout << hat_qu95 << endl;
      //cout << myfunctions::quantil (betaj_estrela_menos_betaj, 0.7258,
	//				 nrep_boot) << endl;
      li95 =
	temp (1) - myfunctions::quantil (betaj_estrela_menos_betaj, hat_qu95,
					 nrep_boot);


     ampl95_percentil_duplo (i) = ls95 - li95;

      if (li95 <= beta (1)
	  && beta (1) <= ls95)
	cob95_percentil_duplo (i) = 1;
      else
	cob95_percentil_duplo (i) = 0;

      if (beta (1) < li95)
	ncobesq95_percentil_duplo (i) = 1;
      else
	ncobesq95_percentil_duplo (i) = 0;

      if (beta (1) > ls95)
	ncobdi95_percentil_duplo (i) = 1;
      else
	ncobdi95_percentil_duplo (i) = 0;


    } // AQUI TERMINA O LACO MONTE CARLO
    //}
       time_t rawtime_1;
       time (&rawtime_1);
       timeinfo = localtime (&rawtime_1);
     saida << ">> [*] Horario de termino da simulacao: " <<  asctime(timeinfo) << endl; 
     saida  << "(*) TEMPO DE EXECUCAO: " <<  float(clock() - tempo_inicial)/CLOCKS_PER_SEC << " segundos / " <<
         (float(clock() - tempo_inicial)/CLOCKS_PER_SEC)/60 << " minutos / " <<
         ((float(clock() - tempo_inicial)/CLOCKS_PER_SEC)/60)/60 << " horas / " << 
         (((float(clock() - tempo_inicial)/CLOCKS_PER_SEC)/60)/60)/24 << " dias." << endl << endl;
     saida << "----------------------------------------------------------------------------------------" << endl;
     saida << "                              BOOTSTRAP T-PERCENTIL                                     " << endl;
     saida << "----------------------------------------------------------------------------------------" << endl; 
     saida << "------------------------------------> 95% <---------------------------------------------" << endl;
     saida << "COBERTURA = " << (arma::sum (cob95_t_percentil) / nrep)*100 << endl;
     saida << "AMPLITUDE = " << (arma::sum (ampl95_t_percentil) / nrep) << endl;
     saida << "NAO COBERTURA A ESQUERDA = " << (arma::sum (ncobesq95_t_percentil) / nrep)*100 << endl;
     saida << "NAO COBERTURA A DIREITA = " << (arma::sum (ncobdi95_t_percentil) / nrep)*100 << endl;
     saida << "----------------------------------------------------------------------------------------" << endl;
     saida << "                            BOOTSTRAP T-PERCENTIL DUPLO                                 " << endl;
     saida << "----------------------------------------------------------------------------------------" << endl; 
     saida << "------------------------------------> 95% <---------------------------------------------" << endl;
     saida << "COBERTURA = " << (arma::sum (cob95_t_percentil_duplo) / nrep)*100 << endl;
     saida << "AMPLITUDE = " << (arma::sum (ampl95_t_percentil_duplo) / nrep) << endl;
     saida << "NAO COBERTURA A ESQUERDA = " << (arma::sum (ncobesq95_t_percentil_duplo) / nrep)*100 << endl;
     saida << "NAO COBERTURA A DIREITA = " << (arma::sum (ncobdi95_t_percentil_duplo) / nrep)*100 << endl;
     saida << "----------------------------------------------------------------------------------------" << endl;
     saida << "                                BOOTSTRAP PERCENTIL                                     " << endl;
     saida << "----------------------------------------------------------------------------------------" << endl; 
     saida << "------------------------------------> 95% <---------------------------------------------" << endl;
     saida << "COBERTURA = " << (arma::sum (cob95_percentil) / nrep)*100 << endl;
     saida << "AMPLITUDE = " << (arma::sum (ampl95_percentil) / nrep) << endl;
     saida << "NAO COBERTURA A ESQUERDA = " << (arma::sum (ncobesq95_percentil) / nrep)*100 << endl;
     saida << "NAO COBERTURA A DIREITA = " << (arma::sum (ncobdi95_percentil) / nrep)*100 << endl;
     saida << "----------------------------------------------------------------------------------------" << endl;
     saida << "                              BOOTSTRAP PERCENTIL DUPLO                                 " << endl;
     saida << "----------------------------------------------------------------------------------------" << endl; 
     saida << "------------------------------------> 95% <---------------------------------------------" << endl;
     saida << "COBERTURA = " << (arma::sum (cob95_percentil_duplo) / nrep)*100 << endl;
     saida << "AMPLITUDE = " << (arma::sum (ampl95_percentil_duplo) / nrep) << endl;
     saida << "NAO COBERTURA A ESQUERDA = " << (arma::sum (ncobesq95_percentil_duplo) / nrep)*100 << endl;
     saida << "NAO COBERTURA A DIREITA = " << (arma::sum (ncobdi95_percentil_duplo) / nrep)*100 << endl;
     saida.close();

     //cout << "\a" << endl; // ALERTA SONORO.
     return 0;

} // AQUI TERMINA A FUNCAO main().
