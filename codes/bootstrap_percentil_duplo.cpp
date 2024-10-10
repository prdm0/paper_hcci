/* *********************************************************************
                 Modelos de Regressao Heteroscedastico
                  Intervalos de Confiança Bootstrap
                Bootstrap e Bootstrap Percentil Duplo
========================================================================
Orientador: Francisco Cribari Neto.
E-mail: cribari@de.ufpe.br 
Orientando: Pedro Rafael Diniz Marinho.
E-mail - pedro.rafael.marinho@gmail.com
Mestrado em Estatistica - UFPE.
************************************************************************ */

// NOTAS SOBRE O PROGRAMA:

// Esse programa calcula o intervalos de confiancas para modelos lineares heteroscedasticos
// utilizando o bootstrap percentil simples e duplo. Ao rodar o programa é obtido as estimativas
// do bootstrap percentil simples e duplo.

/* Versoes das biliotecas utilizadas
   Armadillo - versao 3.2.4
   GSL - versao 4.6.3
*/

/*Compilando o codigo usando a biblioteca armadillo e gsl

Comando para compilacao: g++ -O3 -march=native -mtune=corei7 -funroll-loops -frerun-loop-opt -funroll-all-loops -ffast-math -o bootstrap_percentil_duplo bootstrap_percentil_duplo.cpp -lgsl -larmadillo -lgslcblas
*/

/*Instalacao da biblioteca armadillo no GNU/Linux
Ubuntu: apt-get install libarmadillo2 && libarmadillo-dev
Fedora: yum install armadillo
Site: http://arma.sourceforge.net/

Para instalar a biblioteca pelo codigo fonte siga os seguintes passos:
(1) apt-get install cmake
(2) Na pasta da biblioteca execute:
  	(2.1) ./configure
    (2.2) make
    (2.3) make install
*/
//#define ARMA_DONT_USE_BLAS /*A compilacao da biblioteca blas pode ser 32-bits. Logo, nao sera eficiente a sua utilizacao.*/
//#define ARMA_USE_LAPACK
#include <iostream>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <time.h>
#include "armadillo"		/* Biblioteca de Algebra Linear para C++ */
#include <boost/math/distributions/normal.hpp>	/* Utilizado para obtencao dos quantis da normal */
#include <boost/math/distributions/students_t.hpp>	/* Utilizado para obtencao dos quantis da t-student */

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

/* (1) Esse namespace tata-se de funcoes gerais que nao estao implementadas na biblioteca armadillo.
   (2) Apesar de algumas das funcoes implementadas nesse namespace estarem implementadas em bibliotecas
       como por exemplo a GSL, as funções aqui implementadas podem trabalhar diretamente com os tipos de dados
       suportados pela biblioteca armadillo.
   (3) As funcoes buscam ser de facil uso.
   (4) Informacoes para o uso das funcoes implementadas nesse namespace podem ser encontradas nos comentarios 
       destas funcoes. */

// PARA RODAR O PROGRAMA APENAS MUDE OS VALORES DAS VARIAVEIS DEFINIDAS NO 
// PAINEL DE CONTROLE.

int nrep = 5000;           // NUMERO DE REPLICAS DE MONTE CARLO.
int nrep_boot = 500;	   // NUMERO DE REPLICAS DE BOOTSTRAP.
int nrep_boot_duplo = 250; // NUMERO DE REPLICAS DO BOOTSTAP DUPLO.
int samplesize = 1;	   // NUMERO DE REPLICACOES DA MATRIZ X. A MATRIZ X SERA REPLICADAS samplisize VEZES.      
int nobs = 20;		   // NUMERO DE OBSERVACOES. SE esquema = 1, A MATRIZ X TERA nobs LINHAS. NO CASO EM QUE esquema = 2 
		           // A MATRIZ X TERA nobs*samplesize LINHAS.      
int esquema = 2;	   // SE esquema = 1 A OPCAO samplesize SERA DESCONSIDERADA. DESSA FORMA, A SEGUNDA COLUNA DA MATRIX X
			   // SERA GERADA DIRETAMENTE DE UMA DISTRIBUICAO T COM 3 GRAUS DE LIBERDADE. CASO A ESCOLHA SEJA
			   // samplesize = 2 GERAMOS INICIALMENTE UMA MATRIZ COM nobs LINHAS E POSTERIORMENTE REPLICAMOS ESSA 
			   // MATRIZ samplesize VEZES. 
double lambda = 9;	   // BASTA FIXAR O VALOR DE LAMBDA QUE O VALOR DA CONSTANTE "a" E ESCOLHIDO AUTOMATICAMENTE. 
                           // ASSIM O VALOR DE LAMBDA TRABALHADO SERA MUITO PROXIMO AO VALOR DE LAMBIDA ESCOLHIDO.
                           // POR EXEMPLO, PARA "lambda = 9" O LAMBIDA ESCOLHIDO É IGUAL A 9.00017.                      
int dist_erro = 1;	   // ESCOLHA DA DISTRIBUICAO DOS ERROS: 1: normal; 2: t(3); 3: chi-squared(2).
int dist_t = 1;		   // 1: rademacher; 2: normal padrao.

int
main ()
{

  const clock_t tempo_inicial = clock ();

  double a;

  // VETOR DE UNS. VETOR COM OS PARAMETROS VERDADEIROS.

  vec beta = ones < vec > (2);	

  gsl_rng *r;

  //GERADOR UTILIZADO. 

  r = gsl_rng_alloc (gsl_rng_tt800);

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
  //X.save("X.mat",arma_ascii);

  //X.load("X.mat",arma_ascii);

  mat eta = X * beta;		// PREDITOR LINEAR.

  mat P = inv (trans (X) * X) * trans (X);	// P = (X'X)^{-1}*X'.

  mat H = X * P;		// MATRIZ CHAPEU, H = X(X'X)^{-1}X'.

  mat h = diagvec (H);		// VETOR DE MEDIDAS DE ALAVANCAGEM.

  vec contador (nobs);		//      ARMAZENA OS PONTOS DE ALTA ALAVANCAGEM.

  // CONTANDO O NUMERO DE PONTOS DE ALAVANCA.
  for (int d = 0; d < nobs; d++)
    {
      if (h (d) > 4.0 / nobs)
	contador (d) = 1;
      else
	contador (d) = 0;
    }

  // "A" É UM VETOR COM POSSIVEIS DANDIDATOS A SER O VALOR DE "a" QUE NOS DARA UM LAMBDA PROXIMO
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

  sigma2 = exp (a_utilizado * X.col (1));	// VETOR DE VARIANCIAS.
  sigma = sqrt (sigma2);	// VETOR DE DESVIOS PADROES. 
  lambda = sigma2.max () / sigma2.min ();	// RAZAO ENTRE O MAXIMO E O MINIMO DAS VARIANCIAS.

  // DADOS PRELIMINARES. INFORMACOES SOBRE O NUMERO DE REPLICAS DE MONTE CARLO, BOOTSTRAP, BOOTSTRAP DUPLO.
  // TAMBEM E APRESENTADO INFORMACOES SOBRE O VALOR DE LAMBDA UTILIZADO E O VALOR DE "a" ESCOLHIDO, ASSIM COMO
  // O NUMERO DE PONTOS DE ALTA ALAVANCAGEM.

  cout << "\t \t DADOS DA SIMULACAO" << endl << endl;
  cout << ">> [*] nobs = " << nobs << endl;
  cout << ">> [*] lambda = " << lambda << endl;
  cout << ">> [*] a = " << a_utilizado << endl;
  cout << ">> [*] nrep_boot = " << nrep_boot << endl;
  cout << ">> [*] nrep_boot_duplo = " << nrep_boot_duplo << endl;
  cout << ">> [*] Quant. de pontos de alavanca = " << arma::sum (contador) <<
    endl;

  if (dist_erro == 1)
    cout << ">> [*] Distribuicao do erro = normal" << endl;
  if (dist_erro == 2)
    cout << ">> [*] Distribuicao do erro = t(3)" << endl;
  if (dist_erro == 3)
    cout << ">> [*] Distribuicao do erro = qui-quadrado(2)" << endl;

  if (dist_t == 1)
    cout << ">> [*] Distribuicao de t^* = rademacher" << endl;
  if (dist_t == 2)
    cout << ">> [*] Distribuicao de t^* = normal padrao" << endl;

  cout << ">> [*] Gerador utilizado = " << "gsl_rng_tt800 da biblioteca GSL"
    << endl;
  cout << ">> [*] Semente do gerador = 0" << endl;

  // REGRESSANDO QUE SERA GERADO DENTRO DAS REPLICAS DE MONTE CARLO.
  vec Y (nobs);

  // ALGUMAS VARIAVEIS NECESSARIAS.
  vec beta2 (nrep_boot), t_estrela (nobs), y_estrela (nobs);	// ALGUMAS VARIAVEIS NECESSARIAS.
  vec epsilon_chapeu_boot_duplo (nobs), beta_chapeu_boot_duplo,
    beta2_chapeu_boot_duplo_temp (nrep_boot_duplo);
  vec betaj_estrela_menos_betaj (nrep_boot);	// OUTRAS VARIAVEIS NECESSARIAS.

  // VARIAVEIS QUE ARMAZENAM AS nrep COBERTURAS ESTIMADAS PELO BOOTSTRAP PERCENTIL INTERIOR.
  vec cob90_boot_percentil (nrep), cob95_boot_percentil (nrep),
    cob99_boot_percentil (nrep);

  // VARIAVEIS QUE ARMAZENAM AS nrep AMPLITUDES ESTIMADAS PELO BOOTRSTRAP PERCENTIL INTERIOR.
  vec ampl90_boot_percentil (nrep), ampl95_boot_percentil (nrep),
    ampl99_boot_percentil (nrep);

  // VARIAVEIS QUE ARMAZENAM AS nrep NAO COBERTURAS A ESQUERDA DO BOOTSTRAP PERCENTIL INTERIOR.
  vec ncobesq90_boot_percentil (nrep), ncobesq95_boot_percentil (nrep),
    ncobesq99_boot_percentil (nrep);

  // VARIAVEIS QUE ARMAZENAM AS nrep NAO COBERTURAS A DIREITA DO BOOTSTRAP PERCENTIL INTERIOR.
  vec ncobdi90_boot_percentil (nrep), ncobdi95_boot_percentil (nrep),
    ncobdi99_boot_percentil (nrep);

  // LIMITES DOS INTERVALOS ESTIMADOS PELO BOOTSTRAP PERCENTIL INTERIOR.  
  double li90, ls90, li95, ls95, li99, ls99;

  // VARIAVEIS QUE ARMAZENAM AS AMPLITUDES DOS nrep INTERVALOS ESTIMADOS PELO BOOTSTRAP PERCENTIL DUPLO (EXTERIOR).               
  vec ampl90_percentil_duplo (nrep), ampl95_percentil_duplo (nrep),
    ampl99_percentil_duplo (nrep);

  // VARIAVEIS QUE ARMAZENAM AS COBERTURAS ESTIMADAS DOS nrep INTERVALOS ESTIMADOS PELO BOOSTRAP PERCENTIL DUPLO (EXTERIOR).
  vec cob90_percentil_duplo (nrep), cob95_percentil_duplo (nrep),
    cob99_percentil_duplo (nrep);

  // VARIAVEIS QUE ARMAZENAM AS NAO COBERTURAS A ESQUERDA DO BOOTSTRAP PERCENTIL DUPLO (EXTERIOR).
  vec ncobesq90_percentil_duplo (nrep), ncobesq95_percentil_duplo (nrep),
    ncobesq99_percentil_duplo (nrep);

  // VARIAVEIS QUE ARMAZENAM AS NAO COBERTURAS A DIREITA DO BOOTSTRAP PERCENTIL DUPLO (EXTERIOR).
  vec ncobdi90_percentil_duplo (nrep), ncobdi95_percentil_duplo (nrep),
    ncobdi99_percentil_duplo (nrep);

  // UTILIZADO NA GERACAO DO VALOR DE t^*.
  double numero;

  // AQUI COMECA O LACO DE MONTE CARLO.
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

      mat temp = P * Y;		// ESTIMADO DE MINIMOS QUADRADOS ORDINARIO.
      mat Xtemp = X * temp;
      mat epsilon_chapeu = Y - Xtemp;

      vec u_estrela (nrep_boot);
      u_estrela.zeros ();
      double u_estrela_numerador = 0;

      // AQUI COMMECA O LACO BOOTSTRAP.                       
      for (int k = 0; k < nrep_boot; k++)
	{
	  u_estrela_numerador = 0;	// UTILIZADO NO BOOTSTRAP DUPLO.
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

	  vec beta_chapeu_boot = P * y_estrela;

	  beta2 (k) = beta_chapeu_boot (1);	// BETA2 DA REPLICA DE BOOTSTRAP.

	  epsilon_chapeu_boot_duplo = y_estrela - X * beta_chapeu_boot;

	  // Aqui comeca o bootstrap duplo
	  for (int m = 0; m < nrep_boot_duplo; m++)
	    {

	      vec y_estrela_estrela (nobs);	// Variavel resposta dentro do bootstrap
	      vec t_estrela_estrela (nobs);	// Número aleatório com média zero e variância 1.

	      double numero;
	      for (int a = 0; a < nobs; a++)
		{
		  double numero = gsl_ran_gaussian (r, 1);
		  t_estrela_estrela (a) = numero;
		}

	      for (int t = 0; t < nobs; t++)
		{
		  y_estrela_estrela (t) =
		    as_scalar ((X.row (t)) * beta_chapeu_boot +
			       t_estrela_estrela (t) *
			       epsilon_chapeu_boot_duplo (t) / sqrt (1 -
								     h (t)));

		}

	      beta_chapeu_boot_duplo = P * y_estrela_estrela;	// Estimativa dos betas estrela (bootstrap). \hat{beta^{*}}
	      beta2_chapeu_boot_duplo_temp (m) = as_scalar (beta_chapeu_boot_duplo (1));	

	      if (beta_chapeu_boot_duplo (1) <=
		  2 * beta_chapeu_boot (1) - temp (1))
		u_estrela_numerador = 1 + u_estrela_numerador;

	    }// AQUI TERMINA O LACO BOOTSTRAP DUPLO.

	  u_estrela (k) = u_estrela_numerador / nrep_boot_duplo;

	  betaj_estrela_menos_betaj (k) = beta_chapeu_boot (1) - temp (1);

	}// AQUI TERMINA O LACO BOOTSTRAP.

      // INTERVALO BOOTSTRAP PERCENTIL (BOOTSTRAP INTERIOR).
      li95 = myfunctions::quantil (beta2, 0.025, nrep_boot);
      ls95 = myfunctions::quantil (beta2, 0.975, nrep_boot);
      if (beta (1) >= li95 && beta (1) <= ls95)
	cob95_boot_percentil (i) = 1;
      else
	cob95_boot_percentil (i) = 0;

      ampl95_boot_percentil (i) = ls95 - li95;

      if (beta (1) < li95)
	ncobesq95_boot_percentil (i) = 1;
      else
	ncobesq95_boot_percentil (i) = 0;

      if (beta (1) > ls95)
	ncobdi95_boot_percentil (i) = 1;
      else
	ncobdi95_boot_percentil (i) = 0;

      // INTERVALO PERCENTIL BOOTSTRAP DUPLO - 95% (BOOTSTRAP EXTERIOR).
      double hat_ql95 = myfunctions::quantil (u_estrela, 0.025, nrep_boot);
      double hat_qu95 = myfunctions::quantil (u_estrela, 0.975, nrep_boot);
      cout << hat_qu95 << endl;
      double ls95_percentil_duplo =
	temp (1) - myfunctions::quantil (betaj_estrela_menos_betaj, hat_ql95,
					 nrep_boot);
      double li95_percentil_duplo =
	temp (1) - myfunctions::quantil (betaj_estrela_menos_betaj, 10,
					 nrep_boot);

      ampl95_percentil_duplo (i) =
	ls95_percentil_duplo - li95_percentil_duplo;

      if (li95_percentil_duplo <= beta (1)
	  && beta (1) <= ls95_percentil_duplo)
	cob95_percentil_duplo (i) = 1;
      else
	cob95_percentil_duplo (i) = 0;

      if (beta (1) < li95_percentil_duplo)
	ncobesq95_percentil_duplo (i) = 1;
      else
	ncobesq95_percentil_duplo (i) = 0;

      if (beta (1) > ls95_percentil_duplo)
	ncobdi95_percentil_duplo (i) = 1;
      else
	ncobdi95_percentil_duplo (i) = 0;

    }// AQUI TERMINA O LACO MONTE CARLO.

  cout << endl << "\t \t RESULTADOS - BOOTSTRAP PERCENTIL - IC95" << endl <<
    endl;
  cout << "------->> Cobertura (%) = " << (arma::sum (cob95_boot_percentil) /
					   nrep) * 100 << endl;
  cout << "------->> Nao cobertura a esquerda (%) = " << (arma::sum
							  (ncobesq95_boot_percentil)
							  / nrep) *
    100 << endl;
  cout << "------->> Nao cobertura a direita (%) = " << (arma::sum
							 (ncobdi95_boot_percentil)
							 / nrep) *
    100 << endl;
  cout << "------->> Amplitude media = " << (arma::sum (ampl95_boot_percentil)
					     / nrep) * 100 << endl << endl;

  cout << endl << "\t \t RESULTADOS - BOOTSTRAP DUPLO PERCENTIL - IC95" <<
    endl << endl;
  cout << "------->> Cobertura (%) = " << (arma::sum (cob95_percentil_duplo) /
					   nrep) * 100 << endl;
  cout << "------->> Nao cobertura a esquerda (%) = " << (arma::sum
							  (ncobesq95_percentil_duplo)
							  / nrep) *
    100 << endl;
  cout << "------->> Nao cobertura a direita (%) = " << (arma::sum
							 (ncobdi95_percentil_duplo)
							 / nrep) *
    100 << endl;
  cout << "------->> Amplitude media = " <<
    (arma::sum (ampl95_percentil_duplo) / nrep) * 100 << endl << endl;

  cout << "(*) TEMPO DE EXECUCAO: " << float (clock () -
					      tempo_inicial) /
    CLOCKS_PER_SEC << " segundos / " << (float (clock () - tempo_inicial) /
					 CLOCKS_PER_SEC)/60 << " minutos / "
    << ((float (clock () - tempo_inicial) / CLOCKS_PER_SEC)/60) /
    60 << " horas / " <<
    (((float (clock () - tempo_inicial) / CLOCKS_PER_SEC)/60) / 60) /
    24 << " dias." << endl << endl;

  return 0;
}
