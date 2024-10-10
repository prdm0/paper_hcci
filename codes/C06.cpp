/* *********************************************************************
	Modelos de Regressão Heterocedástico
	Intervalos de Confiança Bootstrap
	Esse programa faz os seguites intervalos
	(1) Bootstrap percentil
	(2) Bootstrap duplo percentil
	(3) Bootstrap t
	(4) Bootstrap duplo t   
	=====================================================================
Orientador: Francisco Cribari Neto.
E-mail: cribari@de.ufpe.br 
Orientando: Pedro Rafael Diniz Marinho.
E-mail - pedro.rafael.marinho@gmail.com
Mestrado em Estatística - UFPE.
 *********************************************************************** */

// NOTAS SOBRE O PROGRAMA:

// Esse programa faz a avaliação dos intervalos de confianças sem utilizar esquemas
// bootstap e também utilizando esquemas bootstarp: bootstrap percetil, bootstrap
// percentil duplo, bootstrap t e bootstrap t duplo para todos os HC's: HC0, HC2,
// HC3, HC4 e HC5. É evidente que o bootstrap percentil e bootstrap duplo percentil
// não faz uso dos estimadores HC.

// Esse programa faz uso de duas correcoes. Uma correcao na quantidade do denominador
// da variavel z^* e a outra correcao corrige o desvio que entra no calculo dos 
// limites do intervalo de confinaca. Esse esquema nao é correto mas foi mantido
// nesse código fonte. O esquema correto do bootstrap t duplo foi acrescentado. Esse
// esquema foi inspirado no algoritmo do artigo B.D.,McCULLOUGH, H.D.,VINOD. Implementing
// the Double Bootstrap, Computational Economics, 12, 79-95, 1998.


/* Versões das biliotecas utilizadas
	Armadillo - versão 3.900.4  
	GSL - versão 4.8.1
 */

/*Compilando o código usando a biblioteca armadillo, openblas e gsl. O usuário deve
  se certificar que a biblioteca armadillo está configurada para reconhecer o openblas.
  Caso não esteja reconhecendo dará um erro o uso da flag -lopenblas. O usuário tem duas 
  opção: compilar sem usar a flag ou instalar a biblioteca openblas que é mais otimizada.
  Comando para compilação: g++ -O3 -o 04 C04.cpp -lopenblas -lgsl -larmadillo
 */

/*Instalação da biblioteca armadillo no GNU/Linux
Ubuntu: apt-get install libarmadillo2 && libarmadillo-dev
Fedora: yum install armadillo
Manjaro: yaourt -S armadillo
Site: http://arma.sourceforge.net/

Para instalar a biblioteca pelo código fonte siga os seguintes passos:
(1) cmake
(2) Na pasta da biblioteca execute:
(2.1) ./configure
(2.2) make
(2.3) make install
 */

/*A compilação da biblioteca blas pode ser 32-bits. Logo, não será eficiente a sua utilização. */
//#define ARMA_DONT_USE_BLAS
//#define ARMA_USE_LAPACK

#include <iostream>
//#include <omp.h>
#include <sstream>
#include <string.h>
#include <fstream>		/* Biblioteca para leitura e escrita em arquivos */
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_cdf.h>
#include <time.h>
#include "armadillo"		/* Biblioteca de Algebra Linear para C++ */

using namespace arma;
using namespace std;

namespace myfunctions{
	double quantil(vec dados, double p, int n){
		vec xx = sort(dados);
		double x[n];
		for(int i = 0; i < n; i++){
			x[i] = xx(i);
		}return gsl_stats_quantile_from_sorted_data(x, 1, n, p);
	}
	// ESSE QUANTIL É O QUE É DESCRITO NA MAIORIA DOS ALGORITMOS BOOTSTRAP. ESSE QUANTIL PEGA A POSICAO (n+1)*alpha, EM QUE
	// alfa É O PERCENTIL DE INTERESSE.
	double quantil1(vec dados, double p, int n){
		vec xx = sort(dados);
		int indice = floor((n + 1) * p);
		if(indice < 0)
			indice = 0;
		if(indice >= n)
			indice = n - 1;
		double resul = xx(indice);
		return resul;
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

int nrep = 2000; // NUMERO DE REPLICAS DE MONTE CARLO.
int nrep_boot = 200; // NUMERO DE REPLICAS DO BOOTSTRAP T-PERCENTIL.
int nrep_boot_duplo = 100; // NUMERO DE REPLICAS DO BOOTSTRAP DUPLO T-PERCENTIL. 
int samplesize = 1; // NUMERO DE REPLICACOES DA MATRIZ X. A MATRIZ X SERA REPLICADAS samplisize VEZES.      
int nobs = 20; // NUMERO DE OBSERVACOES. SE esquema = 1, A MATRIZ X TERA nobs LINHAS. NO CASO EM QUE esquema = 2 
// A MATRIZ X TERA nobs*samplesize LINHAS.      
int esquema = 2; // SE esquema = 1 A OPCAO samplesize SERA DESCONSIDERADA. DESSA FORMA, A SEGUNDA COLUNA DA MATRIX X
// SERA GERADA DIRETAMENTE DE UMA DISTRIBUICAO T COM 3 GRAUS DE LIBERDADE. CASO A ESCOLHA SEJA
// esquema = 2 GERAMOS INICIALMENTE UMA MATRIZ COM nobs LINHAS E POSTERIORMENTE REPLICAMOS ESSA 
// MATRIZ samplesize VEZES. 
double lambda = 49; // BASTA FIXAR O VALOR DE LAMBDA QUE O VALOR DA CONSTANTE "a" É ESCOLHIDO AUTOMATICAMENTE.
// ASSIM O VALOR DE  LAMBDA TRABALHADO SERA MUITO PROXIMO AO VALOR DE LAMBIDA ESCOLHIDO.
// POR EXEMPLO, PARA "lambda = 9" O LAMBIDA ESCOLHIDO É IGUAL A 9.00017. 
int balanceado = 2; // BALANCEADO (SEM PONTOS DE ALAVANCA) SE	balanceado = 1. NAO BALANCEADO, OU SEJA,
// COM PONTOS DE ALAVANCA SE balanceado = 2;
int dist_erro = 4; // ECOLHA DA DISTRIBUICAO DOS ERROS: 1: normal; 2: t(3); 3: chi-squared(2); 4: weibull(2,3)
// 5: gumbel type II (2.5,2), 6: gamma(3,1.5).
int dist_t = 1;	// 1: rademacher; 2: normal padrao.
int gerador = 2; // 1: tt800; 2: mt19937; 3: random256_bsd;
int semente = 1987; // SEMENTE DO GERADOR;
int ncorrecoes = 2; // NUMERO DE CORRECOES UTILIZADAS. SE ncorrecoes = 1 APENAS O ERRO PADRAO
int pivo = 1; // QUANTIDADE PIVOTAL QUE SERA UTILIZADA. pivo PODERA ASSUMIR OS VALORES 1 E 2.
//(QUANTIDADE NO DENOMINADOR DA VARIÁVEL z^{*}) SERÁ CORRIGIDO. PARA ISSO, É UTILIZADO O BOOTSTRAP
// INTERIOR PARA ESTIMATIVA DO VIÉS. SE ncorrecoes = 2, TAMBÉM SERÁ CORRIGIDO O DESVIO QUE ENTRA
// NO CÁLCULO DO INTERVALO DE CONFIANÇA. PARA ISSO, É UTILIZADO O BOOTSTAP EXTERIOR.

int main()
{
	time_t rawtime;
	struct tm *timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);

	const clock_t tempo_inicial = clock();

	ofstream saida("teste.txt");
    
	// BANCO DE DADOS AUXILIARES. ESSES BANCOS DE DADOS SERAO UTILIZADOS
	// PARA CONSTRUCAO DE GRAFICOS NA LINGUAGEM R. TODOS OS DADOS REFEREM-SE
	// AO NIVEL NOMINAL DE 95%.
	ofstream lils_ols("lim_inf_lim_sup_ols.txt");
	ofstream lils_hc0("lim_inf_lim_sup_hc0.txt");
	ofstream lils_hc2("lim_inf_lim_sup_hc2.txt");
	ofstream lils_hc3("lim_inf_lim_sup_hc3.txt");
	ofstream lils_hc4("lim_inf_lim_sup_hc4.txt");
	ofstream lils_hc5("lim_inf_lim_sup_hc5.txt");

	ofstream lils_percentil("lim_inf_lim_sup_percentil.txt");
	ofstream lils_percentil_duplo("lim_inf_lim_sup_percentil_duplo.txt");
	ofstream lils_hc0_bootstrapt("lim_inf_lim_sup_hc0_bootstrapt.txt");
	ofstream lils_hc0_bootstrapt_duplo("lim_inf_lim_sup_hc0_bootstrapt_duplo.txt");
	ofstream lils_hc2_bootstrapt("lim_inf_lim_sup_hc2_bootstrapt.txt");
	ofstream lils_hc2_bootstrapt_duplo("lim_inf_lim_sup_hc2_bootstrapt_duplo.txt");
	ofstream lils_hc3_bootstrapt("lim_inf_lim_sup_hc3_bootstrapt.txt");
	ofstream lils_hc3_bootstrapt_duplo("lim_inf_lim_sup_hc3_bootstrapt_duplo.txt");
	ofstream lils_hc4_bootstrapt("lim_inf_lim_sup_hc4_bootstrapt.txt");
	ofstream lils_hc4_bootstrapt_duplo("lim_inf_lim_sup_hc4_bootstrapt_duplo.txt");
	ofstream lils_hc5_bootstrapt("lim_inf_lim_sup_hc5_bootstrapt.txt");
	ofstream lils_hc5_bootstrapt_duplo("lim_inf_lim_sup_hc5_bootstrapt_duplo.txt");

        ofstream pivo_ols("pivo_ols.txt");
        ofstream pivo_hc0("pivo_hc0.txt");
        ofstream pivo_hc2("pivo_hc2.txt");
        ofstream pivo_hc3("pivo_hc3.txt");
        ofstream pivo_hc4("pivo_hc4.txt");
        ofstream pivo_hc5("pivo_hc5.txt");

	vec beta = ones<vec>(2);	// VETOR DE UNS. VETOR COM OS PARAMETROS VERDADEIROS.

	// Definição do gerador 
	gsl_rng *r;

	//GERADOR UTILZADO. 

	if(gerador==1){
		r = gsl_rng_alloc(gsl_rng_tt800);
	}
	if(gerador==2){
		r = gsl_rng_alloc (gsl_rng_mt19937);
	}
	if(gerador==3){
		r = gsl_rng_alloc(gsl_rng_random256_bsd);
	}

	// DEFININDO SEMENTE DO GERADOR.

	gsl_rng_set(r,semente);

	mat X(nobs,2);

	// PRIMEIRO ESQUEMA PARA GERACAO DA MATRIZ X.

	if(esquema==1){
		X = ones<mat>(nobs,2);
		if(balanceado == 2){
			for(int linhas = 0; linhas < nobs; linhas++) {
				X(linhas,1) = gsl_ran_tdist(r,3);
			}
		}else{
			int contando_influencia = 1;
			while(contando_influencia!=0){
				contando_influencia = 0;
				double alavanca = 4.0/nobs;					
				for(int i = 0; i < nobs; i++) {
					X(i,1) = gsl_ran_tdist(r,3);
				}
				mat  matriz_chapeu = X*inv(sympd(trans(X)*X))*trans(X);
				vec diag_chapeu  = diagvec(matriz_chapeu);	
				for(int i = 0; i < nobs;i++){
					if(diag_chapeu(i) > alavanca) contando_influencia = 1;
				}
			}
		}
	}

	if(esquema == 2){
		X = ones <mat>(nobs,2);
		if(balanceado == 2){
			for(int linhas = 0; linhas < nobs; linhas++){
				X(linhas,1) = gsl_ran_tdist(r,3);
			}
		}
		else{
			int contando_influencia = 1;
			while(contando_influencia!=0){
				contando_influencia = 0;
				double alavanca = 4.0/nobs;					
				for(int i = 0; i < nobs; i++) {
					X(i,1) = gsl_ran_tdist(r,3);
				}
				mat  matriz_chapeu = X*inv(sympd(trans(X)*X))*trans(X);
				vec diag_chapeu = diagvec(matriz_chapeu);	
				for(int i = 0; i < nobs;i++){
					if(diag_chapeu(i) > alavanca) contando_influencia = 1;
				}
			}		
		}

		mat X1;
		X1 = X;
		int l = 1;

		while(l<samplesize){
			X = join_cols(X,X1);
			l++;
		}
		nobs = nobs * samplesize;
	}

	// SEGUNDO ESQUEMA PARA GERACAO DA MATRIZ X.

	// PREDITOR LINEAR.
	//cout << X << endl;
	mat eta = X*beta;

	// P = (X'X)^{-1}*X'

	mat P = inv(sympd(trans(X)*X))*trans(X);

	// TRANSPOSTA DA MATRIZ P.

	mat Pt = trans(P);

	// MATRIZ CHAPEU, H = X(X'X)^{-1}X'.

	mat H = X*P;

	// VETOR DE MEDIDAS DE ALAVANCAGEM.

	vec h = diagvec(H);

	// USADO EM HC0, HC2, HC3, HC4 E HC5.
	vec weight0 = ones<vec>(nobs), weight2 = 1.0/(1.0-h),
		 weight3 = 1.0/pow((1.0-h), 2.0), weight4(nobs), weight5(nobs);

	double media_h = mean(h);
	for(int l=0; l<nobs; l++){
		if((h(l)/media_h)<4.0){
			weight4(l) = 1.0/pow((1-h(l)),h(l)/media_h);
		}else{
			weight4(l) = 1.0/pow((1-h(l)),4.0);
		}
	}

	double max;
	if(4.0>=(0.7*arma::max(h))/media_h)
		max = 4.0;
	else
		max = 0.7*arma::max(h)/media_h;

	for(int l=0; l<nobs; l++){
		if((h(l) / media_h)<=max)
			weight5(l) = 1.0/sqrt(pow((1-h(l)),h(l)/media_h));
		else
			weight5(l) = 1.0/sqrt(pow((1-h(l)),max));
	}	

	// ARMAZENA OS PONTOS DE ALTA ALAVANCAGEM.
	vec contador = zeros<vec>(nobs);

	// CONTANDO O NUMERO DE PONTOS DE ALAVANCA.

	for(int d=0;d<nobs;d++){
		if(h(d)>4.0/nobs)	// É CONSIDERADO PONTO DE ALAVANCA OBSERVACOES MAIORES QUE 2p/n.
			contador(d) = 1;
		else
			contador(d) = 0;
	}

	// "A" É UM VETOR COM POSSIVEIS CANDIDATOS A SER O VALOR DE "a" QUE NOS DARA UM LAMBDA PROXIMO
	// DO VALOR DE lambda ESCOLHIDO.

	vec A(4000000);
	A(0) = 0;
	for(int s=1;s<4000000;s++){
		A(s) = A(s-1)+0.00001;
	}

	double lambda_utilizado, a_utilizado;

	if(lambda==1){
		a_utilizado=0;
	}

	if (lambda!=1){
		int s = 0;
		mat resultado;
		while(lambda_utilizado<=lambda-0.00001){
			resultado = exp(A(s)*X.col(1));
			lambda_utilizado = resultado.max()/resultado.min();
			s++;
		}
		a_utilizado = as_scalar(A(s));
	}

	vec sigma2(nobs), sigma(nobs);

	// VETOR DE VARIANCIAS.

	sigma2 = exp(a_utilizado*X.col(1));

	// VETOR DE DESVIOS PADROES.     
	sigma = sqrt(sigma2);

	// RAZAO ENTRE O MAXIMO E O MINIMO DAS VARIANCIAS.      
	lambda = sigma2.max()/sigma2.min();

	// DADOS PRELIMINARES. INFORMACOES SOBRE O NUMERO DE REPLICAS DE MONTE CARLO, BOOTSTRAP, BOOTSTRAP DUPLO.
	// TAMBEM E APRESENTADO INFORMACOES SOBRE O VALOR DE LAMBDA UTILIZADO E O VALOR DE "a" ESCOLHIDO, ASSIM COMO
	// O NUMERO DE PONTOS DE ALTA ALAVANCAGEM.

	saida << "\n \t \t DADOS DA SIMULACAO" << endl << endl;
	saida << ">> [*] nobs = " << nobs << endl;
	saida << ">> [*] lambda = " << lambda << endl;
	saida << ">> [*] a = " << a_utilizado << endl;
	saida << ">> [*] nrep = " << nrep << endl;
	saida << ">> [*] nrep_boot = " << nrep_boot << endl;
	saida << ">> [*] nrep_boot_duplo = " << nrep_boot_duplo << endl;
	saida << ">> [*] ncorrecoes = " << ncorrecoes << endl;
	saida << ">> [*] h_max = " << h.max() << ", 2p/n = " << 4.0/nobs << " e 4p/n = " << 8.0/nobs << endl;
	saida << ">> [*] Quant. de pontos de alavanca = " <<
		arma::sum(contador) << endl;
	if (dist_erro==1)
		saida << ">> [*] Distribuicao do erro = normal" << endl;
	if (dist_erro==2)
		saida << ">> [*] Distribuicao do erro = t(3)" << endl;
	if (dist_erro==3)
		saida << ">> [*] Distribuicao do erro = qui-quadrado(2)" << endl;
	if (dist_erro==4)
		saida << ">> [*] Distribuicao do erro = weibull(2,3)" << endl;
	if (dist_erro==5)
		saida << ">> [*] Distribuicao do erro = gumbel(2.5,2)" << endl;
	if (dist_erro==6)
		saida << ">> [*] Distribuicao do erro = gama(3,1.5)" << endl;

	if (dist_t==1)
		saida << ">> [*] Distribuicao de t^* = rademacher" << endl;
	if (dist_t==2)
		saida << ">> [*] Distribuicao de t^* = normal padrao" << endl;

	if(gerador==1)
		saida << ">> [*] Gerador utilizado = tt800" << endl;
	if(gerador==2)
		saida << ">> [*] Gerador utilizado = mt19937" << endl;
	if(gerador==3)
		saida << ">> [*] Gerador utilizado = random256_bsd" << endl;

	saida << ">> [*] Semente do gerador = " << semente << endl;
	saida << ">> [*] Horario de inico da simulacao: " << asctime(timeinfo);

	//VARIAVEIS DO BOOTSTRAP.

	vec epsilon_chapeu(nobs), y_estrela(nobs), beta_chapeu_boot;
	vec beta2_chapeu_boot;
	vec beta2_chapeu_boot_temp(nrep_boot);

	vec z_estrela0(nrep_boot), z_estrela2(nrep_boot), z_estrela3(nrep_boot),
		 z_estrela4(nrep_boot), z_estrela5(nrep_boot) ,z0_estrela_duplo(nrep_boot),
		 z2_estrela_duplo(nrep_boot), z3_estrela_duplo(nrep_boot), z4_estrela_duplo(nrep_boot),
		 z5_estrela_duplo(nrep_boot), z_estrela_estrela0(nrep_boot_duplo), 
		 z_estrela_estrela2(nrep_boot_duplo), z_estrela_estrela3(nrep_boot_duplo),
		 z_estrela_estrela4(nrep_boot_duplo), z_estrela_estrela5(nrep_boot_duplo);

	vec betaj_estrela_menos_betaj(nrep_boot); // VARIAVEL UTILIZADA NO BOOTSTRAP PERCENTIL.
	vec beta2(nrep_boot); // VARIAVEL UTILIZADA NO BOOTSTRAP PERCENTIL.

	vec cob95_percentil(nrep), cob99_percentil(nrep),
		 cob90_percentil(nrep), ncobesq95_percentil(nrep),
		 ncobdi95_percentil(nrep), ncobesq99_percentil(nrep),
		 ncobdi99_percentil(nrep), ncobesq90_percentil(nrep),
		 ncobdi90_percentil(nrep), ampl95_percentil(nrep),
		 ampl90_percentil(nrep), ampl99_percentil(nrep);

	vec cob95_percentil_duplo(nrep), cob99_percentil_duplo(nrep),
		 cob90_percentil_duplo(nrep), ncobesq95_percentil_duplo(nrep),
		 ncobdi95_percentil_duplo(nrep), ncobesq99_percentil_duplo(nrep),
		 ncobdi99_percentil_duplo(nrep), ncobesq90_percentil_duplo(nrep),
		 ncobdi90_percentil_duplo(nrep), ampl95_percentil_duplo(nrep),
		 ampl90_percentil_duplo(nrep), ampl99_percentil_duplo(nrep);

	vec cob_0_95_t_percentil(nrep), cob_0_99_t_percentil(nrep),
		 cob_0_90_t_percentil(nrep), ncobesq_0_95_t_percentil(nrep),
		 ncobdi_0_95_t_percentil(nrep), ncobesq_0_99_t_percentil(nrep),
		 ncobdi_0_99_t_percentil(nrep), ncobesq_0_90_t_percentil(nrep),
		 ncobdi_0_90_t_percentil(nrep), ampl_0_95_t_percentil(nrep),
		 ampl_0_90_t_percentil(nrep), ampl_0_99_t_percentil(nrep);

	vec cob_2_95_t_percentil(nrep), cob_2_99_t_percentil(nrep),
		 cob_2_90_t_percentil(nrep), ncobesq_2_95_t_percentil(nrep),
		 ncobdi_2_95_t_percentil(nrep), ncobesq_2_99_t_percentil(nrep),
		 ncobdi_2_99_t_percentil(nrep), ncobesq_2_90_t_percentil(nrep),
		 ncobdi_2_90_t_percentil(nrep), ampl_2_95_t_percentil(nrep),
		 ampl_2_90_t_percentil(nrep), ampl_2_99_t_percentil(nrep);

	vec cob_3_95_t_percentil(nrep), cob_3_99_t_percentil(nrep),
		 cob_3_90_t_percentil(nrep), ncobesq_3_95_t_percentil(nrep),
		 ncobdi_3_95_t_percentil(nrep), ncobesq_3_99_t_percentil(nrep),
		 ncobdi_3_99_t_percentil(nrep), ncobesq_3_90_t_percentil(nrep),
		 ncobdi_3_90_t_percentil(nrep), ampl_3_95_t_percentil(nrep),
		 ampl_3_90_t_percentil(nrep), ampl_3_99_t_percentil(nrep);

	vec cob_4_95_t_percentil(nrep), cob_4_99_t_percentil(nrep),
		 cob_4_90_t_percentil(nrep), ncobesq_4_95_t_percentil(nrep),
		 ncobdi_4_95_t_percentil(nrep), ncobesq_4_99_t_percentil(nrep),
		 ncobdi_4_99_t_percentil(nrep), ncobesq_4_90_t_percentil(nrep),
		 ncobdi_4_90_t_percentil(nrep), ampl_4_95_t_percentil(nrep),
		 ampl_4_90_t_percentil(nrep), ampl_4_99_t_percentil(nrep);

	vec cob_5_95_t_percentil(nrep), cob_5_99_t_percentil(nrep),
		 cob_5_90_t_percentil(nrep), ncobesq_5_95_t_percentil(nrep),
		 ncobdi_5_95_t_percentil(nrep), ncobesq_5_99_t_percentil(nrep),
		 ncobdi_5_99_t_percentil(nrep), ncobesq_5_90_t_percentil(nrep),
		 ncobdi_5_90_t_percentil(nrep), ampl_5_95_t_percentil(nrep),
		 ampl_5_90_t_percentil(nrep), ampl_5_99_t_percentil(nrep);

	//double li95, li90, ls90, ls95, li99, ls99;

	double li_0_90, li_2_90, li_3_90, li_4_90, li_5_90,
			 ls_0_90, ls_2_90, ls_3_90, ls_4_90, ls_5_90,
			 li_0_95, li_2_95, li_3_95, li_4_95, li_5_95,
			 ls_0_95, ls_2_95, ls_3_95, ls_4_95, ls_5_95,
			 li_0_99, li_2_99, li_3_99, li_4_99, li_5_99,
			 ls_0_99, ls_2_99, ls_3_99, ls_4_99, ls_5_99;

	// VARIAVEIS DO BOOTSTRAP DUPLO.

	vec epsilon_chapeu_boot_duplo, beta_chapeu_boot_duplo,
		 beta2_chapeu_boot_duplo, y_estrela_estrela, t_estrela_estrela;

	vec cob_0_95_t_percentil_duplo(nrep), cob_0_99_t_percentil_duplo(nrep),
		 cob_0_90_t_percentil_duplo(nrep), ncobesq_0_95_t_percentil_duplo(nrep),
		 ncobdi_0_95_t_percentil_duplo(nrep), ncobesq_0_99_t_percentil_duplo(nrep),
		 ncobdi_0_99_t_percentil_duplo(nrep), ncobesq_0_90_t_percentil_duplo(nrep),
		 ncobdi_0_90_t_percentil_duplo(nrep), ampl_0_95_t_percentil_duplo(nrep),
		 ampl_0_90_t_percentil_duplo(nrep), ampl_0_99_t_percentil_duplo(nrep);

	vec cob_2_95_t_percentil_duplo(nrep), cob_2_99_t_percentil_duplo(nrep),
		 cob_2_90_t_percentil_duplo(nrep), ncobesq_2_95_t_percentil_duplo(nrep),
		 ncobdi_2_95_t_percentil_duplo(nrep), ncobesq_2_99_t_percentil_duplo(nrep),
		 ncobdi_2_99_t_percentil_duplo(nrep), ncobesq_2_90_t_percentil_duplo(nrep),
		 ncobdi_2_90_t_percentil_duplo(nrep), ampl_2_95_t_percentil_duplo(nrep),
		 ampl_2_90_t_percentil_duplo(nrep), ampl_2_99_t_percentil_duplo(nrep);

	vec cob_3_95_t_percentil_duplo(nrep), cob_3_99_t_percentil_duplo(nrep),
		 cob_3_90_t_percentil_duplo(nrep), ncobesq_3_95_t_percentil_duplo(nrep),
		 ncobdi_3_95_t_percentil_duplo(nrep), ncobesq_3_99_t_percentil_duplo(nrep),
		 ncobdi_3_99_t_percentil_duplo(nrep), ncobesq_3_90_t_percentil_duplo(nrep),
		 ncobdi_3_90_t_percentil_duplo(nrep), ampl_3_95_t_percentil_duplo(nrep),
		 ampl_3_90_t_percentil_duplo(nrep), ampl_3_99_t_percentil_duplo(nrep);

	vec cob_4_95_t_percentil_duplo(nrep), cob_4_99_t_percentil_duplo(nrep),
		 cob_4_90_t_percentil_duplo(nrep), ncobesq_4_95_t_percentil_duplo(nrep),
		 ncobdi_4_95_t_percentil_duplo(nrep), ncobesq_4_99_t_percentil_duplo(nrep),
		 ncobdi_4_99_t_percentil_duplo(nrep), ncobesq_4_90_t_percentil_duplo(nrep),
		 ncobdi_4_90_t_percentil_duplo(nrep), ampl_4_95_t_percentil_duplo(nrep),
		 ampl_4_90_t_percentil_duplo(nrep), ampl_4_99_t_percentil_duplo(nrep);

	vec cob_5_95_t_percentil_duplo(nrep), cob_5_99_t_percentil_duplo(nrep),
		 cob_5_90_t_percentil_duplo(nrep), ncobesq_5_95_t_percentil_duplo(nrep),
		 ncobdi_5_95_t_percentil_duplo(nrep), ncobesq_5_99_t_percentil_duplo(nrep),
		 ncobdi_5_99_t_percentil_duplo(nrep), ncobesq_5_90_t_percentil_duplo(nrep),
		 ncobdi_5_90_t_percentil_duplo(nrep), ampl_5_95_t_percentil_duplo(nrep),
		 ampl_5_90_t_percentil_duplo(nrep), ampl_5_99_t_percentil_duplo(nrep);

	vec cob_0_95_t_percentil_duplo1(nrep), cob_0_99_t_percentil_duplo1(nrep),
		 cob_0_90_t_percentil_duplo1(nrep), ncobesq_0_95_t_percentil_duplo1(nrep),
		 ncobdi_0_95_t_percentil_duplo1(nrep), ncobesq_0_99_t_percentil_duplo1(nrep),
		 ncobdi_0_99_t_percentil_duplo1(nrep), ncobesq_0_90_t_percentil_duplo1(nrep),
		 ncobdi_0_90_t_percentil_duplo1(nrep), ampl_0_95_t_percentil_duplo1(nrep),
		 ampl_0_90_t_percentil_duplo1(nrep), ampl_0_99_t_percentil_duplo1(nrep);

	vec cob_2_95_t_percentil_duplo1(nrep), cob_2_99_t_percentil_duplo1(nrep),
		 cob_2_90_t_percentil_duplo1(nrep), ncobesq_2_95_t_percentil_duplo1(nrep),
		 ncobdi_2_95_t_percentil_duplo1(nrep), ncobesq_2_99_t_percentil_duplo1(nrep),
		 ncobdi_2_99_t_percentil_duplo1(nrep), ncobesq_2_90_t_percentil_duplo1(nrep),
		 ncobdi_2_90_t_percentil_duplo1(nrep), ampl_2_95_t_percentil_duplo1(nrep),
		 ampl_2_90_t_percentil_duplo1(nrep), ampl_2_99_t_percentil_duplo1(nrep);

	vec cob_3_95_t_percentil_duplo1(nrep), cob_3_99_t_percentil_duplo1(nrep),
		 cob_3_90_t_percentil_duplo1(nrep), ncobesq_3_95_t_percentil_duplo1(nrep),
		 ncobdi_3_95_t_percentil_duplo1(nrep), ncobesq_3_99_t_percentil_duplo1(nrep),
		 ncobdi_3_99_t_percentil_duplo1(nrep), ncobesq_3_90_t_percentil_duplo1(nrep),
		 ncobdi_3_90_t_percentil_duplo1(nrep), ampl_3_95_t_percentil_duplo1(nrep),
		 ampl_3_90_t_percentil_duplo1(nrep), ampl_3_99_t_percentil_duplo1(nrep);

	vec cob_4_95_t_percentil_duplo1(nrep), cob_4_99_t_percentil_duplo1(nrep),
		 cob_4_90_t_percentil_duplo1(nrep), ncobesq_4_95_t_percentil_duplo1(nrep),
		 ncobdi_4_95_t_percentil_duplo1(nrep), ncobesq_4_99_t_percentil_duplo1(nrep),
		 ncobdi_4_99_t_percentil_duplo1(nrep), ncobesq_4_90_t_percentil_duplo1(nrep),
		 ncobdi_4_90_t_percentil_duplo1(nrep), ampl_4_95_t_percentil_duplo1(nrep),
		 ampl_4_90_t_percentil_duplo1(nrep), ampl_4_99_t_percentil_duplo1(nrep);

	vec cob_5_95_t_percentil_duplo1(nrep), cob_5_99_t_percentil_duplo1(nrep),
		 cob_5_90_t_percentil_duplo1(nrep), ncobesq_5_95_t_percentil_duplo1(nrep),
		 ncobdi_5_95_t_percentil_duplo1(nrep), ncobesq_5_99_t_percentil_duplo1(nrep),
		 ncobdi_5_99_t_percentil_duplo1(nrep), ncobesq_5_90_t_percentil_duplo1(nrep),
		 ncobdi_5_90_t_percentil_duplo1(nrep), ampl_5_95_t_percentil_duplo1(nrep),
		 ampl_5_90_t_percentil_duplo1(nrep), ampl_5_99_t_percentil_duplo1(nrep);

	vec Y = ones<vec>(nobs);

	// A MATRIZ C ABAIXO FORNECE OS ELEMENTOS c_jj	QUE SERAO UTILIZADOS
	// PARA AVALIAR O INTERVALOR DE CONFIANÇA OLS NO CASO DE HOMOSCEDASTICIDADE.

	mat C = inv(sympd(trans(X)*X));

	// ESSE PRODUTO SER FEITO FORA DE MONTE CARLO É IMPORTANTE
	// PARA QUE EM CADA RÉPLICA DE MONTE CARLO NÃO SEJA FEITO
	// INVERSAS E PRODUROS MATRICIAIS SEM NECESSIDADE.

	mat produtos = C*trans(X);

	// VALOR CRITICO
	// QUANTIS 1-\alpha/2 PARA DAS DISTRIBUICOES NORMAL PADRAO E T-STUDENT(n-p)
	// GRAUS DE LIBERDADE. 

	double vc_z1 = gsl_cdf_ugaussian_Pinv(1-0.01/2); // CONFIANCA DE 99%
	double vc_z5 = gsl_cdf_ugaussian_Pinv(1-0.05/2); // CONFIANCA DE 95%
	double vc_z10 = gsl_cdf_ugaussian_Pinv(1-0.10/2); // CONFIANCA DE 90%
	double vc_t1 = gsl_cdf_tdist_Pinv(1-0.01/2,nobs-2); // CONFIANCA DE 99%
	double vc_t5 = gsl_cdf_tdist_Pinv(1-0.05/2,nobs-2); // CONFIANCA DE 95%
	double vc_t10 = gsl_cdf_tdist_Pinv(1-0.10/2,nobs-2); // CONFIANCA DE 90%

	vec cob90_t_ols(nrep), cob95_t_ols(nrep), cob99_t_ols(nrep),
		 cob90_t_hc0(nrep), cob95_t_hc0(nrep), cob99_t_hc0(nrep),
		 cob90_t_hc2(nrep), cob95_t_hc2(nrep), cob99_t_hc2(nrep),	
		 cob90_t_hc3(nrep), cob95_t_hc3(nrep), cob99_t_hc3(nrep),
		 cob90_t_hc4(nrep), cob95_t_hc4(nrep), cob99_t_hc4(nrep),
		 cob90_t_hc5(nrep), cob95_t_hc5(nrep), cob99_t_hc5(nrep);

	vec ncobesq90_t_ols(nrep), ncobesq95_t_ols(nrep), ncobesq99_t_ols(nrep),
		 ncobesq90_t_hc0(nrep), ncobesq95_t_hc0(nrep), ncobesq99_t_hc0(nrep),
		 ncobesq90_t_hc2(nrep), ncobesq95_t_hc2(nrep), ncobesq99_t_hc2(nrep),
		 ncobesq90_t_hc3(nrep), ncobesq95_t_hc3(nrep), ncobesq99_t_hc3(nrep),
		 ncobesq90_t_hc4(nrep), ncobesq95_t_hc4(nrep), ncobesq99_t_hc4(nrep),
		 ncobesq90_t_hc5(nrep), ncobesq95_t_hc5(nrep), ncobesq99_t_hc5(nrep);

	vec ncobdi90_t_ols(nrep), ncobdi95_t_ols(nrep), ncobdi99_t_ols(nrep),
		 ncobdi90_t_hc0(nrep), ncobdi95_t_hc0(nrep), ncobdi99_t_hc0(nrep),
		 ncobdi90_t_hc2(nrep), ncobdi95_t_hc2(nrep), ncobdi99_t_hc2(nrep),
		 ncobdi90_t_hc3(nrep), ncobdi95_t_hc3(nrep), ncobdi99_t_hc3(nrep),
		 ncobdi90_t_hc4(nrep), ncobdi95_t_hc4(nrep), ncobdi99_t_hc4(nrep),
		 ncobdi90_t_hc5(nrep), ncobdi95_t_hc5(nrep), ncobdi99_t_hc5(nrep);

	vec cob90_z_ols(nrep), cob95_z_ols(nrep), cob99_z_ols(nrep),
		 cob90_z_hc0(nrep), cob95_z_hc0(nrep), cob99_z_hc0(nrep),
		 cob90_z_hc2(nrep), cob95_z_hc2(nrep), cob99_z_hc2(nrep),	
		 cob90_z_hc3(nrep), cob95_z_hc3(nrep), cob99_z_hc3(nrep),
		 cob90_z_hc4(nrep), cob95_z_hc4(nrep), cob99_z_hc4(nrep),
		 cob90_z_hc5(nrep), cob95_z_hc5(nrep), cob99_z_hc5(nrep);

	vec ncobesq90_z_ols(nrep), ncobesq95_z_ols(nrep), ncobesq99_z_ols(nrep),
		 ncobesq90_z_hc0(nrep), ncobesq95_z_hc0(nrep), ncobesq99_z_hc0(nrep),
		 ncobesq90_z_hc2(nrep), ncobesq95_z_hc2(nrep), ncobesq99_z_hc2(nrep),
		 ncobesq90_z_hc3(nrep), ncobesq95_z_hc3(nrep), ncobesq99_z_hc3(nrep),
		 ncobesq90_z_hc4(nrep), ncobesq95_z_hc4(nrep), ncobesq99_z_hc4(nrep),
		 ncobesq90_z_hc5(nrep), ncobesq95_z_hc5(nrep), ncobesq99_z_hc5(nrep);

	vec ncobdi90_z_ols(nrep), ncobdi95_z_ols(nrep), ncobdi99_z_ols(nrep),
		 ncobdi90_z_hc0(nrep), ncobdi95_z_hc0(nrep), ncobdi99_z_hc0(nrep),
		 ncobdi90_z_hc2(nrep), ncobdi95_z_hc2(nrep), ncobdi99_z_hc2(nrep),
		 ncobdi90_z_hc3(nrep), ncobdi95_z_hc3(nrep), ncobdi99_z_hc3(nrep),
		 ncobdi90_z_hc4(nrep), ncobdi95_z_hc4(nrep), ncobdi99_z_hc4(nrep),
		 ncobdi90_z_hc5(nrep), ncobdi95_z_hc5(nrep), ncobdi99_z_hc5(nrep);

	vec ampl90_t_ols(nrep), ampl95_t_ols(nrep),  ampl99_t_ols(nrep),
		 ampl90_t_hc0(nrep), ampl95_t_hc0(nrep),  ampl99_t_hc0(nrep),
		 ampl90_t_hc2(nrep), ampl95_t_hc2(nrep),  ampl99_t_hc2(nrep),
		 ampl90_t_hc3(nrep), ampl95_t_hc3(nrep),  ampl99_t_hc3(nrep),
		 ampl90_t_hc4(nrep), ampl95_t_hc4(nrep),  ampl99_t_hc4(nrep),
		 ampl90_t_hc5(nrep), ampl95_t_hc5(nrep),  ampl99_t_hc5(nrep);

	vec ampl90_z_ols(nrep), ampl95_z_ols(nrep),  ampl99_z_ols(nrep),
		 ampl90_z_hc0(nrep), ampl95_z_hc0(nrep),  ampl99_z_hc0(nrep),
		 ampl90_z_hc2(nrep), ampl95_z_hc2(nrep),  ampl99_z_hc2(nrep),
		 ampl90_z_hc3(nrep), ampl95_z_hc3(nrep),  ampl99_z_hc3(nrep),
		 ampl90_z_hc4(nrep), ampl95_z_hc4(nrep),  ampl99_z_hc4(nrep),
		 ampl90_z_hc5(nrep), ampl95_z_hc5(nrep),  ampl99_z_hc5(nrep);

	// UTILIZADO NA GERACAO DO VALOR DE t^*.
	double numero;

	// AQUI COMECA O LACO DE MONTE CARLO.
	//#pragma opm paralell for
	for(int i=0;i<nrep;i++){
		if(dist_erro==1){
			for(int v=0;v<nobs;v++){
				Y(v) = eta(v)+sigma(v)*gsl_ran_gaussian(r,1.0);
			}
		}
		if(dist_erro==2){
			for(int v=0;v<nobs;v++){
				Y(v) = eta(v)+sigma(v)*(gsl_ran_tdist(r,3)/sqrt(1.5));
			}
		}

		if(dist_erro==3){
			for(int v=0;v<nobs;v++){
				Y(v) = eta(v)+sigma(v)*(gsl_ran_chisq(r,2)-2.0)/2.0;
			}
		}

		if(dist_erro==4){
			for(int v=0;v<nobs;v++){
				Y(v) = eta(v)+sigma(v)*(gsl_ran_weibull(r,2,3)-1.785959)/0.6491006;
			}
		}

		if(dist_erro==5){
			for(int v=0;v<nobs;v++){
				Y(v) = eta(v)+sigma(v)*(gsl_ran_gumbel2(r,2.5,2)-1.965001)/2.032706;
			}
		}

		// DISTRIBUICAO GAMMA PARA OS ERROS USANDO O ALGORITMO DE KNUTH.	
		if(dist_erro==6){
			for(int v=0;v<nobs;v++){
				Y(v) = eta(v)+sigma(v)*(gsl_ran_gamma_knuth(r,2.5,2)-1.550078)/1.052454;
			}
		}

		mat temp = produtos*Y;
		mat resid2 = pow((Y-X*temp),2.0); // EPSILON AO QUADRADO. 
		mat omega0 = diagmat(resid2%weight0); // MATRIZ OMEGA ESTIMADO. É UMA MATRIZ DIAGONAL N POR N.
		mat omega2 = diagmat(resid2%weight2); // MATRIZ OMEGA ESTIMADO. É UMA MATRIZ DIAGONAL N POR N.
		mat omega3 = diagmat(resid2%weight3); // MATRIZ OMEGA ESTIMADO. É UMA MATRIZ DIAGONAL N POR N.
		mat omega4 = diagmat(resid2%weight4); // MATRIZ OMEGA ESTIMADO. É UMA MATRIZ DIAGONAL N POR N.
		mat omega5 = diagmat(resid2%weight5); // MATRIZ OMEGA ESTIMADO. É UMA MATRIZ DIAGONAL N POR N.

		mat HC0 = P*omega0*Pt;
		mat HC2 = P*omega2*Pt;
		mat HC3 = P*omega3*Pt;
		mat HC4 = P*omega4*Pt;
		mat HC5 = P*omega5*Pt;

		vec diagonal_hc4 = diagvec(HC4);
		double variancia_maxima = diagonal_hc4.max();
		double variancia_minima = diagonal_hc4.min();

		double OLS = as_scalar(sum(resid2))/(nobs-2) * C(1,1);		

		//epsilon_chapeu = Y-X*temp; // ESTIMATIVAS DOS ERROS.
		epsilon_chapeu = Y-X*temp; // ESTIMATIVAS DOS ERROS.

		vec hc0_b(nrep_boot), hc2_b(nrep_boot), hc3_b(nrep_boot), hc4_b(nrep_boot),
			 hc5_b(nrep_boot), u_estrela(nrep_boot), Z0_j(nrep_boot), Z2_j(nrep_boot),
			 Z3_j(nrep_boot), Z4_j(nrep_boot), Z5_j(nrep_boot);

		vec hc0_duplo(nrep_boot_duplo), hc2_duplo(nrep_boot_duplo), hc3_duplo(nrep_boot_duplo),
			 hc4_duplo(nrep_boot_duplo), hc5_duplo(nrep_boot_duplo);

      // CALCULO DAS QUANTIDADES PIVOTAIS CALCULADAS (\hat{\beta_j} - 1)/\sqrt{HCk(1,1)}.
		// ESSES PIVOS SERAO UTILIZADOS PARA VARIFICAR O AJUSTAMENTO DESSES VALORES A UMA
		// DISTRIBUICAO t(n-p) OU A UMA DISTRIBUICAO NORMAL PADRAO. COM ESSES VALORES PODERA
		// SER CALCULADO AS ESTATISTICAS DE CRAMÉR-VON MISSES E ANDERSON DARLING BEM COMO
		// CONSTRUIR QQ-PLOTS OU PP-PLOTS.

      double pivools, pivohc0, pivohc2, pivohc3, pivohc4, pivohc5;

   	pivools = (temp(1) - 1)/sqrt(OLS);
    	pivohc0 = (temp(1) - 1)/sqrt(HC0(1,1));
    	pivohc2 = (temp(1) - 1)/sqrt(HC2(1,1));
		pivohc3 = (temp(1) - 1)/sqrt(HC3(1,1));
		pivohc4 = (temp(1) - 1)/sqrt(HC4(1,1));
		pivohc5 = (temp(1) - 1)/sqrt(HC5(1,1));

      pivo_ols << pivools << endl;
      pivo_hc0 << pivohc0 << endl;
      pivo_hc2 << pivohc2 << endl;
      pivo_hc3 << pivohc3 << endl;
		pivo_hc4 << pivohc4 << endl;
		pivo_hc5 << pivohc5 << endl;

		// INTERVALOS PARA QUANTIL DE UMA DISTRIBUICAO T-STUDENT COM n-p 
		// GRAUS DE LIBERDADE. AVALIACAO DOS INTERVALOS SEM UTILIZAR BOOTSTRAP. 
		// CONFIANCA DE 90%

		double li = temp(1) - vc_t10*sqrt(OLS);
		double ls = temp(1) + vc_t10*sqrt(OLS);

		if(li<=beta(1) && beta(1)<=ls)
			cob90_t_ols(i) = 1;
		else
			cob90_t_ols(i) = 0;

		if(beta(1)<li)
			ncobesq90_t_ols(i) = 1;
		else
			ncobesq90_t_ols(i) = 0;

		if(beta(1)>ls)
			ncobdi90_t_ols(i) = 1;
		else
			ncobdi90_t_ols(i) = 0;
		ampl90_t_ols(i) = ls - li;

		li = temp(1) - vc_t10*sqrt(HC0(1,1));
		ls = temp(1) + vc_t10*sqrt(HC0(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob90_t_hc0(i) = 1;
		else
			cob90_t_hc0(i) = 0;

		if(beta(1)<li)
			ncobesq90_t_hc0(i) = 1;
		else
			ncobesq90_t_hc0(i) = 0;

		if(beta(1)>ls)
			ncobdi90_t_hc0(i) = 1;
		else
			ncobdi90_t_hc0(i) = 0;
		ampl90_t_hc0(i) = ls - li;

		li = temp(1) - vc_t10*sqrt(HC2(1,1));
		ls = temp(1) + vc_t10*sqrt(HC2(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob90_t_hc2(i) = 1;
		else
			cob90_t_hc2(i) = 0;

		if(beta(1)<li)
			ncobesq90_t_hc2(i) = 1;
		else
			ncobesq90_t_hc2(i) = 0;

		if(beta(1)>ls)
			ncobdi90_t_hc2(i) = 1;
		else
			ncobdi90_t_hc2(i) = 0;
		ampl90_t_hc2(i) = ls - li;

		li = temp(1) - vc_t10*sqrt(HC3(1,1));
		ls = temp(1) + vc_t10*sqrt(HC3(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob90_t_hc3(i) = 1;
		else
			cob90_t_hc3(i) = 0;

		if(beta(1)<li)
			ncobesq90_t_hc3(i) = 1;
		else
			ncobesq90_t_hc3(i) = 0;

		if(beta(1)>ls)
			ncobdi90_t_hc3(i) = 1;
		else
			ncobdi90_t_hc3(i) = 0;
		ampl90_t_hc3(i) = ls - li;

		li = temp(1) - vc_t10*sqrt(HC4(1,1));
		ls = temp(1) + vc_t10*sqrt(HC4(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob90_t_hc4(i) = 1;
		else
			cob90_t_hc4(i) = 0;

		if(beta(1)<li)
			ncobesq90_t_hc4(i) = 1;
		else
			ncobesq90_t_hc4(i) = 0;

		if(beta(1)>ls)
			ncobdi90_t_hc4(i) = 1;
		else
			ncobdi90_t_hc4(i) = 0;
		ampl90_t_hc4(i) = ls - li;

		li = temp(1) - vc_t10*sqrt(HC5(1,1));
		ls = temp(1) + vc_t10*sqrt(HC5(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob90_t_hc5(i) = 1;
		else
			cob90_t_hc5(i) = 0;

		if(beta(1)<li)
			ncobesq90_t_hc5(i) = 1;
		else
			ncobesq90_t_hc5(i) = 0;

		if(beta(1)>ls)
			ncobdi90_t_hc5(i) = 1;
		else
			ncobdi90_t_hc5(i) = 0;
		ampl90_t_hc5(i) = ls - li;

		// CONFIANCA DE 95%
      	
		li = temp(1) - vc_t5*sqrt(OLS);
		ls = temp(1) + vc_t5*sqrt(OLS);

      lils_ols << li << endl;
		lils_ols << ls << endl;

		if(li<=beta(1) && beta(1)<=ls)
			cob95_t_ols(i) = 1;
		else
			cob95_t_ols(i) = 0;

		if(beta(1)<li)
			ncobesq95_t_ols(i) = 1;
		else
			ncobesq95_t_ols(i) = 0;

		if(beta(1)>ls)
			ncobdi95_t_ols(i) = 1;
		else
			ncobdi95_t_ols(i) = 0;
		ampl95_t_ols(i) = ls - li;

		li = temp(1) - vc_t5*sqrt(HC0(1,1));
		ls = temp(1) + vc_t5*sqrt(HC0(1,1));
      
      lils_hc0 << li << endl;
		lils_hc0 << ls << endl;

		if(li<=beta(1) && beta(1)<=ls)
			cob95_t_hc0(i) = 1;
		else
			cob95_t_hc0(i) = 0;

		if(beta(1)<li)
			ncobesq95_t_hc0(i) = 1;
		else
			ncobesq95_t_hc0(i) = 0;

		if(beta(1)>ls)
			ncobdi95_t_hc0(i) = 1;
		else
			ncobdi95_t_hc0(i) = 0;
		ampl95_t_hc0(i) = ls - li;

		li = temp(1) - vc_t5*sqrt(HC2(1,1));
		ls = temp(1) + vc_t5*sqrt(HC2(1,1));

      lils_hc2 << li << endl;
		lils_hc2 << ls << endl;

		if(li<=beta(1) && beta(1)<=ls)
			cob95_t_hc2(i) = 1;
		else
			cob95_t_hc2(i) = 0;

		if(beta(1)<li)
			ncobesq95_t_hc2(i) = 1;
		else
			ncobesq95_t_hc2(i) = 0;

		if(beta(1)>ls)
			ncobdi95_t_hc2(i) = 1;
		else
			ncobdi95_t_hc2(i) = 0;
		ampl95_t_hc2(i) = ls - li;

		li = temp(1) - vc_t5*sqrt(HC3(1,1));
		ls = temp(1) + vc_t5*sqrt(HC3(1,1));

      lils_hc3 << li << endl;
		lils_hc3 << ls << endl;

		if(li<=beta(1) && beta(1)<=ls)
			cob95_t_hc3(i) = 1;
		else
			cob95_t_hc3(i) = 0;

		if(beta(1)<li)
			ncobesq95_t_hc3(i) = 1;
		else
			ncobesq95_t_hc3(i) = 0;

		if(beta(1)>ls)
			ncobdi95_t_hc3(i) = 1;
		else
			ncobdi95_t_hc3(i) = 0;
		ampl95_t_hc3(i) = ls - li;

		li = temp(1) - vc_t5*sqrt(HC4(1,1));
		ls = temp(1) + vc_t5*sqrt(HC4(1,1));

      lils_hc4 << li << endl;
		lils_hc4 << ls << endl;

		if(li<=beta(1) && beta(1)<=ls)
			cob95_t_hc4(i) = 1;
		else
			cob95_t_hc4(i) = 0;

		if(beta(1)<li)
			ncobesq95_t_hc4(i) = 1;
		else
			ncobesq95_t_hc4(i) = 0;

		if(beta(1)>ls)
			ncobdi95_t_hc4(i) = 1;
		else
			ncobdi95_t_hc4(i) = 0;
		ampl95_t_hc4(i) = ls - li;

		li = temp(1) - vc_t5*sqrt(HC5(1,1));
		ls = temp(1) + vc_t5*sqrt(HC5(1,1));

      lils_hc5 << li << endl;
		lils_hc5 << ls << endl;

		if(li<=beta(1) && beta(1)<=ls)
			cob95_t_hc5(i) = 1;
		else
			cob95_t_hc5(i) = 0;

		if(beta(1)<li)
			ncobesq95_t_hc5(i) = 1;
		else
			ncobesq95_t_hc5(i) = 0;

		if(beta(1)>ls)
			ncobdi95_t_hc5(i) = 1;
		else
			ncobdi95_t_hc5(i) = 0;
		ampl95_t_hc5(i) = ls -li;

		// CONFIANCA DE 99%

		li = temp(1) - vc_t1*sqrt(OLS);
		ls = temp(1) + vc_t1*sqrt(OLS);

		if(li<=beta(1) && beta(1)<=ls)
			cob99_t_ols(i) = 1;
		else
			cob99_t_ols(i) = 0;

		if(beta(1)<li)
			ncobesq99_t_ols(i) = 1;
		else
			ncobesq99_t_ols(i) = 0;

		if(beta(1)>ls)
			ncobdi99_t_ols(i) = 1;
		else
			ncobdi99_t_ols(i) = 0;
		ampl99_t_ols(i) = ls - li;

		li = temp(1) - vc_t1*sqrt(HC0(1,1));
		ls = temp(1) + vc_t1*sqrt(HC0(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob99_t_hc0(i) = 1;
		else
			cob99_t_hc0(i) = 0;

		if(beta(1)<li)
			ncobesq99_t_hc0(i) = 1;
		else
			ncobesq99_t_hc0(i) = 0;

		if(beta(1)>ls)
			ncobdi99_t_hc0(i) = 1;
		else
			ncobdi99_t_hc0(i) = 0;
		ampl99_t_hc0(i) = ls - li;

		li = temp(1) - vc_t1*sqrt(HC2(1,1));
		ls = temp(1) + vc_t1*sqrt(HC2(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob99_t_hc2(i) = 1;
		else
			cob99_t_hc2(i) = 0;

		if(beta(1)<li)
			ncobesq99_t_hc2(i) = 1;
		else
			ncobesq99_t_hc2(i) = 0;

		if(beta(1)>ls)
			ncobdi99_t_hc2(i) = 1;
		else
			ncobdi99_t_hc2(i) = 0;
		ampl99_t_hc2(i) =  ls - li;

		li = temp(1) - vc_t1*sqrt(HC3(1,1));
		ls = temp(1) + vc_t1*sqrt(HC3(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob99_t_hc3(i) = 1;
		else
			cob99_t_hc3(i) = 0;

		if(beta(1)<li)
			ncobesq99_t_hc3(i) = 1;
		else
			ncobesq99_t_hc3(i) = 0;

		if(beta(1)>ls)
			ncobdi99_t_hc3(i) = 1;
		else
			ncobdi99_t_hc3(i) = 0;
		ampl99_t_hc3(i) = ls - li;

		li = temp(1) - vc_t1*sqrt(HC4(1,1));
		ls = temp(1) + vc_t1*sqrt(HC4(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob99_t_hc4(i) = 1;
		else
			cob99_t_hc4(i) = 0;

		if(beta(1)<li)
			ncobesq99_t_hc4(i) = 1;
		else
			ncobesq99_t_hc4(i) = 0;

		if(beta(1)>ls)
			ncobdi99_t_hc4(i) = 1;
		else
			ncobdi99_t_hc4(i) = 0;
		ampl99_t_hc4(i) = ls - li;

		li = temp(1) - vc_t1*sqrt(HC5(1,1));
		ls = temp(1) + vc_t1*sqrt(HC5(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob99_t_hc5(i) = 1;
		else
			cob99_t_hc5(i) = 0;

		if(beta(1)<li)
			ncobesq99_t_hc5(i) = 1;
		else
			ncobesq99_t_hc5(i) = 0;

		if(beta(1)>ls)
			ncobdi99_t_hc5(i) = 1;
		else
			ncobdi99_t_hc5(i) = 0;
		ampl99_t_hc5(i) = ls - li;

		// INTERVALOS PARA QUANTIL DE UMA DISTRIBUICAO NORMAL PADRAO.
		// AVALIACAO DOS INTERVALOS SEM UTILIZAR BOOTSTRAP. 
		// CONFIANCA DE 90%

		li = temp(1) - vc_z10*sqrt(OLS);
		ls = temp(1) + vc_z10*sqrt(OLS);

		if(li<=beta(1) && beta(1)<=ls)
			cob90_z_ols(i) = 1;
		else
			cob90_z_ols(i) = 0;

		if(beta(1)<li)
			ncobesq90_z_ols(i) = 1;
		else
			ncobesq90_z_ols(i) = 0;

		if(beta(1)>ls)
			ncobdi90_z_ols(i) = 1;
		else
			ncobdi90_z_ols(i) = 0;
		ampl90_z_ols(i) = ls - li;

		li = temp(1) - vc_z10*sqrt(HC0(1,1));
		ls = temp(1) + vc_z10*sqrt(HC0(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob90_z_hc0(i) = 1;
		else
			cob90_z_hc0(i) = 0;

		if(beta(1)<li)
			ncobesq90_z_hc0(i) = 1;
		else
			ncobesq90_z_hc0(i) = 0;

		if(beta(1)>ls)
			ncobdi90_z_hc0(i) = 1;
		else
			ncobdi90_z_hc0(i) = 0;
		ampl90_z_hc0(i) = ls - li;

		li = temp(1) - vc_z10*sqrt(HC2(1,1));
		ls = temp(1) + vc_z10*sqrt(HC2(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob90_z_hc2(i) = 1;
		else
			cob90_z_hc2(i) = 0;

		if(beta(1)<li)
			ncobesq90_z_hc2(i) = 1;
		else
			ncobesq90_z_hc2(i) = 0;

		if(beta(1)>ls)
			ncobdi90_z_hc2(i) = 1;
		else
			ncobdi90_z_hc2(i) = 0;
		ampl90_z_hc2(i) = ls - li;

		li = temp(1) - vc_z10*sqrt(HC3(1,1));
		ls = temp(1) + vc_z10*sqrt(HC3(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob90_z_hc3(i) = 1;
		else
			cob90_z_hc3(i) = 0;

		if(beta(1)<li)
			ncobesq90_z_hc3(i) = 1;
		else
			ncobesq90_z_hc3(i) = 0;

		if(beta(1)>ls)
			ncobdi90_z_hc3(i) = 1;
		else
			ncobdi90_z_hc3(i) = 0;
		ampl90_z_hc3(i) = ls - li;

		li = temp(1) - vc_z10*sqrt(HC4(1,1));
		ls = temp(1) + vc_z10*sqrt(HC4(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob90_z_hc4(i) = 1;
		else
			cob90_z_hc4(i) = 0;

		if(beta(1)<li)
			ncobesq90_z_hc4(i) = 1;
		else
			ncobesq90_z_hc4(i) = 0;

		if(beta(1)>ls)
			ncobdi90_z_hc4(i) = 1;
		else
			ncobdi90_z_hc4(i) = 0;
		ampl90_z_hc4(i) = ls - li;

		li = temp(1) - vc_z10*sqrt(HC5(1,1));
		ls = temp(1) + vc_z10*sqrt(HC5(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob90_z_hc5(i) = 1;
		else
			cob90_z_hc5(i) = 0;

		if(beta(1)<li)
			ncobesq90_z_hc5(i) = 1;
		else
			ncobesq90_z_hc5(i) = 0;

		if(beta(1)>ls)
			ncobdi90_z_hc5(i) = 1;
		else
			ncobdi90_z_hc5(i) = 0;
		ampl90_z_hc5(i) = ls - li;

		// CONFIANCA DE 95%

		li = temp(1) - vc_z5*sqrt(OLS);
		ls = temp(1) + vc_z5*sqrt(OLS);

		if(li<=beta(1) && beta(1)<=ls)
			cob95_z_ols(i) = 1;
		else
			cob95_z_ols(i) = 0;

		if(beta(1)<li)
			ncobesq95_z_ols(i) = 1;
		else
			ncobesq95_z_ols(i) = 0;

		if(beta(1)>ls)
			ncobdi95_z_ols(i) = 1;
		else
			ncobdi95_z_ols(i) = 0;
		ampl95_z_ols(i) = ls - li;

		li = temp(1) - vc_z5*sqrt(HC0(1,1));
		ls = temp(1) + vc_z5*sqrt(HC0(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob95_z_hc0(i) = 1;
		else
			cob95_z_hc0(i) = 0;

		if(beta(1)<li)
			ncobesq95_z_hc0(i) = 1;
		else
			ncobesq95_z_hc0(i) = 0;

		if(beta(1)>ls)
			ncobdi95_z_hc0(i) = 1;
		else
			ncobdi95_z_hc0(i) = 0;
		ampl95_z_hc0(i) = ls - li;

		li = temp(1) - vc_z5*sqrt(HC2(1,1));
		ls = temp(1) + vc_z5*sqrt(HC2(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob95_z_hc2(i) = 1;
		else
			cob95_z_hc2(i) = 0;

		if(beta(1)<li)
			ncobesq95_z_hc2(i) = 1;
		else
			ncobesq95_z_hc2(i) = 0;

		if(beta(1)>ls)
			ncobdi95_z_hc2(i) = 1;
		else
			ncobdi95_z_hc2(i) = 0;
		ampl95_z_hc2(i) = ls - li;

		li = temp(1) - vc_z5*sqrt(HC3(1,1));
		ls = temp(1) + vc_z5*sqrt(HC3(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob95_z_hc3(i) = 1;
		else
			cob95_z_hc3(i) = 0;

		if(beta(1)<li)
			ncobesq95_z_hc3(i) = 1;
		else
			ncobesq95_z_hc3(i) = 0;

		if(beta(1)>ls)
			ncobdi95_z_hc3(i) = 1;
		else
			ncobdi95_z_hc3(i) = 0;
		ampl95_z_hc3(i) = ls - li;

		li = temp(1) - vc_z5*sqrt(HC4(1,1));
		ls = temp(1) + vc_z5*sqrt(HC4(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob95_z_hc4(i) = 1;
		else
			cob95_z_hc4(i) = 0;

		if(beta(1)<li)
			ncobesq95_z_hc4(i) = 1;
		else
			ncobesq95_z_hc4(i) = 0;

		if(beta(1)>ls)
			ncobdi95_z_hc4(i) = 1;
		else
			ncobdi95_z_hc4(i) = 0;
		ampl95_z_hc4(i) = ls - li;

		li = temp(1) - vc_z5*sqrt(HC5(1,1));
		ls = temp(1) + vc_z5*sqrt(HC5(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob95_z_hc5(i) = 1;
		else
			cob95_z_hc5(i) = 0;

		if(beta(1)<li)
			ncobesq95_z_hc5(i) = 1;
		else
			ncobesq95_z_hc5(i) = 0;

		if(beta(1)>ls)
			ncobdi95_z_hc5(i) = 1;
		else
			ncobdi95_z_hc5(i) = 0;
		ampl95_z_hc5(i) = ls -li;

		// CONFIANCA DE 99%

		li = temp(1) - vc_z1*sqrt(OLS);
		ls = temp(1) + vc_z1*sqrt(OLS);

		if(li<=beta(1) && beta(1)<=ls)
			cob99_z_ols(i) = 1;
		else
			cob99_z_ols(i) = 0;

		if(beta(1)<li)
			ncobesq99_z_ols(i) = 1;
		else
			ncobesq99_z_ols(i) = 0;

		if(beta(1)>ls)
			ncobdi99_z_ols(i) = 1;
		else
			ncobdi99_z_ols(i) = 0;
		ampl99_z_ols(i) = ls - li;

		li = temp(1) - vc_z1*sqrt(HC0(1,1));
		ls = temp(1) + vc_z1*sqrt(HC0(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob99_z_hc0(i) = 1;
		else
			cob99_z_hc0(i) = 0;

		if(beta(1)<li)
			ncobesq99_z_hc0(i) = 1;
		else
			ncobesq99_z_hc0(i) = 0;

		if(beta(1)>ls)
			ncobdi99_z_hc0(i) = 1;
		else
			ncobdi99_z_hc0(i) = 0;
		ampl99_z_hc0(i) = ls - li;

		li = temp(1) - vc_z1*sqrt(HC2(1,1));
		ls = temp(1) + vc_z1*sqrt(HC2(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob99_z_hc2(i) = 1;
		else
			cob99_z_hc2(i) = 0;

		if(beta(1)<li)
			ncobesq99_z_hc2(i) = 1;
		else
			ncobesq99_z_hc2(i) = 0;

		if(beta(1)>ls)
			ncobdi99_z_hc2(i) = 1;
		else
			ncobdi99_z_hc2(i) = 0;
		ampl99_z_hc2(i) =  ls - li;

		li = temp(1) - vc_z1*sqrt(HC3(1,1));
		ls = temp(1) + vc_z1*sqrt(HC3(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob99_z_hc3(i) = 1;
		else
			cob99_z_hc3(i) = 0;

		if(beta(1)<li)
			ncobesq99_z_hc3(i) = 1;
		else
			ncobesq99_z_hc3(i) = 0;

		if(beta(1)>ls)
			ncobdi99_z_hc3(i) = 1;
		else
			ncobdi99_z_hc3(i) = 0;
		ampl99_z_hc3(i) = ls - li;

		li = temp(1) - vc_z1*sqrt(HC4(1,1));
		ls = temp(1) + vc_z1*sqrt(HC4(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob99_z_hc4(i) = 1;
		else
			cob99_z_hc4(i) = 0;

		if(beta(1)<li)
			ncobesq99_z_hc4(i) = 1;
		else
			ncobesq99_z_hc4(i) = 0;

		if(beta(1)>ls)
			ncobdi99_z_hc4(i) = 1;
		else
			ncobdi99_z_hc4(i) = 0;
		ampl99_z_hc4(i) = ls - li;

		li = temp(1) - vc_z1*sqrt(HC5(1,1));
		ls = temp(1) + vc_z1*sqrt(HC5(1,1));

		if(li<=beta(1) && beta(1)<=ls)
			cob99_z_hc5(i) = 1;
		else
			cob99_z_hc5(i) = 0;

		if(beta(1)<li)
			ncobesq99_z_hc5(i) = 1;
		else
			ncobesq99_z_hc5(i) = 0;

		if(beta(1)>ls)
			ncobdi99_z_hc5(i) = 1;
		else
			ncobdi99_z_hc5(i) = 0;
		ampl99_z_hc5(i) = ls - li;

		u_estrela.zeros();
		double u_estrela_numerador = 0, contador0_Z_j = 0, contador2_Z_j = 0,
				 contador3_Z_j = 0, contador4_Z_j = 0, contador5_Z_j = 0;

		mat Xtemp = X*temp;

		// AQUI COMECA O LACO BOOTSTRAP. 
		// A PARTIR DESSE PONTO SERAO CONSTRUIDOS ESQUEMAS BOOTSTRAP E BOOTSTRAP
		// DUPLO PARA GERACAO DE INTERVALOS DE CONFIANCAS MAIS PRECISOS.
		for(int k=0;k<nrep_boot;k++){
			u_estrela_numerador = 0;
			vec y_estrela(nobs), t_estrela(nobs);	// VARIAVEL RESPOSTA UTILIZADA NO BOOTSTRAP.
			// NUMERO ALEATORIO COM MEDIA ZERO E VARINCIA UM.
			if(dist_t==2){
				for(int t=0;t<nobs;t++){
					numero = gsl_ran_gaussian(r,1.0);
					t_estrela(t) = numero;
				}
			}

			if(dist_t==1){
				for(int t=0;t<nobs;t++){
					numero = gsl_rng_uniform(r);
					if(numero <= 0.5)
						t_estrela(t) = -1;
					if(numero > 0.5)
						t_estrela(t) = 1;
				}
				//media_t_estrela << mean(t_estrela) << endl;

			}

		y_estrela = Xtemp+t_estrela%epsilon_chapeu/sqrt(1-h); // CONFERIDO.
			
			// cout << gsl_rng_uniform_int(r,nobs) << endl;
			// AQUI TEMOS AS ESTIMATIVAS DE \hat{{\beta^{*}}_j}. LEMBRANDO QUE NOSSO INTERESSE EH \hat{{\beta^{*}}_2}

		beta_chapeu_boot = produtos*y_estrela;	// ESTIMATIVA DOS BETAS ESTRELA (BOOTSTRAP). \hat{beta^{*}}
		beta2_chapeu_boot_temp(k) = as_scalar(beta_chapeu_boot(1));

		mat Xtemp_b = X*beta_chapeu_boot;
		mat resid2_b = pow((y_estrela-X*beta_chapeu_boot),2.0);
		mat omega0 = diagmat(resid2_b%weight0); // MATRIZ OMEGA ESTIMADA. É UMA MATRIZ DIAGONAL N POR N.
		mat omega2 = diagmat(resid2_b%weight2); // MATRIZ OMEGA ESTIMADA. É UMA MATRIZ DIAGONAL N POR N.
		mat omega3 = diagmat(resid2_b%weight3); // MATRIZ OMEGA ESTIMADA. É UMA MATRIZ DIAGONAL N POR N.
		mat omega4 = diagmat(resid2_b%weight4); // MATRIZ OMEGA ESTIMADA. É UMA MATRIZ DIAGONAL N POR N.
		mat omega5 = diagmat(resid2_b%weight5); // MATRIZ OMEGA ESTIMADA. É UMA MATRIZ DIAGONAL N POR N.

		mat HC0_b = P*omega0*Pt;
		mat HC2_b = P*omega2*Pt;
		mat HC3_b = P*omega3*Pt;
		mat HC4_b = P*omega4*Pt;
		mat HC5_b = P*omega5*Pt;

		hc0_b(k) = sqrt(as_scalar(HC0_b(1,1)));
		hc2_b(k) = sqrt(as_scalar(HC2_b(1,1)));
		hc3_b(k) = sqrt(as_scalar(HC3_b(1,1)));
		hc4_b(k) = sqrt(as_scalar(HC4_b(1,1)));
		hc5_b(k) = sqrt(as_scalar(HC5_b(1,1)));
                
                if(pivo==1){
			z_estrela0(k) = (as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/sqrt(HC0_b(1,1));
			z_estrela2(k) = (as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/sqrt(HC2_b(1,1));
			z_estrela3(k) = (as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/sqrt(HC3_b(1,1));
			z_estrela4(k) = (as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/sqrt(HC4_b(1,1));
			z_estrela5(k) = (as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/sqrt(HC5_b(1,1));
                }

                if(pivo==2){
			z_estrela0(k) = abs(as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/sqrt(HC0_b(1,1));
			z_estrela2(k) = abs(as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/sqrt(HC2_b(1,1));
			z_estrela3(k) = abs(as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/sqrt(HC3_b(1,1));
			z_estrela4(k) = abs(as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/sqrt(HC4_b(1,1));
			z_estrela5(k) = abs(as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/sqrt(HC5_b(1,1));
                }

		beta2(k) = beta_chapeu_boot(1); // BETA2 DA REPLICA DE BOOTSTRAP.
		epsilon_chapeu_boot_duplo = y_estrela-X*beta_chapeu_boot; // SERA UTILIZADO NO BOOTSTRAP DUPLO.

		// VETOR QUE IRA ARMAZENAR AS ESTIMATIVAS HC DO BOOTSTRAP DUPLO QUE
		// SERA UTILIZADO PARA CORRIGIR O ERRO PADRAO DO BOOTSTRAP EXTERIOR.
		vec hc_duplo0(nrep_boot_duplo);
		vec hc_duplo2(nrep_boot_duplo);
		vec hc_duplo3(nrep_boot_duplo);
		vec hc_duplo4(nrep_boot_duplo);
		vec hc_duplo5(nrep_boot_duplo);

		double desvio0_b = sqrt(as_scalar(HC0_b(1,1)));
		double desvio2_b = sqrt(as_scalar(HC2_b(1,1)));
		double desvio3_b = sqrt(as_scalar(HC3_b(1,1)));
		double desvio4_b = sqrt(as_scalar(HC4_b(1,1)));
		double desvio5_b = sqrt(as_scalar(HC5_b(1,1)));

		contador0_Z_j = 0;
		contador2_Z_j = 0;
		contador3_Z_j = 0;
		contador4_Z_j = 0;
		contador5_Z_j = 0;
      
		// AQUI COMECA O BOOTSTRAP DUPLO.
		//#pragma omp parallel for
		for (int m=0;m<nrep_boot_duplo;m++){

			vec y_estrela_estrela(nobs);	// VARIAVEL RESPOSTA DENTRO DO BOOTSTRAP DUPLO.
			vec t_estrela_estrela(nobs);	// NUMERO ALEATORIO COM MEDIA 0 E VARIANCIA 1.

			if (dist_t==2){
				for (int t=0; t<nobs;t++){
					numero = gsl_ran_gaussian(r,1.0);
					t_estrela_estrela(t) = numero;
				}
			}

			if (dist_t==1){
				for (int t=0;t<nobs;t++){
					numero = gsl_rng_uniform(r);
					if(numero<=0.5)
						t_estrela_estrela(t) = -1;
					if (numero>0.5)
						t_estrela_estrela(t) = 1;
				}
			}

			y_estrela_estrela = Xtemp_b+t_estrela_estrela%epsilon_chapeu_boot_duplo/sqrt(1-h);

			// A VARIAVEL PRODUTOS REFERE-SE A (X'X)^-1X'
			beta_chapeu_boot_duplo = produtos*y_estrela_estrela;

			mat resid2_b_duplo = pow((y_estrela_estrela-X*beta_chapeu_boot_duplo),2.0); // RESIDUO AO QUADRADO.

			mat omega0 = diagmat(resid2_b_duplo%weight0); // MATRIZ OMEGA ESTIMADA. É UMA MATRIZ DIAGONAL N POR N.
			mat omega2 = diagmat(resid2_b_duplo%weight2); // MATRIZ OMEGA ESTIMADA. É UMA MATRIZ DIAGONAL N POR N.
			mat omega3 = diagmat(resid2_b_duplo%weight3); // MATRIZ OMEGA ESTIMADA. É UMA MATRIZ DIAGONAL N POR N.
			mat omega4 = diagmat(resid2_b_duplo%weight4); // MATRIZ OMEGA ESTIMADA. É UMA MATRIZ DIAGONAL N POR N.
			mat omega5 = diagmat(resid2_b_duplo%weight5); // MATRIZ OMEGA ESTIMADA. É UMA MATRIZ DIAGONAL N POR N.

			mat HC0_b_duplo = P*omega0*Pt;
			mat HC2_b_duplo = P*omega2*Pt;
			mat HC3_b_duplo = P*omega3*Pt;
			mat HC4_b_duplo = P*omega4*Pt;
			mat HC5_b_duplo = P*omega5*Pt;

			hc0_duplo(m) = sqrt(as_scalar(HC0_b_duplo(1,1)));
			hc2_duplo(m) = sqrt(as_scalar(HC2_b_duplo(1,1)));
			hc3_duplo(m) = sqrt(as_scalar(HC3_b_duplo(1,1)));
			hc4_duplo(m) = sqrt(as_scalar(HC4_b_duplo(1,1)));
			hc5_duplo(m) = sqrt(as_scalar(HC5_b_duplo(1,1)));

                        if(pivo==1){
				z_estrela_estrela0(m) = (as_scalar(beta_chapeu_boot_duplo(1)-beta_chapeu_boot(1)))/sqrt(HC0_b_duplo(1,1));
				z_estrela_estrela2(m) = (as_scalar(beta_chapeu_boot_duplo(1)-beta_chapeu_boot(1)))/sqrt(HC2_b_duplo(1,1));
				z_estrela_estrela3(m) = (as_scalar(beta_chapeu_boot_duplo(1)-beta_chapeu_boot(1)))/sqrt(HC3_b_duplo(1,1));
				z_estrela_estrela4(m) = (as_scalar(beta_chapeu_boot_duplo(1)-beta_chapeu_boot(1)))/sqrt(HC4_b_duplo(1,1));
				z_estrela_estrela5(m) = (as_scalar(beta_chapeu_boot_duplo(1)-beta_chapeu_boot(1)))/sqrt(HC5_b_duplo(1,1));
                        }


                        if(pivo==2){
				z_estrela_estrela0(m) = abs(as_scalar(beta_chapeu_boot_duplo(1)-beta_chapeu_boot(1)))/sqrt(HC0_b_duplo(1,1));
				z_estrela_estrela2(m) = abs(as_scalar(beta_chapeu_boot_duplo(1)-beta_chapeu_boot(1)))/sqrt(HC2_b_duplo(1,1));
				z_estrela_estrela3(m) = abs(as_scalar(beta_chapeu_boot_duplo(1)-beta_chapeu_boot(1)))/sqrt(HC3_b_duplo(1,1));
				z_estrela_estrela4(m) = abs(as_scalar(beta_chapeu_boot_duplo(1)-beta_chapeu_boot(1)))/sqrt(HC4_b_duplo(1,1));
				z_estrela_estrela5(m) = abs(as_scalar(beta_chapeu_boot_duplo(1)-beta_chapeu_boot(1)))/sqrt(HC5_b_duplo(1,1));
                        }

			if(z_estrela_estrela0(m)<=z_estrela0(k))
				contador0_Z_j = contador0_Z_j+1;
			if(z_estrela_estrela2(m)<=z_estrela2(k))
				contador2_Z_j = contador2_Z_j+1;
			if(z_estrela_estrela3(m)<=z_estrela3(k))
				contador3_Z_j = contador3_Z_j+1;
			if(z_estrela_estrela4(m)<=z_estrela4(k))
				contador4_Z_j = contador4_Z_j+1;
			if(z_estrela_estrela5(m)<=z_estrela5(k))
				contador5_Z_j = contador5_Z_j+1;

			if(beta_chapeu_boot_duplo(1)<=2*beta_chapeu_boot(1)-temp(1))
				u_estrela_numerador = 1+u_estrela_numerador;

		} // AQUI TERMINA O LACO DO BOOTSTRAP DUPLO.

		Z0_j(k) = contador0_Z_j/nrep_boot_duplo;
		Z2_j(k) = contador2_Z_j/nrep_boot_duplo;
		Z3_j(k) = contador3_Z_j/nrep_boot_duplo;
		Z4_j(k) = contador4_Z_j/nrep_boot_duplo;
		Z5_j(k) = contador5_Z_j/nrep_boot_duplo;
		u_estrela(k) = u_estrela_numerador/nrep_boot_duplo;
		betaj_estrela_menos_betaj(k) = beta_chapeu_boot(1)-temp(1);
		// BOOTSTRAP T CORRIGINDO A QUANTIDADE NO DENOMIZADOR DA VARIAVEL z^*(errado). O ESQUEMA CORRETO TAMBEM ESTA
		// SENDO CALCULADO NESSE CODIGO FONTE. ELE FAZ USO DA VARIAVEL Z_j PARA CORRIGIR O QUANTIL CALCULADO SOBRE
		// z^*.
		z0_estrela_duplo(k) = (as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/
			(2*sum(hc0_duplo)/nrep_boot_duplo-desvio0_b);
		z2_estrela_duplo(k) = (as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/
			(2*sum(hc2_duplo)/nrep_boot_duplo-desvio2_b);
		z3_estrela_duplo(k) = (as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/
			(2*sum(hc3_duplo)/nrep_boot_duplo-desvio3_b);
		z4_estrela_duplo(k) = (as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/
			(2*sum(hc4_duplo)/nrep_boot_duplo-desvio4_b);
		z5_estrela_duplo(k) = (as_scalar(beta2_chapeu_boot_temp(k)-temp(1)))/
			(2*sum(hc5_duplo)/nrep_boot_duplo-desvio5_b);
		} // AQUI TERMINA O LACO BOOTSTRAP.

		// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		//                              INTERVALOS PARA 90%
		// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

		// CONTANDO CONVERGÊNCIAS PARA O BOOSTRAP T DUPLO. (ESSE ESQUEMA DE BOOTSTRAP DUPLO NAO É CORRETO,
		// CONTUDO, ESTOU EM CASOS DE HOMOSCEDASTICIDADE. PRECISA SER VERIFICADO.)    

		double quantil0_inferior90 = myfunctions::quantil1(z0_estrela_duplo,0.95,nrep_boot);
		double quantil0_superior90 = myfunctions::quantil1(z0_estrela_duplo,0.05,nrep_boot);

		double quantil2_inferior90 = myfunctions::quantil1(z2_estrela_duplo,0.95,nrep_boot);
		double quantil2_superior90 = myfunctions::quantil1(z2_estrela_duplo,0.05,nrep_boot);

		double quantil3_inferior90 = myfunctions::quantil1(z3_estrela_duplo,0.95,nrep_boot);
		double quantil3_superior90 = myfunctions::quantil1(z3_estrela_duplo,0.05,nrep_boot);

		double quantil4_inferior90 = myfunctions::quantil1(z4_estrela_duplo,0.95,nrep_boot);
		double quantil4_superior90 = myfunctions::quantil1(z4_estrela_duplo,0.05,nrep_boot);

		double quantil5_inferior90 = myfunctions::quantil1(z5_estrela_duplo,0.95,nrep_boot);
		double quantil5_superior90 = myfunctions::quantil1(z5_estrela_duplo,0.05,nrep_boot);

		if(ncorrecoes==2){
			// AQUI ESTAMOS CORRIGINDO O CALCULO DOS LIMITES INFERIORES E SUPERIORES DO INTERVALO DE CONFIANCA.
			// ESSA CORRECAO FAZ USO DO BOOTSTRAP EXTERIOR.
			li_0_90 = temp(1,0)-quantil0_inferior90*(2*sum(hc0_b)/nrep_boot-sqrt(HC0(1,1)));
			ls_0_90 = temp(1,0)-quantil0_superior90*(2*sum(hc0_b)/nrep_boot-sqrt(HC0(1,1)));
			li_2_90 = temp(1,0)-quantil2_inferior90*(2*sum(hc2_b)/nrep_boot-sqrt(HC2(1,1)));
			ls_2_90 = temp(1,0)-quantil2_superior90*(2*sum(hc2_b)/nrep_boot-sqrt(HC2(1,1)));
			li_3_90 = temp(1,0)-quantil3_inferior90*(2*sum(hc3_b)/nrep_boot-sqrt(HC3(1,1)));
			ls_3_90 = temp(1,0)-quantil3_superior90*(2*sum(hc3_b)/nrep_boot-sqrt(HC3(1,1)));
			li_4_90 = temp(1,0)-quantil4_inferior90*(2*sum(hc4_b)/nrep_boot-sqrt(HC4(1,1)));
			ls_4_90 = temp(1,0)-quantil4_superior90*(2*sum(hc4_b)/nrep_boot-sqrt(HC4(1,1)));
			li_5_90 = temp(1,0)-quantil5_inferior90*(2*sum(hc5_b)/nrep_boot-sqrt(HC5(1,1)));
			ls_5_90 = temp(1,0)-quantil5_superior90*(2*sum(hc5_b)/nrep_boot-sqrt(HC5(1,1)));
		}

		if(ncorrecoes==1){
			// AQUI ESTAMOS CONSTRUINDO OS LIMITES DOS INTERVALOS DE CONFIANCAS SEM USAR A CORRECAO DO DESVIO QUE
			// ENTRA NO CALCULO DOS LIMITES.        
			li_0_90 = temp(1,0)-quantil0_inferior90*sqrt(HC0(1,1));
			ls_0_90 = temp(1,0)-quantil0_superior90*sqrt(HC0(1,1));
			li_2_90 = temp(1,0)-quantil2_inferior90*sqrt(HC2(1,1));
			ls_2_90 = temp(1,0)-quantil2_superior90*sqrt(HC2(1,1));
			li_3_90 = temp(1,0)-quantil3_inferior90*sqrt(HC3(1,1));
			ls_3_90 = temp(1,0)-quantil3_superior90*sqrt(HC3(1,1));
			li_4_90 = temp(1,0)-quantil4_inferior90*sqrt(HC4(1,1));
			ls_4_90 = temp(1,0)-quantil4_superior90*sqrt(HC4(1,1));
			li_5_90 = temp(1,0)-quantil5_inferior90*sqrt(HC5(1,1));
			ls_5_90 = temp(1,0)-quantil5_superior90*sqrt(HC5(1,1));
		}

		if(beta(1)>=li_0_90&&beta(1)<=ls_0_90){
			cob_0_90_t_percentil_duplo(i) = 1;
		}else
			cob_0_90_t_percentil_duplo(i) = 0;

		if(beta(1)>=li_2_90&&beta(1)<=ls_2_90){
			cob_2_90_t_percentil_duplo(i) = 1;
		}else
			cob_2_90_t_percentil_duplo(i) = 0;

		if(beta(1)>=li_3_90&&beta(1)<=ls_3_90){
			cob_3_90_t_percentil_duplo(i) = 1;
		}else
			cob_3_90_t_percentil_duplo(i) = 0;

		if(beta(1)>=li_4_90&&beta(1)<=ls_4_90){
			cob_4_90_t_percentil_duplo(i) = 1;
		}else
			cob_4_90_t_percentil_duplo(i) = 0;

		if(beta(1)>=li_5_90&&beta(1)<=ls_5_90){
			cob_5_90_t_percentil_duplo(i) = 1;
		}else
			cob_5_90_t_percentil_duplo(i) = 0;

		if(beta(1)<li_0_90){
			ncobesq_0_90_t_percentil_duplo(i) = 1;
		}else
			ncobesq_0_90_t_percentil_duplo(i) = 0;

		if(beta(1)<li_2_90){
			ncobesq_2_90_t_percentil_duplo(i) = 1;
		}else
			ncobesq_2_90_t_percentil_duplo(i) = 0;

		if(beta(1)<li_3_90){
			ncobesq_3_90_t_percentil_duplo(i) = 1;
		}else
			ncobesq_3_90_t_percentil_duplo(i) = 0;

		if(beta(1)<li_4_90){
			ncobesq_4_90_t_percentil_duplo(i) = 1;
		}else
			ncobesq_4_90_t_percentil_duplo(i) = 0;

		if(beta(1)<li_5_90){
			ncobesq_5_90_t_percentil_duplo(i) = 1;
		}else
			ncobesq_5_90_t_percentil_duplo(i) = 0;

		if(beta(1)>ls_0_90){
			ncobdi_0_90_t_percentil_duplo(i) = 1;
		}else
			ncobdi_0_90_t_percentil_duplo(i) = 0;

		ampl_0_90_t_percentil_duplo(i) = ls_0_90-li_0_90;

		if(beta(1)>ls_2_90){
			ncobdi_2_90_t_percentil_duplo(i) = 1;
		}else
			ncobdi_2_90_t_percentil_duplo(i) = 0;

		ampl_2_90_t_percentil_duplo(i) = ls_2_90-li_2_90;

		if(beta(1)>ls_3_90){
			ncobdi_3_90_t_percentil_duplo(i) = 1;
		}else
			ncobdi_3_90_t_percentil_duplo(i) = 0;

		ampl_3_90_t_percentil_duplo(i) = ls_3_90-li_0_90;

		if(beta(1)>ls_4_90){
			ncobdi_4_90_t_percentil_duplo(i) = 1;
		}else
			ncobdi_4_90_t_percentil_duplo(i) = 0;

		ampl_4_90_t_percentil_duplo(i) = ls_4_90-li_4_90;

		if(beta(1)>ls_5_90){
			ncobdi_5_90_t_percentil_duplo(i) = 1;
		}else
			ncobdi_5_90_t_percentil_duplo(i) = 0;

		ampl_5_90_t_percentil_duplo(i) = ls_5_90-li_5_90;

		// CONTANDO CONVERGÊNCIAS PARA O BOOTSTRAP T (AQUI NÃO É O BOOSTRAP DUPLO.)

		quantil0_inferior90 = myfunctions::quantil1(z_estrela0,0.95,nrep_boot);
		quantil0_superior90 = myfunctions::quantil1(z_estrela0,0.05,nrep_boot);
		li_0_90 = temp(1,0)-quantil0_inferior90*sqrt(HC0(1,1));
		ls_0_90 = temp(1,0)-quantil0_superior90*sqrt(HC0(1,1));

		quantil2_inferior90 = myfunctions::quantil1(z_estrela2,0.95,nrep_boot);
		quantil2_superior90 = myfunctions::quantil1(z_estrela2,0.05,nrep_boot);
		li_2_90 = temp(1,0)-quantil2_inferior90*sqrt(HC2(1,1));
		ls_2_90 = temp(1,0)-quantil2_superior90*sqrt(HC2(1,1));

		quantil3_inferior90 = myfunctions::quantil1(z_estrela3,0.95,nrep_boot);
		quantil3_superior90 = myfunctions::quantil1(z_estrela3,0.05,nrep_boot);
		li_3_90 = temp(1,0)-quantil3_inferior90*sqrt(HC3(1,1));
		ls_3_90 = temp(1,0)-quantil3_superior90*sqrt(HC3(1,1));

		quantil4_inferior90 = myfunctions::quantil1(z_estrela4,0.95,nrep_boot);
		quantil4_superior90 = myfunctions::quantil1(z_estrela4,0.05,nrep_boot);
		li_4_90 = temp(1,0)-quantil4_inferior90*sqrt(HC4(1,1));
		ls_4_90 = temp(1,0)-quantil4_superior90*sqrt(HC4(1,1));

		quantil5_inferior90 = myfunctions::quantil1(z_estrela5,0.95,nrep_boot);
		quantil5_superior90 = myfunctions::quantil1(z_estrela5,0.05,nrep_boot);
		li_5_90 = temp(1,0)-quantil5_inferior90*sqrt(HC5(1,1));
		ls_5_90 = temp(1,0)-quantil5_superior90*sqrt(HC5(1,1));

		if(beta(1)>=li_0_90 && beta(1)<=ls_0_90){
			cob_0_90_t_percentil(i) = 1;
		}else
			cob_0_90_t_percentil(i) = 0;

		if(beta(1)>=li_2_90 && beta(1)<=ls_2_90){
			cob_2_90_t_percentil(i) = 1;
		}else
			cob_2_90_t_percentil(i) = 0;

		if(beta(1)>=li_3_90 && beta(1)<=ls_3_90){
			cob_3_90_t_percentil(i) = 1;
		}else
			cob_3_90_t_percentil(i) = 0;

		if(beta(1)>=li_4_90 && beta(1)<=ls_4_90){
			cob_4_90_t_percentil(i) = 1;
		}else
			cob_4_90_t_percentil(i) = 0;

		if(beta(1)>=li_5_90 && beta(1)<=ls_5_90){
			cob_5_90_t_percentil(i) = 1;
		}else
			cob_5_90_t_percentil(i) = 0;

		if(beta(1)<li_0_90){
			ncobesq_0_90_t_percentil(i) = 1;
		}else
			ncobesq_0_90_t_percentil(i) = 0;

		if(beta(1)<li_2_90){
			ncobesq_2_90_t_percentil(i) = 1;
		}else
			ncobesq_2_90_t_percentil(i) = 0;

		if(beta(1)<li_3_90){
			ncobesq_3_90_t_percentil(i) = 1;
		}else
			ncobesq_3_90_t_percentil(i) = 0;

		if(beta(1)<li_4_90){
			ncobesq_4_90_t_percentil(i) = 1;
		}else
			ncobesq_4_90_t_percentil(i) = 0;

		if(beta(1)<li_5_90){
			ncobesq_5_90_t_percentil(i) = 1;
		}else
			ncobesq_5_90_t_percentil(i) = 0;

		if(beta(1)>ls_0_90){
			ncobdi_0_90_t_percentil(i) = 1;
		}else
			ncobdi_0_90_t_percentil(i) = 0;

		ampl_0_90_t_percentil(i) = ls_0_90-li_0_90;

		if(beta(1)>ls_2_90){
			ncobdi_2_90_t_percentil(i) = 1;
		}else
			ncobdi_2_90_t_percentil(i) = 0;

		ampl_2_90_t_percentil(i) = ls_2_90-li_2_90;

		if(beta(1)>ls_3_90){
			ncobdi_3_90_t_percentil(i) = 1;
		}else
			ncobdi_3_90_t_percentil(i) = 0;

		ampl_3_90_t_percentil(i) = ls_3_90-li_3_90;

		if(beta(1)>ls_4_90){
			ncobdi_4_90_t_percentil(i) = 1;
		}else
			ncobdi_4_90_t_percentil(i) = 0;

		ampl_4_90_t_percentil(i) = ls_4_90-li_4_90;

		if(beta(1)>ls_5_90){
			ncobdi_5_90_t_percentil(i) = 1;
		}else
			ncobdi_5_90_t_percentil(i) = 0;

		ampl_5_90_t_percentil(i) = ls_5_90-li_5_90;

		// INTERVALO BOOTSTRAP PERCENTIL.
		double li90 = myfunctions::quantil1(beta2,0.05,nrep_boot);
		double ls90 = myfunctions::quantil1(beta2,0.95,nrep_boot);
		if(beta(1)>=li90 && beta(1)<=ls90)
			cob90_percentil(i) = 1;
		else
			cob90_percentil(i) = 0;

		ampl90_percentil(i) = ls90-li90;

		if(beta(1)<li90)
			ncobesq90_percentil(i) = 1;
		else
			ncobesq90_percentil(i) = 0;

		if(beta(1)>ls90)
			ncobdi90_percentil(i) = 1;
		else
			ncobdi90_percentil(i) = 0;

		// INTERVALO PERCENTIL BOOTSTRAP DUPLO - 90% (BOOTSTRAP EXTERIOR).
		double hat_ql90 = myfunctions::quantil1(u_estrela,0.05,nrep_boot);
		double hat_qu90 = myfunctions::quantil1(u_estrela,0.95,nrep_boot);
		ls90 = myfunctions::quantil1(beta2,hat_qu90,nrep_boot);
		li90 = myfunctions::quantil1(beta2,hat_ql90,nrep_boot);

		//ls90 = temp(1)-myfunctions::quantil1(betaj_estrela_menos_betaj,hat_ql90,nrep_boot);
		//li90 = temp(1)-myfunctions::quantil1(betaj_estrela_menos_betaj,hat_qu90,nrep_boot);

		ampl90_percentil_duplo(i) = ls90-li90;

		if(li90<=beta(1) && beta(1)<=ls90)
			cob90_percentil_duplo(i) = 1;
		else
			cob90_percentil_duplo(i) = 0;

		if(beta(1)<li90)
			ncobesq90_percentil_duplo(i) = 1;
		else
			ncobesq90_percentil_duplo(i) = 0;

		if(beta(1)>ls90)
			ncobdi90_percentil_duplo(i) = 1;
		else
			ncobdi90_percentil_duplo(i) = 0;

		//INTERVALO BOORSTRAP T DUPLO (CORRETO). BASEADO NO ALGORITMO DAS PAGINAS 84-85 DO ARTIGO:
		//IMPLEMENTING THE DOUBLE BOOTSTRAP, MCCULLOUCH AND VINOD, COMPUTATIONAL ECONOMICS, 1998.

		quantil0_inferior90 = myfunctions::quantil1(z_estrela0, myfunctions::quantil1(Z0_j,0.95,nrep_boot),nrep_boot);
		quantil0_superior90 = myfunctions::quantil1(z_estrela0,myfunctions::quantil1(Z0_j,0.05,nrep_boot),nrep_boot);
		li_0_90 = temp(1,0)-quantil0_inferior90*sqrt(HC0(1,1));
		ls_0_90 = temp(1,0)-quantil0_superior90*sqrt(HC0(1,1));

		quantil2_inferior90 = myfunctions::quantil1(z_estrela2, myfunctions::quantil1(Z2_j,0.95,nrep_boot),nrep_boot);
		quantil2_superior90 = myfunctions::quantil1(z_estrela2,myfunctions::quantil1(Z2_j,0.05,nrep_boot),nrep_boot);
		li_2_90 = temp(1,0)-quantil2_inferior90*sqrt(HC2(1,1));
		ls_2_90 = temp(1,0)-quantil2_superior90*sqrt(HC2(1,1));

		quantil3_inferior90 = myfunctions::quantil1(z_estrela3, myfunctions::quantil1(Z3_j,0.95,nrep_boot),nrep_boot);
		quantil3_superior90 = myfunctions::quantil1(z_estrela3,myfunctions::quantil1(Z3_j,0.05,nrep_boot),nrep_boot);
		li_3_90 = temp(1,0)-quantil3_inferior90*sqrt(HC3(1,1));
		ls_3_90 = temp(1,0)-quantil3_superior90*sqrt(HC3(1,1));

		quantil4_inferior90 = myfunctions::quantil1(z_estrela4, myfunctions::quantil1(Z4_j,0.95,nrep_boot),nrep_boot);
		quantil4_superior90 = myfunctions::quantil1(z_estrela4,myfunctions::quantil1(Z4_j,0.05,nrep_boot),nrep_boot);
		li_4_90 = temp(1,0)-quantil4_inferior90*sqrt(HC4(1,1));
		ls_4_90 = temp(1,0)-quantil4_superior90*sqrt(HC4(1,1));

		quantil5_inferior90 = myfunctions::quantil1(z_estrela5, myfunctions::quantil1(Z5_j,0.95,nrep_boot),nrep_boot);
		quantil5_superior90 = myfunctions::quantil1(z_estrela5,myfunctions::quantil1(Z5_j,0.05,nrep_boot),nrep_boot);
		li_5_90 = temp(1,0)-quantil5_inferior90*sqrt(HC5(1,1));
		ls_5_90 = temp(1,0)-quantil5_superior90*sqrt(HC5(1,1));

		if(beta(1)>=li_0_90 && beta(1)<=ls_0_90){
			cob_0_90_t_percentil_duplo1(i) = 1;
		}else
			cob_0_90_t_percentil_duplo1(i) = 0;

		if(beta(1)>=li_2_90 && beta(1)<=ls_2_90){
			cob_2_90_t_percentil_duplo1(i) = 1;
		}else
			cob_2_90_t_percentil_duplo1(i) = 0;

		if(beta(1)>=li_3_90 && beta(1)<=ls_3_90){
			cob_3_90_t_percentil_duplo1(i) = 1;
		}else
			cob_3_90_t_percentil_duplo1(i) = 0;

		if(beta(1)>=li_4_90 && beta(1)<=ls_4_90){
			cob_4_90_t_percentil_duplo1(i) = 1;
		}else
			cob_4_90_t_percentil_duplo1(i) = 0;

		if(beta(1)>=li_5_90 && beta(1)<=ls_5_90){
			cob_5_90_t_percentil_duplo1(i) = 1;
		}else
			cob_5_90_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_0_90){
			ncobesq_0_90_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_0_90_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_2_90){
			ncobesq_2_90_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_2_90_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_3_90){
			ncobesq_3_90_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_3_90_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_4_90){
			ncobesq_4_90_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_4_90_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_5_90){
			ncobesq_5_90_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_5_90_t_percentil_duplo1(i) = 0;

		if(beta(1)>ls_0_90){
			ncobdi_0_90_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_0_90_t_percentil_duplo1(i) = 0;

		ampl_0_90_t_percentil_duplo1(i) = ls_0_90-li_0_90;

		if(beta(1)>ls_2_90){
			ncobdi_2_90_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_2_90_t_percentil_duplo1(i) = 0;

		ampl_2_90_t_percentil_duplo1(i) = ls_2_90-li_2_90;

		if(beta(1)>ls_3_90){
			ncobdi_3_90_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_3_90_t_percentil_duplo1(i) = 0;

		ampl_3_90_t_percentil_duplo1(i) = ls_3_90-li_3_90;

		if(beta(1)>ls_4_90){
			ncobdi_4_90_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_4_90_t_percentil_duplo1(i) = 0;

		ampl_4_90_t_percentil_duplo1(i) = ls_4_90-li_4_90;

		if(beta(1)>ls_5_90){
			ncobdi_5_90_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_5_90_t_percentil_duplo1(i) = 0;

		ampl_5_90_t_percentil_duplo1(i) = ls_5_90-li_5_90;

		// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		//                              INTERVALOS PARA 95%
		// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

		// CONTANDO CONVERGÊNCIAS PARA O BOOSTRAP T DUPLO. (ESSE ESQUEMA DE BOOTSTRAP DUPLO NAO É CORRETO,
		// CONTUDO, ESTOU EM CASOS DE HOMOSCEDASTICIDADE. PRECISA SER VERIFICADO.)    

		double quantil0_inferior95 = myfunctions::quantil1(z0_estrela_duplo,0.975,nrep_boot);
		double quantil0_superior95 = myfunctions::quantil1(z0_estrela_duplo,0.025,nrep_boot);

		double quantil2_inferior95 = myfunctions::quantil1(z2_estrela_duplo,0.975,nrep_boot);
		double quantil2_superior95 = myfunctions::quantil1(z2_estrela_duplo,0.025,nrep_boot);

		double quantil3_inferior95 = myfunctions::quantil1(z3_estrela_duplo,0.975,nrep_boot);
		double quantil3_superior95 = myfunctions::quantil1(z3_estrela_duplo,0.025,nrep_boot);

		double quantil4_inferior95 = myfunctions::quantil1(z4_estrela_duplo,0.975,nrep_boot);
		double quantil4_superior95 = myfunctions::quantil1(z4_estrela_duplo,0.025,nrep_boot);

		double quantil5_inferior95 = myfunctions::quantil1(z5_estrela_duplo,0.975,nrep_boot);
		double quantil5_superior95 = myfunctions::quantil1(z5_estrela_duplo,0.025,nrep_boot);

		if(ncorrecoes==2){
			// AQUI ESTAMOS CORRIGINDO O CALCULO DOS LIMITES INFERIORES E SUPERIORES DO INTERVALO DE CONFIANCA.
			// ESSA CORRECAO FAZ USO DO BOOTSTRAP EXTERIOR.
			li_0_95 = temp(1,0)-quantil0_inferior95*(2*sum(hc0_b)/nrep_boot-sqrt(HC0(1,1)));
			ls_0_95 = temp(1,0)-quantil0_superior95*(2*sum(hc0_b)/nrep_boot-sqrt(HC0(1,1)));
			li_2_95 = temp(1,0)-quantil2_inferior95*(2*sum(hc2_b)/nrep_boot-sqrt(HC2(1,1)));
			ls_2_95 = temp(1,0)-quantil2_superior95*(2*sum(hc2_b)/nrep_boot-sqrt(HC2(1,1)));
			li_3_95 = temp(1,0)-quantil3_inferior95*(2*sum(hc3_b)/nrep_boot-sqrt(HC3(1,1)));
			ls_3_95 = temp(1,0)-quantil3_superior95*(2*sum(hc3_b)/nrep_boot-sqrt(HC3(1,1)));
			li_4_95 = temp(1,0)-quantil4_inferior95*(2*sum(hc4_b)/nrep_boot-sqrt(HC4(1,1)));
			ls_4_95 = temp(1,0)-quantil4_superior95*(2*sum(hc4_b)/nrep_boot-sqrt(HC4(1,1)));
			li_5_95 = temp(1,0)-quantil5_inferior95*(2*sum(hc5_b)/nrep_boot-sqrt(HC5(1,1)));
			ls_5_95 = temp(1,0)-quantil5_superior95*(2*sum(hc5_b)/nrep_boot-sqrt(HC5(1,1)));
		}

		if(ncorrecoes==1){
			// AQUI ESTAMOS CONSTRUINDO OS LIMITES DOS INTERVALOS DE CONFIANCAS SEM USAR A CORRECAO DO DESVIO QUE
			// ENTRA NO CALCULO DOS LIMITES.        
			li_0_95 = temp(1,0)-quantil0_inferior95*sqrt(HC0(1,1));
			ls_0_95 = temp(1,0)-quantil0_superior95*sqrt(HC0(1,1));
			li_2_95 = temp(1,0)-quantil2_inferior95*sqrt(HC2(1,1));
			ls_2_95 = temp(1,0)-quantil2_superior95*sqrt(HC2(1,1));
			li_3_95 = temp(1,0)-quantil3_inferior95*sqrt(HC3(1,1));
			ls_3_95 = temp(1,0)-quantil3_superior95*sqrt(HC3(1,1));
			li_4_95 = temp(1,0)-quantil4_inferior95*sqrt(HC4(1,1));
			ls_4_95 = temp(1,0)-quantil4_superior95*sqrt(HC4(1,1));
			li_5_95 = temp(1,0)-quantil5_inferior95*sqrt(HC5(1,1));
			ls_5_95 = temp(1,0)-quantil5_superior95*sqrt(HC5(1,1));
		}

		if(beta(1)>=li_0_95&&beta(1)<=ls_0_95){
			cob_0_95_t_percentil_duplo(i) = 1;
		}else
			cob_0_95_t_percentil_duplo(i) = 0;

		if(beta(1)>=li_2_95&&beta(1)<=ls_2_95){
			cob_2_95_t_percentil_duplo(i) = 1;
		}else
			cob_2_95_t_percentil_duplo(i) = 0;

		if(beta(1)>=li_3_95&&beta(1)<=ls_3_95){
			cob_3_95_t_percentil_duplo(i) = 1;
		}else
			cob_3_95_t_percentil_duplo(i) = 0;

		if(beta(1)>=li_4_95&&beta(1)<=ls_4_95){
			cob_4_95_t_percentil_duplo(i) = 1;
		}else
			cob_4_95_t_percentil_duplo(i) = 0;

		if(beta(1)>=li_5_95&&beta(1)<=ls_5_95){
			cob_5_95_t_percentil_duplo(i) = 1;
		}else
			cob_5_95_t_percentil_duplo(i) = 0;

		if(beta(1)<li_0_95){
			ncobesq_0_95_t_percentil_duplo(i) = 1;
		}else
			ncobesq_0_95_t_percentil_duplo(i) = 0;

		if(beta(1)<li_2_95){
			ncobesq_2_95_t_percentil_duplo(i) = 1;
		}else
			ncobesq_2_95_t_percentil_duplo(i) = 0;

		if(beta(1)<li_3_95){
			ncobesq_3_95_t_percentil_duplo(i) = 1;
		}else
			ncobesq_3_95_t_percentil_duplo(i) = 0;

		if(beta(1)<li_4_95){
			ncobesq_4_95_t_percentil_duplo(i) = 1;
		}else
			ncobesq_4_95_t_percentil_duplo(i) = 0;

		if(beta(1)<li_5_95){
			ncobesq_5_95_t_percentil_duplo(i) = 1;
		}else
			ncobesq_5_95_t_percentil_duplo(i) = 0;

		if(beta(1)>ls_0_95){
			ncobdi_0_95_t_percentil_duplo(i) = 1;
		}else
			ncobdi_0_95_t_percentil_duplo(i) = 0;

		ampl_0_95_t_percentil_duplo(i) = ls_0_95-li_0_95;

		if(beta(1)>ls_2_95){
			ncobdi_2_95_t_percentil_duplo(i) = 1;
		}else
			ncobdi_2_95_t_percentil_duplo(i) = 0;

		ampl_2_95_t_percentil_duplo(i) = ls_2_95-li_2_95;

		if(beta(1)>ls_3_95){
			ncobdi_3_95_t_percentil_duplo(i) = 1;
		}else
			ncobdi_3_95_t_percentil_duplo(i) = 0;

		ampl_3_95_t_percentil_duplo(i) = ls_3_95-li_0_95;

		if(beta(1)>ls_4_95){
			ncobdi_4_95_t_percentil_duplo(i) = 1;
		}else
			ncobdi_4_95_t_percentil_duplo(i) = 0;

		ampl_4_95_t_percentil_duplo(i) = ls_4_95-li_4_95;

		if(beta(1)>ls_5_95){
			ncobdi_5_95_t_percentil_duplo(i) = 1;
		}else
			ncobdi_5_95_t_percentil_duplo(i) = 0;

		ampl_5_95_t_percentil_duplo(i) = ls_5_95-li_5_95;

		// CONTANDO CONVERGÊNCIAS PARA O BOOTSTRAP T (AQUI NÃO É O BOOSTRAP DUPLO.)

		quantil0_inferior95 = myfunctions::quantil1(z_estrela0,0.975,nrep_boot);
		quantil0_superior95 = myfunctions::quantil1(z_estrela0,0.025,nrep_boot);
		li_0_95 = temp(1,0)-quantil0_inferior95*sqrt(HC0(1,1));
		ls_0_95 = temp(1,0)-quantil0_superior95*sqrt(HC0(1,1));

		lils_hc0_bootstrapt << li_0_95 << endl;
      lils_hc0_bootstrapt << ls_0_95 << endl;

		quantil2_inferior95 = myfunctions::quantil1(z_estrela2,0.975,nrep_boot);
		quantil2_superior95 = myfunctions::quantil1(z_estrela2,0.025,nrep_boot);
		li_2_95 = temp(1,0)-quantil2_inferior95*sqrt(HC2(1,1));
		ls_2_95 = temp(1,0)-quantil2_superior95*sqrt(HC2(1,1));
      
		lils_hc2_bootstrapt << li_2_95 << endl;
      lils_hc2_bootstrapt << ls_2_95 << endl;

		quantil3_inferior95 = myfunctions::quantil1(z_estrela3,0.975,nrep_boot);
		quantil3_superior95 = myfunctions::quantil1(z_estrela3,0.025,nrep_boot);
		li_3_95 = temp(1,0)-quantil3_inferior95*sqrt(HC3(1,1));
		ls_3_95 = temp(1,0)-quantil3_superior95*sqrt(HC3(1,1));

      lils_hc3_bootstrapt << li_3_95 << endl;
      lils_hc3_bootstrapt << ls_3_95 << endl;

		quantil4_inferior95 = myfunctions::quantil1(z_estrela4,0.975,nrep_boot);
		quantil4_superior95 = myfunctions::quantil1(z_estrela4,0.025,nrep_boot);
		li_4_95 = temp(1,0)-quantil4_inferior95*sqrt(HC4(1,1));
		ls_4_95 = temp(1,0)-quantil4_superior95*sqrt(HC4(1,1));

      lils_hc4_bootstrapt << li_4_95 << endl;
      lils_hc4_bootstrapt << ls_4_95 << endl;

		quantil5_inferior95 = myfunctions::quantil1(z_estrela5,0.975,nrep_boot);
		quantil5_superior95 = myfunctions::quantil1(z_estrela5,0.025,nrep_boot);
		li_5_95 = temp(1,0)-quantil5_inferior95*sqrt(HC5(1,1));
		ls_5_95 = temp(1,0)-quantil5_superior95*sqrt(HC5(1,1));

      lils_hc5_bootstrapt << li_5_95 << endl;
      lils_hc5_bootstrapt << ls_5_95 << endl;

		if(beta(1)>=li_0_95 && beta(1)<=ls_0_95){
			cob_0_95_t_percentil(i) = 1;
		}else
			cob_0_95_t_percentil(i) = 0;

		if(beta(1)>=li_2_95 && beta(1)<=ls_2_95){
			cob_2_95_t_percentil(i) = 1;
		}else
			cob_2_95_t_percentil(i) = 0;

		if(beta(1)>=li_3_95 && beta(1)<=ls_3_95){
			cob_3_95_t_percentil(i) = 1;
		}else
			cob_3_95_t_percentil(i) = 0;

		if(beta(1)>=li_4_95 && beta(1)<=ls_4_95){
			cob_4_95_t_percentil(i) = 1;
		}else
			cob_4_95_t_percentil(i) = 0;

		if(beta(1)>=li_5_95 && beta(1)<=ls_5_95){
			cob_5_95_t_percentil(i) = 1;
		}else
			cob_5_95_t_percentil(i) = 0;

		if(beta(1)<li_0_95){
			ncobesq_0_95_t_percentil(i) = 1;
		}else
			ncobesq_0_95_t_percentil(i) = 0;

		if(beta(1)<li_2_95){
			ncobesq_2_95_t_percentil(i) = 1;
		}else
			ncobesq_2_95_t_percentil(i) = 0;

		if(beta(1)<li_3_95){
			ncobesq_3_95_t_percentil(i) = 1;
		}else
			ncobesq_3_95_t_percentil(i) = 0;

		if(beta(1)<li_4_95){
			ncobesq_4_95_t_percentil(i) = 1;
		}else
			ncobesq_4_95_t_percentil(i) = 0;

		if(beta(1)<li_5_95){
			ncobesq_5_95_t_percentil(i) = 1;
		}else
			ncobesq_5_95_t_percentil(i) = 0;

		if(beta(1)>ls_0_95){
			ncobdi_0_95_t_percentil(i) = 1;
		}else
			ncobdi_0_95_t_percentil(i) = 0;

		ampl_0_95_t_percentil(i) = ls_0_95-li_0_95;

		if(beta(1)>ls_2_95){
			ncobdi_2_95_t_percentil(i) = 1;
		}else
			ncobdi_2_95_t_percentil(i) = 0;

		ampl_2_95_t_percentil(i) = ls_2_95-li_2_95;

		if(beta(1)>ls_3_95){
			ncobdi_3_95_t_percentil(i) = 1;
		}else
			ncobdi_3_95_t_percentil(i) = 0;

		ampl_3_95_t_percentil(i) = ls_3_95-li_3_95;

		if(beta(1)>ls_4_95){
			ncobdi_4_95_t_percentil(i) = 1;
		}else
			ncobdi_4_95_t_percentil(i) = 0;

		ampl_4_95_t_percentil(i) = ls_4_95-li_4_95;

		if(beta(1)>ls_5_95){
			ncobdi_5_95_t_percentil(i) = 1;
		}else
			ncobdi_5_95_t_percentil(i) = 0;

		ampl_5_95_t_percentil(i) = ls_5_95-li_5_95;


		// INTERVALO BOOTSTRAP PERCENTIL.
		double li95 = myfunctions::quantil1(beta2,0.025,nrep_boot);
		double ls95 = myfunctions::quantil1(beta2,0.975,nrep_boot);

      lils_percentil << li95 << endl;
      lils_percentil << ls95 << endl;

		if(beta(1)>=li95 && beta(1)<=ls95)
			cob95_percentil(i) = 1;
		else
			cob95_percentil(i) = 0;

		ampl95_percentil(i) = ls95-li95;

		if(beta(1)<li95)
			ncobesq95_percentil(i) = 1;
		else
			ncobesq95_percentil(i) = 0;

		if(beta(1)>ls95)
			ncobdi95_percentil(i) = 1;
		else
			ncobdi95_percentil(i) = 0;

		// INTERVALO PERCENTIL BOOTSTRAP DUPLO - 95% (BOOTSTRAP EXTERIOR).
		double hat_ql95 = myfunctions::quantil1(u_estrela,0.025,nrep_boot);
		double hat_qu95 = myfunctions::quantil1(u_estrela,0.975,nrep_boot);
		ls95 = myfunctions::quantil1(beta2,hat_qu95,nrep_boot);
		li95 = myfunctions::quantil1(beta2,hat_ql95,nrep_boot); 

      lils_percentil_duplo << li95 << endl;
      lils_percentil_duplo << ls95 << endl;

		ampl95_percentil_duplo(i) = ls95-li95;

		if(li95<=beta(1) && beta(1)<=ls95)
			cob95_percentil_duplo(i) = 1;
		else
			cob95_percentil_duplo(i) = 0;

		if(beta(1)<li95)
			ncobesq95_percentil_duplo(i) = 1;
		else
			ncobesq95_percentil_duplo(i) = 0;

		if(beta(1)>ls95)
			ncobdi95_percentil_duplo(i) = 1;
		else
			ncobdi95_percentil_duplo(i) = 0;

		//INTERVALO BOORSTRAP T DUPLO (CORRETO). BASEADO NO ALGORITMO DAS PAGINAS 84-85 DO ARTIGO:
		//IMPLEMENTING THE DOUBLE BOOTSTRAP, MCCULLOUCH AND VINOD, COMPUTATIONAL ECONOMICS, 1998.

		quantil0_inferior95 = myfunctions::quantil1(z_estrela0, myfunctions::quantil1(Z0_j,0.975,nrep_boot),nrep_boot);
		quantil0_superior95 = myfunctions::quantil1(z_estrela0,myfunctions::quantil1(Z0_j,0.025,nrep_boot),nrep_boot);
		li_0_95 = temp(1,0)-quantil0_inferior95*sqrt(HC0(1,1));
		ls_0_95 = temp(1,0)-quantil0_superior95*sqrt(HC0(1,1));
      
		lils_hc0_bootstrapt_duplo << li_0_95 << endl;
		lils_hc0_bootstrapt_duplo << ls_0_95 << endl;

		quantil2_inferior95 = myfunctions::quantil1(z_estrela2, myfunctions::quantil1(Z2_j,0.975,nrep_boot),nrep_boot);
		quantil2_superior95 = myfunctions::quantil1(z_estrela2,myfunctions::quantil1(Z2_j,0.025,nrep_boot),nrep_boot);
		li_2_95 = temp(1,0)-quantil2_inferior95*sqrt(HC2(1,1));
		ls_2_95 = temp(1,0)-quantil2_superior95*sqrt(HC2(1,1));

		lils_hc2_bootstrapt_duplo << li_2_95 << endl;
		lils_hc2_bootstrapt_duplo << ls_2_95 << endl;

		quantil3_inferior95 = myfunctions::quantil1(z_estrela3, myfunctions::quantil1(Z3_j,0.975,nrep_boot),nrep_boot);
		quantil3_superior95 = myfunctions::quantil1(z_estrela3,myfunctions::quantil1(Z3_j,0.025,nrep_boot),nrep_boot);
		li_3_95 = temp(1,0)-quantil3_inferior95*sqrt(HC3(1,1));
		ls_3_95 = temp(1,0)-quantil3_superior95*sqrt(HC3(1,1));

      lils_hc3_bootstrapt_duplo << li_3_95 << endl;
      lils_hc3_bootstrapt_duplo << ls_3_95 << endl;

		quantil4_inferior95 = myfunctions::quantil1(z_estrela4, myfunctions::quantil1(Z4_j,0.975,nrep_boot),nrep_boot);
		quantil4_superior95 = myfunctions::quantil1(z_estrela4,myfunctions::quantil1(Z4_j,0.025,nrep_boot),nrep_boot);
		li_4_95 = temp(1,0)-quantil4_inferior95*sqrt(HC4(1,1));
		ls_4_95 = temp(1,0)-quantil4_superior95*sqrt(HC4(1,1));

      lils_hc4_bootstrapt_duplo << li_4_95 << endl;
      lils_hc4_bootstrapt_duplo << ls_4_95 << endl;

		quantil5_inferior95 = myfunctions::quantil1(z_estrela5, myfunctions::quantil1(Z5_j,0.975,nrep_boot),nrep_boot);
		quantil5_superior95 = myfunctions::quantil1(z_estrela5,myfunctions::quantil1(Z5_j,0.025,nrep_boot),nrep_boot);
		li_5_95 = temp(1,0)-quantil5_inferior95*sqrt(HC5(1,1));
		ls_5_95 = temp(1,0)-quantil5_superior95*sqrt(HC5(1,1));

      lils_hc5_bootstrapt_duplo << li_5_95 << endl;
      lils_hc5_bootstrapt_duplo << ls_5_95 << endl;
	
		if(beta(1)>=li_0_95 && beta(1)<=ls_0_95){
			cob_0_95_t_percentil_duplo1(i) = 1;
		}else
			cob_0_95_t_percentil_duplo1(i) = 0;

		if(beta(1)>=li_2_95 && beta(1)<=ls_2_95){
			cob_2_95_t_percentil_duplo1(i) = 1;
		}else
			cob_2_95_t_percentil_duplo1(i) = 0;

		if(beta(1)>=li_3_95 && beta(1)<=ls_3_95){
			cob_3_95_t_percentil_duplo1(i) = 1;
		}else
			cob_3_95_t_percentil_duplo1(i) = 0;

		if(beta(1)>=li_4_95 && beta(1)<=ls_4_95){
			cob_4_95_t_percentil_duplo1(i) = 1;
		}else
			cob_4_95_t_percentil_duplo1(i) = 0;

		if(beta(1)>=li_5_95 && beta(1)<=ls_5_95){
			cob_5_95_t_percentil_duplo1(i) = 1;
		}else
			cob_5_95_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_0_95){
			ncobesq_0_95_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_0_95_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_2_95){
			ncobesq_2_95_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_2_95_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_3_95){
			ncobesq_3_95_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_3_95_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_4_95){
			ncobesq_4_95_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_4_95_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_5_95){
			ncobesq_5_95_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_5_95_t_percentil_duplo1(i) = 0;

		if(beta(1)>ls_0_95){
			ncobdi_0_95_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_0_95_t_percentil_duplo1(i) = 0;

		ampl_0_95_t_percentil_duplo1(i) = ls_0_95-li_0_95;

		if(beta(1)>ls_2_95){
			ncobdi_2_95_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_2_95_t_percentil_duplo1(i) = 0;

		ampl_2_95_t_percentil_duplo1(i) = ls_2_95-li_2_95;

		if(beta(1)>ls_3_95){
			ncobdi_3_95_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_3_95_t_percentil_duplo1(i) = 0;

		ampl_3_95_t_percentil_duplo1(i) = ls_3_95-li_3_95;

		if(beta(1)>ls_4_95){
			ncobdi_4_95_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_4_95_t_percentil_duplo1(i) = 0;

		ampl_4_95_t_percentil_duplo1(i) = ls_4_95-li_4_95;

		if(beta(1)>ls_5_95){
			ncobdi_5_95_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_5_95_t_percentil_duplo1(i) = 0;

		ampl_5_95_t_percentil_duplo1(i) = ls_5_95-li_5_95;

		// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		//                              INTERVALOS PARA 99%
		// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

		// CONTANDO CONVERGÊNCIAS PARA O BOOSTRAP T DUPLO. (ESSE ESQUEMA DE BOOTSTRAP DUPLO NAO É CORRETO,
		// CONTUDO, ESTOU EM CASOS DE HOMOSCEDASTICIDADE. PRECISA SER VERIFICADO.)    

		double quantil0_inferior99 = myfunctions::quantil1(z0_estrela_duplo,0.995,nrep_boot);
		double quantil0_superior99 = myfunctions::quantil1(z0_estrela_duplo,0.005,nrep_boot);

		double quantil2_inferior99 = myfunctions::quantil1(z2_estrela_duplo,0.995,nrep_boot);
		double quantil2_superior99 = myfunctions::quantil1(z2_estrela_duplo,0.005,nrep_boot);

		double quantil3_inferior99 = myfunctions::quantil1(z3_estrela_duplo,0.995,nrep_boot);
		double quantil3_superior99 = myfunctions::quantil1(z3_estrela_duplo,0.005,nrep_boot);

		double quantil4_inferior99 = myfunctions::quantil1(z4_estrela_duplo,0.995,nrep_boot);
		double quantil4_superior99 = myfunctions::quantil1(z4_estrela_duplo,0.005,nrep_boot);

		double quantil5_inferior99 = myfunctions::quantil1(z5_estrela_duplo,0.995,nrep_boot);
		double quantil5_superior99 = myfunctions::quantil1(z5_estrela_duplo,0.005,nrep_boot);

		if(ncorrecoes==2){
			// AQUI ESTAMOS CORRIGINDO O CALCULO DOS LIMITES INFERIORES E SUPERIORES DO INTERVALO DE CONFIANCA.
			// ESSA CORRECAO FAZ USO DO BOOTSTRAP EXTERIOR.
			li_0_99 = temp(1,0)-quantil0_inferior99*(2*sum(hc0_b)/nrep_boot-sqrt(HC0(1,1)));
			ls_0_99 = temp(1,0)-quantil0_superior99*(2*sum(hc0_b)/nrep_boot-sqrt(HC0(1,1)));
			li_2_99 = temp(1,0)-quantil2_inferior99*(2*sum(hc2_b)/nrep_boot-sqrt(HC2(1,1)));
			ls_2_99 = temp(1,0)-quantil2_superior99*(2*sum(hc2_b)/nrep_boot-sqrt(HC2(1,1)));
			li_3_99 = temp(1,0)-quantil3_inferior99*(2*sum(hc3_b)/nrep_boot-sqrt(HC3(1,1)));
			ls_3_99 = temp(1,0)-quantil3_superior99*(2*sum(hc3_b)/nrep_boot-sqrt(HC3(1,1)));
			li_4_99 = temp(1,0)-quantil4_inferior99*(2*sum(hc4_b)/nrep_boot-sqrt(HC4(1,1)));
			ls_4_99 = temp(1,0)-quantil4_superior99*(2*sum(hc4_b)/nrep_boot-sqrt(HC4(1,1)));
			li_5_99 = temp(1,0)-quantil5_inferior99*(2*sum(hc5_b)/nrep_boot-sqrt(HC5(1,1)));
			ls_5_99 = temp(1,0)-quantil5_superior99*(2*sum(hc5_b)/nrep_boot-sqrt(HC5(1,1)));
		}

		if(ncorrecoes==1){
			// AQUI ESTAMOS CONSTRUINDO OS LIMITES DOS INTERVALOS DE CONFIANCAS SEM USAR A CORRECAO DO DESVIO QUE
			// ENTRA NO CALCULO DOS LIMITES.        
			li_0_99 = temp(1,0)-quantil0_inferior99*sqrt(HC0(1,1));
			ls_0_99 = temp(1,0)-quantil0_superior99*sqrt(HC0(1,1));
			li_2_99 = temp(1,0)-quantil2_inferior99*sqrt(HC2(1,1));
			ls_2_99 = temp(1,0)-quantil2_superior99*sqrt(HC2(1,1));
			li_3_99 = temp(1,0)-quantil3_inferior99*sqrt(HC3(1,1));
			ls_3_99 = temp(1,0)-quantil3_superior99*sqrt(HC3(1,1));
			li_4_99 = temp(1,0)-quantil4_inferior99*sqrt(HC4(1,1));
			ls_4_99 = temp(1,0)-quantil4_superior99*sqrt(HC4(1,1));
			li_5_99 = temp(1,0)-quantil5_inferior99*sqrt(HC5(1,1));
			ls_5_99 = temp(1,0)-quantil5_superior99*sqrt(HC5(1,1));
		}

		if(beta(1)>=li_0_99&&beta(1)<=ls_0_99){
			cob_0_99_t_percentil_duplo(i) = 1;
		}else
			cob_0_99_t_percentil_duplo(i) = 0;

		if(beta(1)>=li_2_99&&beta(1)<=ls_2_99){
			cob_2_99_t_percentil_duplo(i) = 1;
		}else
			cob_2_99_t_percentil_duplo(i) = 0;

		if(beta(1)>=li_3_99&&beta(1)<=ls_3_99){
			cob_3_99_t_percentil_duplo(i) = 1;
		}else
			cob_3_99_t_percentil_duplo(i) = 0;

		if(beta(1)>=li_4_99&&beta(1)<=ls_4_99){
			cob_4_99_t_percentil_duplo(i) = 1;
		}else
			cob_4_99_t_percentil_duplo(i) = 0;

		if(beta(1)>=li_5_99&&beta(1)<=ls_5_99){
			cob_5_99_t_percentil_duplo(i) = 1;
		}else
			cob_5_99_t_percentil_duplo(i) = 0;

		if(beta(1)<li_0_99){
			ncobesq_0_99_t_percentil_duplo(i) = 1;
		}else
			ncobesq_0_99_t_percentil_duplo(i) = 0;

		if(beta(1)<li_2_99){
			ncobesq_2_99_t_percentil_duplo(i) = 1;
		}else
			ncobesq_2_99_t_percentil_duplo(i) = 0;

		if(beta(1)<li_3_99){
			ncobesq_3_99_t_percentil_duplo(i) = 1;
		}else
			ncobesq_3_99_t_percentil_duplo(i) = 0;

		if(beta(1)<li_4_99){
			ncobesq_4_99_t_percentil_duplo(i) = 1;
		}else
			ncobesq_4_99_t_percentil_duplo(i) = 0;

		if(beta(1)<li_5_99){
			ncobesq_5_99_t_percentil_duplo(i) = 1;
		}else
			ncobesq_5_99_t_percentil_duplo(i) = 0;

		if(beta(1)>ls_0_99){
			ncobdi_0_99_t_percentil_duplo(i) = 1;
		}else
			ncobdi_0_99_t_percentil_duplo(i) = 0;

		ampl_0_99_t_percentil_duplo(i) = ls_0_99-li_0_99;

		if(beta(1)>ls_2_99){
			ncobdi_2_99_t_percentil_duplo(i) = 1;
		}else
			ncobdi_2_99_t_percentil_duplo(i) = 0;

		ampl_2_99_t_percentil_duplo(i) = ls_2_99-li_2_99;

		if(beta(1)>ls_3_99){
			ncobdi_3_99_t_percentil_duplo(i) = 1;
		}else
			ncobdi_3_99_t_percentil_duplo(i) = 0;

		ampl_3_99_t_percentil_duplo(i) = ls_3_99-li_0_99;

		if(beta(1)>ls_4_99){
			ncobdi_4_99_t_percentil_duplo(i) = 1;
		}else
			ncobdi_4_99_t_percentil_duplo(i) = 0;

		ampl_4_99_t_percentil_duplo(i) = ls_4_99-li_4_99;

		if(beta(1)>ls_5_99){
			ncobdi_5_99_t_percentil_duplo(i) = 1;
		}else
			ncobdi_5_99_t_percentil_duplo(i) = 0;

		ampl_5_99_t_percentil_duplo(i) = ls_5_99-li_5_99;

		// CONTANDO CONVERGÊNCIAS PARA O BOOTSTRAP T (AQUI NÃO É O BOOSTRAP DUPLO.)

		quantil0_inferior99 = myfunctions::quantil1(z_estrela0,0.995,nrep_boot);
		quantil0_superior99 = myfunctions::quantil1(z_estrela0,0.005,nrep_boot);
		li_0_99 = temp(1,0)-quantil0_inferior99*sqrt(HC0(1,1));
		ls_0_99 = temp(1,0)-quantil0_superior99*sqrt(HC0(1,1));

		quantil2_inferior99 = myfunctions::quantil1(z_estrela2,0.995,nrep_boot);
		quantil2_superior99 = myfunctions::quantil1(z_estrela2,0.005,nrep_boot);
		li_2_99 = temp(1,0)-quantil2_inferior99*sqrt(HC2(1,1));
		ls_2_99 = temp(1,0)-quantil2_superior99*sqrt(HC2(1,1));

		quantil3_inferior99 = myfunctions::quantil1(z_estrela3,0.995,nrep_boot);
		quantil3_superior99 = myfunctions::quantil1(z_estrela3,0.005,nrep_boot);
		li_3_99 = temp(1,0)-quantil3_inferior99*sqrt(HC3(1,1));
		ls_3_99 = temp(1,0)-quantil3_superior99*sqrt(HC3(1,1));

		quantil4_inferior99 = myfunctions::quantil1(z_estrela4,0.995,nrep_boot);
		quantil4_superior99 = myfunctions::quantil1(z_estrela4,0.005,nrep_boot);
		li_4_99 = temp(1,0)-quantil4_inferior99*sqrt(HC4(1,1));
		ls_4_99 = temp(1,0)-quantil4_superior99*sqrt(HC4(1,1));

		quantil5_inferior99 = myfunctions::quantil1(z_estrela5,0.995,nrep_boot);
		quantil5_superior99 = myfunctions::quantil1(z_estrela5,0.005,nrep_boot);
		li_5_99 = temp(1,0)-quantil5_inferior99*sqrt(HC5(1,1));
		ls_5_99 = temp(1,0)-quantil5_superior99*sqrt(HC5(1,1));

		if(beta(1)>=li_0_99 && beta(1)<=ls_0_99){
			cob_0_99_t_percentil(i) = 1;
		}else
			cob_0_99_t_percentil(i) = 0;

		if(beta(1)>=li_2_99 && beta(1)<=ls_2_99){
			cob_2_99_t_percentil(i) = 1;
		}else
			cob_2_99_t_percentil(i) = 0;

		if(beta(1)>=li_3_99 && beta(1)<=ls_3_99){
			cob_3_99_t_percentil(i) = 1;
		}else
			cob_3_99_t_percentil(i) = 0;

		if(beta(1)>=li_4_99 && beta(1)<=ls_4_99){
			cob_4_99_t_percentil(i) = 1;
		}else
			cob_4_99_t_percentil(i) = 0;

		if(beta(1)>=li_5_99 && beta(1)<=ls_5_99){
			cob_5_99_t_percentil(i) = 1;
		}else
			cob_5_99_t_percentil(i) = 0;

		if(beta(1)<li_0_99){
			ncobesq_0_99_t_percentil(i) = 1;
		}else
			ncobesq_0_99_t_percentil(i) = 0;

		if(beta(1)<li_2_99){
			ncobesq_2_99_t_percentil(i) = 1;
		}else
			ncobesq_2_99_t_percentil(i) = 0;

		if(beta(1)<li_3_99){
			ncobesq_3_99_t_percentil(i) = 1;
		}else
			ncobesq_3_99_t_percentil(i) = 0;

		if(beta(1)<li_4_99){
			ncobesq_4_99_t_percentil(i) = 1;
		}else
			ncobesq_4_99_t_percentil(i) = 0;

		if(beta(1)<li_5_99){
			ncobesq_5_99_t_percentil(i) = 1;
		}else
			ncobesq_5_99_t_percentil(i) = 0;

		if(beta(1)>ls_0_99){
			ncobdi_0_99_t_percentil(i) = 1;
		}else
			ncobdi_0_99_t_percentil(i) = 0;

		ampl_0_99_t_percentil(i) = ls_0_99-li_0_99;

		if(beta(1)>ls_2_99){
			ncobdi_2_99_t_percentil(i) = 1;
		}else
			ncobdi_2_99_t_percentil(i) = 0;

		ampl_2_99_t_percentil(i) = ls_2_99-li_2_99;

		if(beta(1)>ls_3_99){
			ncobdi_3_99_t_percentil(i) = 1;
		}else
			ncobdi_3_99_t_percentil(i) = 0;

		ampl_3_99_t_percentil(i) = ls_3_99-li_3_99;

		if(beta(1)>ls_4_99){
			ncobdi_4_99_t_percentil(i) = 1;
		}else
			ncobdi_4_99_t_percentil(i) = 0;

		ampl_4_99_t_percentil(i) = ls_4_99-li_4_99;

		if(beta(1)>ls_5_99){
			ncobdi_5_99_t_percentil(i) = 1;
		}else
			ncobdi_5_99_t_percentil(i) = 0;

		ampl_5_99_t_percentil(i) = ls_5_99-li_5_99;


		// INTERVALO BOOTSTRAP PERCENTIL.
		double li99 = myfunctions::quantil1(beta2,0.005,nrep_boot);
		double ls99 = myfunctions::quantil1(beta2,0.995,nrep_boot);
		if(beta(1)>=li99 && beta(1)<=ls99)
			cob99_percentil(i) = 1;
		else
			cob99_percentil(i) = 0;

		ampl99_percentil(i) = ls99-li99;

		if(beta(1)<li99)
			ncobesq99_percentil(i) = 1;
		else
			ncobesq99_percentil(i) = 0;

		if(beta(1)>ls99)
			ncobdi99_percentil(i) = 1;
		else
			ncobdi99_percentil(i) = 0;

		// INTERVALO PERCENTIL BOOTSTRAP DUPLO - 99% (BOOTSTRAP EXTERIOR).
		double hat_ql99 = myfunctions::quantil1(u_estrela,0.005,nrep_boot);
		double hat_qu99 = myfunctions::quantil1(u_estrela,0.995,nrep_boot);
		ls99 = myfunctions::quantil1(beta2,hat_qu99,nrep_boot);
		li99 = myfunctions::quantil1(beta2,hat_ql99,nrep_boot);

		//ls99 = temp(1)-myfunctions::quantil1(betaj_estrela_menos_betaj,hat_ql99,nrep_boot);
		//li99 = temp(1)-myfunctions::quantil1(betaj_estrela_menos_betaj,hat_qu99,nrep_boot);

		ampl99_percentil_duplo(i) = ls99-li99;

		if(li99<=beta(1) && beta(1)<=ls99)
			cob99_percentil_duplo(i) = 1;
		else
			cob99_percentil_duplo(i) = 0;

		if(beta(1)<li99)
			ncobesq99_percentil_duplo(i) = 1;
		else
			ncobesq99_percentil_duplo(i) = 0;

		if(beta(1)>ls99)
			ncobdi99_percentil_duplo(i) = 1;
		else
			ncobdi99_percentil_duplo(i) = 0;

		//INTERVALO BOORSTRAP T DUPLO (CORRETO). BASEADO NO ALGORITMO DAS PAGINAS 84-85 DO ARTIGO:
		//IMPLEMENTING THE DOUBLE BOOTSTRAP, MCCULLOUCH AND VINOD, COMPUTATIONAL ECONOMICS, 1998.

		quantil0_inferior99 = myfunctions::quantil1(z_estrela0, myfunctions::quantil1(Z0_j,0.995,nrep_boot),nrep_boot);
		quantil0_superior99 = myfunctions::quantil1(z_estrela0,myfunctions::quantil1(Z0_j,0.005,nrep_boot),nrep_boot);
		li_0_99 = temp(1,0)-quantil0_inferior99*sqrt(HC0(1,1));
		ls_0_99 = temp(1,0)-quantil0_superior99*sqrt(HC0(1,1));

		quantil2_inferior99 = myfunctions::quantil1(z_estrela2, myfunctions::quantil1(Z2_j,0.995,nrep_boot),nrep_boot);
		quantil2_superior99 = myfunctions::quantil1(z_estrela2,myfunctions::quantil1(Z2_j,0.005,nrep_boot),nrep_boot);
		li_2_99 = temp(1,0)-quantil2_inferior99*sqrt(HC2(1,1));
		ls_2_99 = temp(1,0)-quantil2_superior99*sqrt(HC2(1,1));

		quantil3_inferior99 = myfunctions::quantil1(z_estrela3, myfunctions::quantil1(Z3_j,0.995,nrep_boot),nrep_boot);
		quantil3_superior99 = myfunctions::quantil1(z_estrela3,myfunctions::quantil1(Z3_j,0.005,nrep_boot),nrep_boot);
		li_3_99 = temp(1,0)-quantil3_inferior99*sqrt(HC3(1,1));
		ls_3_99 = temp(1,0)-quantil3_superior99*sqrt(HC3(1,1));

		quantil4_inferior99 = myfunctions::quantil1(z_estrela4, myfunctions::quantil1(Z4_j,0.995,nrep_boot),nrep_boot);
		quantil4_superior99 = myfunctions::quantil1(z_estrela4,myfunctions::quantil1(Z4_j,0.005,nrep_boot),nrep_boot);
		li_4_99 = temp(1,0)-quantil4_inferior99*sqrt(HC4(1,1));
		ls_4_99 = temp(1,0)-quantil4_superior99*sqrt(HC4(1,1));

		quantil5_inferior99 = myfunctions::quantil1(z_estrela5, myfunctions::quantil1(Z5_j,0.995,nrep_boot),nrep_boot);
		quantil5_superior99 = myfunctions::quantil1(z_estrela5,myfunctions::quantil1(Z5_j,0.005,nrep_boot),nrep_boot);
		li_5_99 = temp(1,0)-quantil5_inferior99*sqrt(HC5(1,1));
		ls_5_99 = temp(1,0)-quantil5_superior99*sqrt(HC5(1,1));

		if(beta(1)>=li_0_99 && beta(1)<=ls_0_99){
			cob_0_99_t_percentil_duplo1(i) = 1;
		}else
			cob_0_99_t_percentil_duplo1(i) = 0;

		if(beta(1)>=li_2_99 && beta(1)<=ls_2_99){
			cob_2_99_t_percentil_duplo1(i) = 1;
		}else
			cob_2_99_t_percentil_duplo1(i) = 0;

		if(beta(1)>=li_3_99 && beta(1)<=ls_3_99){
			cob_3_99_t_percentil_duplo1(i) = 1;
		}else
			cob_3_99_t_percentil_duplo1(i) = 0;

		if(beta(1)>=li_4_99 && beta(1)<=ls_4_99){
			cob_4_99_t_percentil_duplo1(i) = 1;
		}else
			cob_4_99_t_percentil_duplo1(i) = 0;

		if(beta(1)>=li_5_99 && beta(1)<=ls_5_99){
			cob_5_99_t_percentil_duplo1(i) = 1;
		}else
			cob_5_99_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_0_99){
			ncobesq_0_99_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_0_99_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_2_99){
			ncobesq_2_99_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_2_99_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_3_99){
			ncobesq_3_99_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_3_99_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_4_99){
			ncobesq_4_99_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_4_99_t_percentil_duplo1(i) = 0;

		if(beta(1)<li_5_99){
			ncobesq_5_99_t_percentil_duplo1(i) = 1;
		}else
			ncobesq_5_99_t_percentil_duplo1(i) = 0;

		if(beta(1)>ls_0_99){
			ncobdi_0_99_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_0_99_t_percentil_duplo1(i) = 0;

		ampl_0_99_t_percentil_duplo1(i) = ls_0_99-li_0_99;

		if(beta(1)>ls_2_99){
			ncobdi_2_99_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_2_99_t_percentil_duplo1(i) = 0;

		ampl_2_99_t_percentil_duplo1(i) = ls_2_99-li_2_99;

		if(beta(1)>ls_3_99){
			ncobdi_3_99_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_3_99_t_percentil_duplo1(i) = 0;

		ampl_3_99_t_percentil_duplo1(i) = ls_3_99-li_3_99;

		if(beta(1)>ls_4_99){
			ncobdi_4_99_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_4_99_t_percentil_duplo1(i) = 0;

		ampl_4_99_t_percentil_duplo1(i) = ls_4_99-li_4_99;

		if(beta(1)>ls_5_99){
			ncobdi_5_99_t_percentil_duplo1(i) = 1;
		}else
			ncobdi_5_99_t_percentil_duplo1(i) = 0;

		ampl_5_99_t_percentil_duplo1(i) = ls_5_99-li_5_99;
	} // AQUI TERMINA O LACO MONTE CARLO

	time_t rawtime_1;
	time (&rawtime_1);
	timeinfo = localtime (&rawtime_1);
	saida << ">> [*] Horario de termino da simulacao: " <<  asctime(timeinfo) << endl; 
	saida  << "(*) TEMPO DE EXECUCAO: " <<  float(clock() - tempo_inicial)/CLOCKS_PER_SEC << " segundos / " <<
		(float(clock() - tempo_inicial)/CLOCKS_PER_SEC)/60 << " minutos / " <<
		((float(clock() - tempo_inicial)/CLOCKS_PER_SEC)/60)/60 << " horas / " << 
		(((float(clock() - tempo_inicial)/CLOCKS_PER_SEC)/60)/60)/24 << " dias." << endl << endl;

	saida << "----------------------------------------------------------" << endl;
	saida << "                INTERVALOS SEM  BOOTSTRAP                 " << endl;
	saida << "----------------------------------------------------------" << "\n" << endl;
	saida << "....... INTERVALO T" << endl;
	saida << "....... NIVEL DE CONFIANCA: 90%" << endl;
	saida << "COBERTURA:      " 
		<< "OLS = " << (sum(cob90_t_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(cob90_t_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(cob90_t_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob90_t_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob90_t_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob90_t_hc5)/nrep)*100 << endl;
	saida << "AMPLITUDE:      "
		<< "OLS = " << sum(ampl90_t_ols)/nrep << ", "
		<< "HC0 = " << sum(ampl90_t_hc0)/nrep << ", "
		<< "HC2 = " << sum(ampl90_t_hc2)/nrep << ", "
		<< "HC3 = " << sum(ampl90_t_hc3)/nrep << ", "
		<< "HC4 = " << sum(ampl90_t_hc4)/nrep << ", "
		<< "HC5 = " << sum(ampl90_t_hc5)/nrep << endl;
	saida << "NAO COB. ESQ.:  " 
		<< "OLS = " << (sum(ncobesq90_t_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(ncobesq90_t_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq90_t_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq90_t_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq90_t_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq90_t_hc5)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  " 
		<< "OLS = " << (sum(ncobdi90_t_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(ncobdi90_t_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi90_t_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi90_t_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi90_t_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobdi90_t_hc5)/nrep)*100 << "\n" << endl;

	saida << "....... INTERVALO T" << endl;
	saida << "....... NIVEL DE CONFIANCA: 95%" << endl;
	saida << "COBERTURA:      " 
		<< "OLS = " << (sum(cob95_t_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(cob95_t_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(cob95_t_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob95_t_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob95_t_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob95_t_hc5)/nrep)*100 << endl;
	saida << "AMPLITUDE:      "
		<< "OLS = " << sum(ampl95_t_ols)/nrep << ", "
		<< "HC0 = " << sum(ampl95_t_hc0)/nrep << ", "
		<< "HC2 = " << sum(ampl95_t_hc2)/nrep << ", "
		<< "HC3 = " << sum(ampl95_t_hc3)/nrep << ", "
		<< "HC4 = " << sum(ampl95_t_hc4)/nrep << ", "
		<< "HC5 = " << sum(ampl95_t_hc5)/nrep << endl;
	saida << "NAO COB. ESQ.:  " 
		<< "OLS = " << (sum(ncobesq95_t_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(ncobesq95_t_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq95_t_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq95_t_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq95_t_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq95_t_hc5)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  " 
		<< "OLS = " << (sum(ncobdi95_t_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(ncobdi95_t_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi95_t_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi95_t_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi95_t_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobdi95_t_hc5)/nrep)*100 << "\n" << endl;

	saida << "....... INTERVALO T" << endl;
	saida << "....... NIVEL DE CONFIANCA: 99%" << endl;
	saida << "COBERTURA:      " 
		<< "OLS = " << (sum(cob99_t_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(cob99_t_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(cob99_t_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob99_t_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob99_t_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob99_t_hc5)/nrep)*100 << endl;
	saida << "AMPLITUDE:      "
		<< "OLS = " << sum(ampl99_t_ols)/nrep << ", "
		<< "HC0 = " << sum(ampl99_t_hc0)/nrep << ", "
		<< "HC2 = " << sum(ampl99_t_hc2)/nrep << ", "
		<< "HC3 = " << sum(ampl99_t_hc3)/nrep << ", "
		<< "HC4 = " << sum(ampl99_t_hc4)/nrep << ", "
		<< "HC5 = " << sum(ampl99_t_hc5)/nrep << endl;
	saida << "NAO COB. ESQ.:  " 
		<< "OLS = " << (sum(ncobesq99_t_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(ncobesq99_t_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq99_t_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq99_t_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq99_t_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq99_t_hc5)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  " 
		<< "OLS = " << (sum(ncobdi99_t_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(ncobdi99_t_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi99_t_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi99_t_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi99_t_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobdi99_t_hc5)/nrep)*100 << "\n" << endl;

	saida << "....... INTERVALO Z" << endl;
	saida << "....... NIVEL DE CONFIANCA: 90%" << endl;
	saida << "COBERTURA:      " 
		<< "OLS = " << (sum(cob90_z_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(cob90_z_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(cob90_z_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob90_z_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob90_z_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob90_z_hc5)/nrep)*100 << endl;
	saida << "AMPLITUDE:      "
		<< "OLS = " << sum(ampl90_z_ols)/nrep << ", "
		<< "HC0 = " << sum(ampl90_z_hc0)/nrep << ", "
		<< "HC2 = " << sum(ampl90_z_hc2)/nrep << ", "
		<< "HC3 = " << sum(ampl90_z_hc3)/nrep << ", "
		<< "HC4 = " << sum(ampl90_z_hc4)/nrep << ", "
		<< "HC5 = " << sum(ampl90_z_hc5)/nrep << endl;
	saida << "NAO COB. ESQ.:  " 
		<< "OLS = " << (sum(ncobesq90_z_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(ncobesq90_z_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq90_z_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq90_z_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq90_z_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq90_z_hc5)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  " 
		<< "OLS = " << (sum(ncobdi90_z_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(ncobdi90_z_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi90_z_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi90_z_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi90_z_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobdi90_z_hc5)/nrep)*100 << "\n" << endl;

	saida << "....... INTERVALO Z" << endl;
	saida << "....... NIVEL DE CONFIANCA: 95%" << endl;
	saida << "COBERTURA:      " 
		<< "OLS = " << (sum(cob95_z_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(cob95_z_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(cob95_z_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob95_z_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob95_z_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob95_z_hc5)/nrep)*100 << endl;
	saida << "AMPLITUDE:      "
		<< "OLS = " << sum(ampl95_z_ols)/nrep << ", "
		<< "HC0 = " << sum(ampl95_z_hc0)/nrep << ", "
		<< "HC2 = " << sum(ampl95_z_hc2)/nrep << ", "
		<< "HC3 = " << sum(ampl95_z_hc3)/nrep << ", "
		<< "HC4 = " << sum(ampl95_z_hc4)/nrep << ", "
		<< "HC5 = " << sum(ampl95_z_hc5)/nrep << endl;
	saida << "NAO COB. ESQ.:  " 
		<< "OLS = " << (sum(ncobesq95_z_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(ncobesq95_z_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq95_z_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq95_z_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq95_z_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq95_z_hc5)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  " 
		<< "OLS = " << (sum(ncobdi95_z_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(ncobdi95_z_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi95_z_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi95_z_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi95_z_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobdi95_z_hc5)/nrep)*100 << "\n" << endl;

	saida << "....... INTERVALO Z" << endl;
	saida << "....... NIVEL DE CONFIANCA: 99%" << endl;
	saida << "COBERTURA:      " 
		<< "OLS = " << (sum(cob99_z_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(cob99_z_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(cob99_z_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob99_z_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob99_z_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob99_z_hc5)/nrep)*100 << endl;
	saida << "AMPLITUDE:      "
		<< "OLS = " << sum(ampl99_z_ols)/nrep << ", "
		<< "HC0 = " << sum(ampl99_z_hc0)/nrep << ", "
		<< "HC2 = " << sum(ampl99_z_hc2)/nrep << ", "
		<< "HC3 = " << sum(ampl99_z_hc3)/nrep << ", "
		<< "HC4 = " << sum(ampl99_z_hc4)/nrep << ", "
		<< "HC5 = " << sum(ampl99_z_hc5)/nrep << endl;
	saida << "NAO COB. ESQ.:  " 
		<< "OLS = " << (sum(ncobesq99_z_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(ncobesq99_z_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq99_z_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq99_z_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq99_z_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq99_z_hc5)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  " 
		<< "OLS = " << (sum(ncobdi99_z_ols)/nrep)*100 << ", "
		<< "HC0 = " << (sum(ncobdi99_z_hc0)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi99_z_hc2)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi99_z_hc3)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi99_z_hc4)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobdi99_z_hc5)/nrep)*100 << "\n" << endl;

	saida << "----------------------------------------------------------" << endl;
	saida << "               INTERVALOS USANDO BOOTSTRAP                " << endl;
	saida << "----------------------------------------------------------" << "\n" << endl;
	saida << "....... BOOTSTRAP PERCENTIL"  << endl;
	saida << "....... NIVEL DE CONFIANCA: 90%" << "\n"	<< endl;
	saida << "COBERTURA:      " << (sum(cob90_percentil)/nrep)*100 << endl;
	saida << "APLITUDE:       " << (sum(ampl90_percentil)/nrep) << endl;
	saida << "NAO COB. ESQ.:  " << (sum(ncobesq90_percentil)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  " << (sum(ncobdi90_percentil)/nrep)*100 << "\n" << endl;

	saida << "....... BOOTSTRAP PERCENTIL DUPLO" << endl;
	saida << "....... NIVEL DE CONFIANCA: 90%" << "\n"	<< endl;
	saida << "COBERTURA:      " << (sum(cob90_percentil_duplo)/nrep)*100 << endl;
	saida << "APLITUDE:       " << (sum(ampl90_percentil_duplo)/nrep) << endl;
	saida << "NAO COB. ESQ.:  " << (sum(ncobesq90_percentil_duplo)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  " << (sum(ncobdi90_percentil_duplo)/nrep)*100 << "\n" << endl;

	saida << "....... BOOTSTRAP T" << endl;
	saida << "....... NIVEL DE CONFIANCA: 90%" << "\n" <<endl;
	saida << "COBERTURA:      " 
		<< "HC0 = " << (sum(cob_0_90_t_percentil)/nrep)*100 << ", " 
		<< "HC2 = " << (sum(cob_2_90_t_percentil)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob_3_90_t_percentil)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob_4_90_t_percentil)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob_5_90_t_percentil)/nrep)*100 << endl;
	saida << "APLITUDE:       " 
		<< "HC0 = " <<(sum(ampl_0_90_t_percentil)/nrep) << ", "
		<< "HC2 = " << (sum(ampl_2_90_t_percentil)/nrep) << ", "
		<< "HC3 = " << (sum(ampl_3_90_t_percentil)/nrep) << ", "
		<< "HC4 = " << (sum(ampl_4_90_t_percentil)/nrep) << ", "
		<< "HC5 = " << (sum(ampl_5_90_t_percentil)/nrep) << endl;
	saida << "NAO COB. ESQ.:  "
		<< "HC0 = " << (sum(ncobesq_0_90_t_percentil)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq_2_90_t_percentil)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq_3_90_t_percentil)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq_4_90_t_percentil)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq_5_90_t_percentil)/nrep)*100 << endl;  
	saida << "NAO COB. DIR.:  " 
		<< "HC0 = " << (sum(ncobdi_0_90_t_percentil)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi_2_90_t_percentil)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi_3_90_t_percentil)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi_4_90_t_percentil)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobdi_5_90_t_percentil)/nrep)*100 << "\n" << endl;

	saida << "....... BOOTSTRAP T DUPLO" << endl;
	saida << "....... NIVEL DE CONFIANCA: 90%" << "\n" << endl;
	saida << "COBERTURA:      " 
		<< "HC0 = " << (sum(cob_0_90_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC2 = " << (sum(cob_2_90_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob_3_90_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob_4_90_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob_5_90_t_percentil_duplo1)/nrep)*100 << endl;
	saida << "APLITUDE:       "
		<< "HC0 = " << (sum(ampl_0_90_t_percentil_duplo1)/nrep) << ", "
		<< "HC2 = " << (sum(ampl_2_90_t_percentil_duplo1)/nrep) << ", "
		<< "HC3 = " << (sum(ampl_3_90_t_percentil_duplo1)/nrep) << ", "
		<< "HC4 = " << (sum(ampl_4_90_t_percentil_duplo1)/nrep) << ", "
		<< "HC5 = " << (sum(ampl_5_90_t_percentil_duplo1)/nrep) << endl;
	saida << "NAO COB. ESQ.:  "
		<< "HC0 = " << (sum(ncobesq_0_90_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq_2_90_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq_3_90_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq_4_90_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq_5_90_t_percentil_duplo1)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  "
		<< "HC0 = " << (sum(ncobdi_0_90_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi_2_90_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi_3_90_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi_4_90_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi_5_90_t_percentil_duplo1)/nrep)*100 << "\n" << endl;

	saida << "....... BOOTSTRAP T DUPLO (ESQUEMA ERRADO)" << endl;
	saida << "....... NIVEL DE CONFIANCA: 90%" << "\n"	<< endl;
	saida << "COBERTURA:      " 
		<< "HC0 = " << (sum(cob_0_90_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC2 = " << (sum(cob_2_90_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob_3_90_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob_4_90_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob_5_90_t_percentil_duplo)/nrep)*100 << endl;
	saida << "APLITUDE:       " 
		<< "HC0 = " << (sum(ampl_0_90_t_percentil_duplo)/nrep) << ", "
		<< "HC2 = " << (sum(ampl_2_90_t_percentil_duplo)/nrep) << ", "
		<< "HC3 = " << (sum(ampl_3_90_t_percentil_duplo)/nrep) << ", "
		<< "HC4 = " << (sum(ampl_4_90_t_percentil_duplo)/nrep) << ", "
		<< "HC5 = " << (sum(ampl_5_90_t_percentil_duplo)/nrep) << endl;
	saida << "NAO COB. ESQ.:  "
		<< "HC0 = " << (sum(ncobesq_0_90_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq_2_90_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq_3_90_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq_4_90_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq_5_90_t_percentil_duplo)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  " 
		<< "HC0 = " << (sum(ncobdi_0_90_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi_2_90_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi_3_90_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi_4_90_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobdi_5_90_t_percentil_duplo)/nrep)*100 << "\n" << endl;

	saida << "....... BOOTSTRAP PERCENTIL"  << endl;
	saida << "....... NIVEL DE CONFIANCA: 95%" << "\n"	<< endl;
	saida << "COBERTURA:      " << (sum(cob95_percentil)/nrep)*100 << endl;
	saida << "APLITUDE:       " << (sum(ampl95_percentil)/nrep) << endl;
	saida << "NAO COB. ESQ.:  " << (sum(ncobesq95_percentil)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  " << (sum(ncobdi95_percentil)/nrep)*100 << "\n" << endl;

	saida << "....... BOOTSTRAP PERCENTIL DUPLO" << endl;
	saida << "....... NIVEL DE CONFIANCA: 95%" << "\n"	<< endl;
	saida << "COBERTURA:      " << (sum(cob95_percentil_duplo)/nrep)*100 << endl;
	saida << "APLITUDE:       " << (sum(ampl95_percentil_duplo)/nrep) << endl;
	saida << "NAO COB. ESQ.:  " << (sum(ncobesq95_percentil_duplo)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  " << (sum(ncobdi95_percentil_duplo)/nrep)*100 << "\n" << endl;

	saida << "....... BOOTSTRAP T" << endl;
	saida << "....... NIVEL DE CONFIANCA: 95%" << "\n" <<endl;
	saida << "COBERTURA:      " 
		<< "HC0 = " << (sum(cob_0_95_t_percentil)/nrep)*100 << ", " 
		<< "HC2 = " << (sum(cob_2_95_t_percentil)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob_3_95_t_percentil)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob_4_95_t_percentil)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob_5_95_t_percentil)/nrep)*100 << endl;
	saida << "APLITUDE:       " 
		<< "HC0 = " <<(sum(ampl_0_95_t_percentil)/nrep) << ", "
		<< "HC2 = " << (sum(ampl_2_95_t_percentil)/nrep) << ", "
		<< "HC3 = " << (sum(ampl_3_95_t_percentil)/nrep) << ", "
		<< "HC4 = " << (sum(ampl_4_95_t_percentil)/nrep) << ", "
		<< "HC5 = " << (sum(ampl_5_95_t_percentil)/nrep) << endl;
	saida << "NAO COB. ESQ.:  "
		<< "HC0 = " << (sum(ncobesq_0_95_t_percentil)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq_2_95_t_percentil)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq_3_95_t_percentil)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq_4_95_t_percentil)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq_5_95_t_percentil)/nrep)*100 << endl;  
	saida << "NAO COB. DIR.:  " 
		<< "HC0 = " << (sum(ncobdi_0_95_t_percentil)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi_2_95_t_percentil)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi_3_95_t_percentil)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi_4_95_t_percentil)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobdi_5_95_t_percentil)/nrep)*100 << "\n" << endl;

	saida << "....... BOOTSTRAP T DUPLO" << endl;
	saida << "....... NIVEL DE CONFIANCA: 95%" << "\n" << endl;
	saida << "COBERTURA:      " 
		<< "HC0 = " << (sum(cob_0_95_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC2 = " << (sum(cob_2_95_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob_3_95_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob_4_95_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob_5_95_t_percentil_duplo1)/nrep)*100 << endl;
	saida << "APLITUDE:       "
		<< "HC0 = " << (sum(ampl_0_95_t_percentil_duplo1)/nrep) << ", "
		<< "HC2 = " << (sum(ampl_2_95_t_percentil_duplo1)/nrep) << ", "
		<< "HC3 = " << (sum(ampl_3_95_t_percentil_duplo1)/nrep) << ", "
		<< "HC4 = " << (sum(ampl_4_95_t_percentil_duplo1)/nrep) << ", "
		<< "HC5 = " << (sum(ampl_5_95_t_percentil_duplo1)/nrep) << endl;
	saida << "NAO COB. ESQ.:  "
		<< "HC0 = " << (sum(ncobesq_0_95_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq_2_95_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq_3_95_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq_4_95_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq_5_95_t_percentil_duplo1)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  "
		<< "HC0 = " << (sum(ncobdi_0_95_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi_2_95_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi_3_95_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi_4_95_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi_5_95_t_percentil_duplo1)/nrep)*100 << "\n" << endl;

	saida << "....... BOOTSTRAP T DUPLO (ESQUEMA ERRADO)" << endl;
	saida << "....... NIVEL DE CONFIANCA: 95%" << "\n"	<< endl;
	saida << "COBERTURA:      " 
		<< "HC0 = " << (sum(cob_0_95_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC2 = " << (sum(cob_2_95_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob_3_95_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob_4_95_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob_5_95_t_percentil_duplo)/nrep)*100 << endl;
	saida << "APLITUDE:       " 
		<< "HC0 = " << (sum(ampl_0_95_t_percentil_duplo)/nrep) << ", "
		<< "HC2 = " << (sum(ampl_2_95_t_percentil_duplo)/nrep) << ", "
		<< "HC3 = " << (sum(ampl_3_95_t_percentil_duplo)/nrep) << ", "
		<< "HC4 = " << (sum(ampl_4_95_t_percentil_duplo)/nrep) << ", "
		<< "HC5 = " << (sum(ampl_5_95_t_percentil_duplo)/nrep) << endl;
	saida << "NAO COB. ESQ.:  "
		<< "HC0 = " << (sum(ncobesq_0_95_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq_2_95_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq_3_95_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq_4_95_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq_5_95_t_percentil_duplo)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  " 
		<< "HC0 = " << (sum(ncobdi_0_95_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi_2_95_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi_3_95_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi_4_95_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobdi_5_95_t_percentil_duplo)/nrep)*100 << "\n" << endl;

	saida << "....... BOOTSTRAP PERCENTIL"  << endl;
	saida << "....... NIVEL DE CONFIANCA: 99%" << "\n"	<< endl;
	saida << "COBERTURA:       " << (sum(cob99_percentil)/nrep)*100 << endl;
	saida << "APLITUDE:        " << (sum(ampl99_percentil)/nrep) << endl;
	saida << "NAO COB. ESQ.:   " << (sum(ncobesq99_percentil)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:   " << (sum(ncobdi99_percentil)/nrep)*100 << "\n" << endl;

	saida << "....... BOOTSTRAP PERCENTIL DUPLO" << endl;
	saida << "....... NIVEL DE CONFIANCA: 99%" << "\n"	<< endl;
	saida << "COBERTURA:       " << (sum(cob99_percentil_duplo)/nrep)*100 << endl;
	saida << "APLITUDE:        " << (sum(ampl99_percentil_duplo)/nrep) << endl;
	saida << "NAO COB. ESQ.:   " << (sum(ncobesq99_percentil_duplo)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:   " << (sum(ncobdi99_percentil_duplo)/nrep)*100 << "\n" << endl;

	saida << "....... BOOTSTRAP T" << endl;
	saida << "....... NIVEL DE CONFIANCA: 99%" << "\n" <<endl;
	saida << "COBERTURA:       " 
		<< "HC0 = " << (sum(cob_0_99_t_percentil)/nrep)*100 << ", " 
		<< "HC2 = " << (sum(cob_2_99_t_percentil)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob_3_99_t_percentil)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob_4_99_t_percentil)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob_5_99_t_percentil)/nrep)*100 << endl;
	saida << "APLITUDE:        " 
		<< "HC0 = " <<(sum(ampl_0_99_t_percentil)/nrep) << ", "
		<< "HC2 = " << (sum(ampl_2_99_t_percentil)/nrep) << ", "
		<< "HC3 = " << (sum(ampl_3_99_t_percentil)/nrep) << ", "
		<< "HC4 = " << (sum(ampl_4_99_t_percentil)/nrep) << ", "
		<< "HC5 = " << (sum(ampl_5_99_t_percentil)/nrep) << endl;
	saida << "NAO COB. ESQ.:   "
		<< "HC0 = " << (sum(ncobesq_0_99_t_percentil)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq_2_99_t_percentil)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq_3_99_t_percentil)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq_4_99_t_percentil)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq_5_99_t_percentil)/nrep)*100 << endl;  
	saida << "NAO COB. DIR.:   " 
		<< "HC0 = " << (sum(ncobdi_0_99_t_percentil)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi_2_99_t_percentil)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi_3_99_t_percentil)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi_4_99_t_percentil)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobdi_5_99_t_percentil)/nrep)*100 << "\n" << endl;

	saida << "....... BOOTSTRAP T DUPLO" << endl;
	saida << "....... NIVEL DE CONFIANCA: 99%" << "\n" << endl;
	saida << "COBERTURA:       " 
		<< "HC0 = " << (sum(cob_0_99_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC2 = " << (sum(cob_2_99_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob_3_99_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob_4_99_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob_5_99_t_percentil_duplo1)/nrep)*100 << endl;
	saida << "APLITUDE:        "
		<< "HC0 = " << (sum(ampl_0_99_t_percentil_duplo1)/nrep) << ", "
		<< "HC2 = " << (sum(ampl_2_99_t_percentil_duplo1)/nrep) << ", "
		<< "HC3 = " << (sum(ampl_3_99_t_percentil_duplo1)/nrep) << ", "
		<< "HC4 = " << (sum(ampl_4_99_t_percentil_duplo1)/nrep) << ", "
		<< "HC5 = " << (sum(ampl_5_99_t_percentil_duplo1)/nrep) << endl;
	saida << "NAO COB. ESQ.:   "
		<< "HC0 = " << (sum(ncobesq_0_99_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq_2_99_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq_3_99_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq_4_99_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq_5_99_t_percentil_duplo1)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:   "
		<< "HC0 = " << (sum(ncobdi_0_99_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi_2_99_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi_3_99_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi_4_99_t_percentil_duplo1)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi_5_99_t_percentil_duplo1)/nrep)*100 << "\n" << endl;

	saida << "....... BOOTSTRAP T DUPLO (ESQUEMA ERRADO)" << endl;
	saida << "....... NIVEL DE CONFIANCA: 99%" << "\n"	<< endl;
	saida << "COBERTURA:      " 
		<< "HC0 = " << (sum(cob_0_99_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC2 = " << (sum(cob_2_99_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC3 = " << (sum(cob_3_99_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC4 = " << (sum(cob_4_99_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC5 = " << (sum(cob_5_99_t_percentil_duplo)/nrep)*100 << endl;
	saida << "APLITUDE:       " 
		<< "HC0 = " << (sum(ampl_0_99_t_percentil_duplo)/nrep) << ", "
		<< "HC2 = " << (sum(ampl_2_99_t_percentil_duplo)/nrep) << ", "
		<< "HC3 = " << (sum(ampl_3_99_t_percentil_duplo)/nrep) << ", "
		<< "HC4 = " << (sum(ampl_4_99_t_percentil_duplo)/nrep) << ", "
		<< "HC5 = " << (sum(ampl_5_99_t_percentil_duplo)/nrep) << endl;
	saida << "NAO COB. ESQ.:  "
		<< "HC0 = " << (sum(ncobesq_0_99_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobesq_2_99_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobesq_3_99_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobesq_4_99_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobesq_5_99_t_percentil_duplo)/nrep)*100 << endl;
	saida << "NAO COB. DIR.:  " 
		<< "HC0 = " << (sum(ncobdi_0_99_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC2 = " << (sum(ncobdi_2_99_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC3 = " << (sum(ncobdi_3_99_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC4 = " << (sum(ncobdi_4_99_t_percentil_duplo)/nrep)*100 << ", "
		<< "HC5 = " << (sum(ncobdi_5_99_t_percentil_duplo)/nrep)*100 << "\n" << endl;

	saida.close();
	//cout << "\a" << endl; // ALERTA SONORO.
	return 0;
} // AQUI TERMINA A FUNCAO main().


