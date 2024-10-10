/* *********************************************************************
                 Modelos de Regressão Heterocedástico
                  Intervalos de Confiança Bootstrap
                      Bootstrap t-percentil
========================================================================
Orientador: Francisco Cribari Neto.
E-mail: cribari@de.ufpe.br 
Orientando: Pedro Rafael Diniz Marinho.
E-mail - pedro.rafael.marinho@gmail.com
Mestrado em Estatística - UFPE.
************************************************************************ */

// NOTAS SOBRE O PROGRAMA:

// Esse programa avalia os intervalos de confianças para os parametros de modelos lineares
// heteroscedasticos construidos pelo metodo boostrap pivotal (t-percentil).

/* Versões das biliotecas utilizadas
   Armadillo - versão 3.2.4
   GSL - versão 4.6.3
*/

/*Compilando o código usando a biblioteca armadillo e gsl
Comando para compilação: g++ -O3 -march=native -mtune=corei7 -funroll-loops -frerun-loop-opt -funroll-all-loops -ffast-math -o bootstrap_t_percentil bootstrap_t_percentil.cpp -lgsl -larmadillo */


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
#define ARMA_DONT_USE_BLAS /*A compilação da biblioteca blas pode ser 32-bits. Logo, não será eficiente a sua utilização.*/
//#define ARMA_USE_LAPACK
#include <iostream>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <time.h>
#include "armadillo" /* Biblioteca de Algebra Linear para C++ */
#include <boost/math/distributions/normal.hpp> /* Utilizado para obtenção dos quantis da normal */
#include <boost/math/distributions/students_t.hpp> /* Utilizado para obtenção dos quantis da t-student */

using namespace arma; 
using namespace std;

namespace myfunctions{
	double quantil(vec dados, double p, int n){
		    vec xx = sort(dados);
		    double x[n];
		    for(int i =0;i<n;i++){
				x[i] = xx(i);
		    }

		    return  gsl_stats_quantile_from_sorted_data(x,1,n,p);
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
// painel de controle. 

// Início do painel de controle 

int nrep = 5000; // NUMERO DE REPLICAS DE MONTE CARLO.
int nrep_boot = 500; // NUMERO DE REPLICAS DO BOOTSTRAP T-PERCENTIL.
int nrep_boot_duplo = 250; // NUMERO DE REPLICAS DO BOOTSTRAP DUPLO T-PERCENTIL. 
int samplesize = 5; // NUMERO DE REPLICACOES DA MATRIZ X. A MATRIZ X SERA REPLICADAS samplisize VEZES.      
int nobs = 20; // NUMERO DE OBSERVACOES. SE esquema = 1, A MATRIZ X TERA nobs LINHAS. NO CASO EM QUE esquema = 2 
                       // A MATRIZ X TERA nobs*samplesize LINHAS.      
int esquema = 2; // SE esquema = 1 A OPCAO samplesize SERA DESCONSIDERADA. DESSA FORMA, A SEGUNDA COLUNA DA MATRIX X
                           // SERA GERADA DIRETAMENTE DE UMA DISTRIBUICAO T COM 3 GRAUS DE LIBERDADE. CASO A ESCOLHA SEJA
                           // samplesize = 2 GERAMOS INICIALMENTE UMA MATRIZ COM nobs LINHAS E POSTERIORMENTE REPLICAMOS ESSA 
                           // MATRIZ samplesize VEZES. 
double lambda = 1;  // BASTA FIXAR O VALOR DE LAMBDA QUE O VALOR DA CONSTANTE "a" É ESCOLHIDO AUTOMATICAMENTE. ASSIM O VALOR DE LAMBDA
                     // TRABALHADO SERA MUITO PROXIMO AO VALOR DE LAMBIDA ESCOLHIDO. POR EXEMPLO, PARA "lambda = 9" O LAMBIDA ESCOLHIDO 
                     // É IGUAL A 9.00017.                      
int dist_erro = 1; // ESCOLHA DA DISTRIBUICAO DOS ERROS: 1: normal; 2: t(3); 3: chi-squared(2)
int dist_t = 1; // 1: rademacher; 2: normal padrao
double niveis[3] = {0.90,0.95,0.99}; // Apenas pode ser definido 3 níveis de confianças, nem mais, nem menos.

// Fim do painel de controle 

int main()
{
    const clock_t tempo_inicial = clock();
    
    double a;

    vec  beta = ones<vec>(2); // VETOR DE UNS. VETOR COM OS PARAMETROS VERDADEIROS.
    
    // Definição do gerador 
    gsl_rng *r;

    //GERADOR UTILZADO. 
     
    r = gsl_rng_alloc(gsl_rng_tt800);  
    
    gsl_rng_set(r,0); // DEFININDO SEMENTE DO GERADOR.
    
    mat X(nobs,2);

    // PRIMEIRO ESQUEMA PARA GERACAO DA MATRIZ X.
    if(esquema==1){
		X = ones<mat>(nobs,2);
		for(int colunas = 1; colunas<2; colunas++)
		{
			for(int linhas = 0; linhas<nobs; linhas++)
			{
				X(linhas,colunas) = gsl_ran_tdist(r,3);
			}
		}
				
	} 

    // SEGUNDO ESQUEMA PARA GERACAO DA MATRIZ X.
    if(esquema==2){
		X = ones<mat>(nobs,2);
		for(int colunas = 1; colunas<2; colunas++)
		{
			for(int linhas = 0; linhas<nobs; linhas++)
			{
				X(linhas,colunas) = gsl_ran_tdist(r,3);
			}
		}
		
		mat X1;
		X1 = X;
		int l=1;
		
		while(l < samplesize){
			X = join_cols(X,X1);
			l++;
		}
		nobs = nobs*samplesize;            
	}    
    
    mat eta = X*beta; // PREDITOR LINEAR.
    
    mat P = inv(sympd(trans(X)*X))*trans(X); // P = (X'X)^{-1}*X'
    
    mat Pt = trans(P); // TRANSPOSTA DA MATRIZ P.
    
    mat H = X*P; // MATRIZ CHAPEU, H = X(X'X)^{-1}X'.
    
    mat h = diagvec(H); // VETOR DE MEDIDAS DE ALAVANCAGEM.
    
    mat hmax = max(h);

    mat g = nobs/2 * h; 
    
    mat weight3 = 1.0/(pow((1.0-h),2.)); // USADA EM HC3.
    
    //Todos elementos do vetor h são maiores que 4? 
    mat  vetor_4 = ones<mat>(X.n_rows) + 3; //Vetor com 1 somado à 3, ou seja, vetor com elementos 4.
    uvec verificando_limite_inferior_g4 = find(g>vetor_4);
    mat g4(nobs,1), weight4(nobs,1);
    
    if(double(verificando_limite_inferior_g4.n_rows)==nobs)
    {
        weight4 = 1.0/pow((1.0-h),4.0);
    }
    
   if(double(verificando_limite_inferior_g4.n_rows)<nobs)
    {
        for(int i=0; i<nobs; i++)
        {
            weight4(i,0) = as_scalar(1.0/pow((1-h(i,0)),g(i,0)));
        }
    } 

    mat gtemp;
    if(as_scalar(nobs*0.7*hmax/1)>4.0){
		gtemp = as_scalar(nobs*0.7*hmax/1);
	}else{
		gtemp = 4.0;
	}
    
    mat vetor_gtemp = zeros<mat>(X.n_rows) + as_scalar(gtemp);
    uvec verificando_limite_inferior_g5 = find(g<vetor_gtemp);
    mat g5(nobs,1), weight5(nobs,1);

    if(verificando_limite_inferior_g5.n_rows < nobs)
    {   
		double expoente;
		expoente = as_scalar(gtemp);
        weight5 = sqrt(1.0/sqrt(pow((1.0-h),expoente)));
    }
    else if(verificando_limite_inferior_g5.n_rows == nobs)
    {
        for(int i=0; i<nobs; i++)
        { 
			weight5(i,0) = as_scalar(sqrt(1.0/sqrt(pow((1.0-h(i,0)),g(i,0)))));
        }
    }

    vec contador(nobs); // 	ARMAZENA OS PONTOS DE ALTA ALAVANCAGEM.
    
    // CONTANDO O NUMERO DE PONTOS DE ALAVANCA.
    for(int d=0; d<nobs; d++){
		if(h(d)>4.0/nobs) contador(d) = 1;
		else contador(d) = 0;
    }
    
    // "A" É UM VETOR COM POSSIVEIS DANDIDATOS A SER O VALOR DE "a" QUE NOS DARA UM LAMBDA PROXIMO
    // DO VALOR DE lambda ESCOLHIDO.
    
    vec A(4000000);
    A(0) = 0;
    for(int s=1;s<4000000;s++){
		A(s) = A(s-1) + 0.000001;
    }
    
    double lambda_utilizado, a_utilizado;
    
    if(lambda == 1){
		 a_utilizado = 0;
    } 
        
    if(lambda != 1){
		int s = 0;
		mat resultado;
		while(lambda_utilizado <= lambda-0.000001){
			 resultado = exp(A(s)*X.col(1));
			 lambda_utilizado = resultado.max()/resultado.min();
			 s++;
		}
	    a_utilizado = as_scalar(A(s));
	}
	
	vec sigma2(nobs), sigma(nobs); 	

    sigma2 = exp(a_utilizado*X.col(1)); // VETOR DE VARIANCIAS. 
    sigma = sqrt(sigma2); // VETOR DE DESVIOS PADROES. 
    lambda = sigma2.max()/sigma2.min(); // RAZAO ENTRE O MAXIMO E O MINIMO DAS VARIANCIAS.     

    // DADOS PRELIMINARES. INFORMACOES SOBRE O NUMERO DE REPLICAS DE MONTE CARLO, BOOTSTRAP, BOOTSTRAP DUPLO.
    // TAMBEM E APRESENTADO INFORMACOES SOBRE O VALOR DE LAMBDA UTILIZADO E O VALOR DE "a" ESCOLHIDO, ASSIM COMO
    // O NUMERO DE PONTOS DE ALTA ALAVANCAGEM.
    
    cout << "\t \t DADOS DA SIMULACAO" << endl << endl;
	cout << ">> [*] nobs = " << nobs << endl;
	cout << ">> [*] lambda = " << lambda << endl;
	cout << ">> [*] a = " << a_utilizado << endl; 
	cout << ">> [*] nrep_boot = " << nrep_boot << endl;
	cout << ">> [*] nrep_boot_duplo = " << nrep_boot_duplo << endl;
    cout << ">> [*] Quant. de pontos de alavanca = " << arma::sum(contador) << endl;    
    
    if(dist_erro == 1) cout << ">> [*] Distribuicao do erro = normal" << endl;
    if(dist_erro == 2) cout << ">> [*] Distribuicao do erro = t(3)" << endl;
    if(dist_erro == 3) cout << ">> [*] Distribuicao do erro = qui-quadrado(2)" << endl;

    if(dist_t == 1) cout << ">> [*] Distribuicao de t^* = rademacher" << endl;
    if(dist_t == 2) cout << ">> [*] Distribuicao de t^* = normal padrao" << endl;
    
    cout << ">> [*] Gerador utilizado = " << "gsl_rng_tt800 da biblioteca GSL" << endl;
    cout << ">> [*] Semente do gerador = 0" << endl;
        
    mat betahat = zeros<mat>(2,nrep); // vector used to store the estimates
    mat tratio = zeros(5, nrep); // matrix used to store t statistics
    
    boost::math::normal dist_n(0.0,1.0); // Distribuição normal padrão;

    double z_intervalo_1 = quantile(dist_n,1-(1-niveis[0])/2); 
    double z_intervalo_2 = quantile(dist_n,1-(1-niveis[1])/2);
    double z_intervalo_3 = quantile(dist_n,1-(1-niveis[2])/2); 

    //Variaveis do bootstrap
    
    vec epsilon_chapeu, y_estrela, beta_chapeu_boot;
    vec beta2_chapeu_boot;
    vec beta2_chapeu_boot_temp(nrep_boot);
    
    vec cob95_t_percentil(nrep), cob99_t_percentil(nrep), cob90_t_percentil(nrep),
        ncobesq95_t_percentil(nrep), ncobdi95_t_percentil(nrep), ncobesq99_t_percentil(nrep), ncobdi99_t_percentil(nrep),
        ncobesq90_t_percentil(nrep), ncobdi90_t_percentil(nrep);

    vec z_estrela(nrep_boot);
    
    double li95_t_percentil, li90_t_percentil, ls90_t_percentil, ls95_t_percentil, li99_t_percentil,ls99_t_percentil;
    
    vec ampl95_t_percentil(nrep), ampl90_t_percentil(nrep), ampl99_t_percentil(nrep);
    
    
    // Variaveis bootstrap duplo
    
    vec epsilon_chapeu_boot_duplo, beta_chapeu_boot_duplo, beta2_chapeu_boot_duplo, beta2_chapeu_boot_duplo_temp(nrep_boot_duplo),
    y_estrela_estrela, t_estrela_estrela, z_estrela_estrela(nrep_boot_duplo);
    
    vec cob95_t_percentil_duplo(nrep_boot), cob99_t_percentil_duplo(nrep_boot), cob90_t_percentil_duplo(nrep_boot),
        ncobesq95_t_percentil_duplo(nrep_boot), ncobdi95_t_percentil_duplo(nrep_boot), ncobesq99_t_percentil_duplo(nrep_boot),
        ncobdi99_t_percentil_duplo(nrep_boot), ncobesq90_t_percentil_duplo(nrep_boot), ncobdi90_t_percentil_duplo(nrep_boot),
        ampl95_t_percentil_duplo(nrep_boot), ampl90_t_percentil_duplo(nrep_boot), ampl99_t_percentil_duplo(nrep_boot);
   
    double li95_t_percentil_duplo, li90_t_percentil_duplo, ls90_t_percentil_duplo, ls95_t_percentil_duplo,
           li99_t_percentil_duplo, ls99_t_percentil_duplo;



    // AQUI COMECA O LACO DE MONTE CARLO.

    vec Y = ones<vec>(nobs);
    mat HC0, HC3, HC4, HC5;
   
    double sigma2hat, diff;
    
    mat produtos = inv(sympd(trans(X)*X))*trans(X);

	// UTILIZADO NA GERACAO DO VALOR DE t^*.
	double numero; 
	
    for(int i=0; i<nrep; i++)
    {
        if(dist_erro == 1)
        {
            for(int v = 0; v<nobs; v++)
            { 
                Y(v) = eta(v) + sigma(v)*gsl_ran_gaussian(r,1.0);
            }
        }
        if(dist_erro == 2)
        {
            for(int v = 0; v<nobs; v++)
            {
                Y(v) = eta(v) + sigma(v)*(gsl_ran_tdist(r,3)/sqrt(1.5));
            }
        }
        
        if(dist_erro == 3)
        {
            for(int v = 0; v<nobs; v++)
            {
                Y(v) = eta(v) + sigma(v)*(gsl_ran_chisq(r,2) - 2.0)/2.0;
            }
        }
        
        double df = nobs-2; // NUMERO DE GRAUS DE LIBERDADE. (nobs-p).
        mat invrXX = inv(sympd(trans(X)*X)); 
        mat temp = invrXX*trans(X)*Y;
        mat resid2 = arma::pow((Y - X*temp),2.0);	
        betahat.col(i) = temp;
        sigma2hat = as_scalar(sum(resid2))/df;
        diff = betahat(1,i)-1; // p=2
        tratio(0,i) = as_scalar(diff/sqrt(sigma2hat*invrXX(1,1))); // OLS // p=2

        mat matrixtemp(nobs,2); // p=2
        	
        resid2 = repmat(resid2,1,2); // p=2
        matrixtemp = resid2%Pt;
        
        //HC0 = P * matrixtemp; // HC0
        //tratio(1,i) = as_scalar(diff/sqrt(HC0(1,1))); // p=2
        //weight3 = repmat(weight3,1,2);
        //HC3 = P * (matrixtemp%weight3);
        //weight3.resize(nobs,1);
        //tratio(2,i) = diff/sqrt(HC3(1,1)); // HC3 stat
        weight4 = repmat(weight4,1,2); // p=2
        HC4 = P * (matrixtemp%weight4);
        //tratio(3,i) = diff/sqrt(HC4(1,1)); // HC4 stat // p=2
        weight4.resize(nobs,1);
        //weight5 = repmat(weight5,1,2);// // p=2
        //HC5 = P * (matrixtemp%weight5);
        //tratio(4,i) = diff/sqrt(HC5(1,1)); // p=2
        //weight5.resize(nobs,1);
		
		epsilon_chapeu = Y - X*temp; // ESTIMATIVAS DOS ERROS.
		mat Xtemp = X*temp; // X*\hat{beta}.
		
        mat HC4_b;
		mat HC4_b_duplo;
			
		// AQUI COMECA O LACO BOOTSTRAP.
		for(int k=0; k<nrep_boot; k++){
			
			vec y_estrela(nobs); // VARIAVEL RESPOSTA UTILIZADA NO BOOTSTRAP.
			vec t_estrela(nobs); // NUMERO ALEATORIO COM MEDIA ZERO E VARINCIA UM.

			if(dist_t == 2){
				for(int t=0; t<nobs; t++){
					numero = gsl_ran_gaussian(r, 1.0);
					t_estrela(t) = numero;
				}
            }

            if(dist_t == 1){
				for(int t=0; t<nobs; t++){
					numero = gsl_rng_uniform(r);
					if(numero<=0.5) t_estrela(t) = -1;
					if(numero>0.5)  t_estrela(t) =  1;
				}
		    }
		    			            
            y_estrela = Xtemp + t_estrela%epsilon_chapeu/sqrt(1-h); // CONFERIDO.
            
            // Aqui temos as estimativas de \hat{{\beta^{*}}_j}. Lembrando que nosso interesse eh \hat{{\beta^{*}}_2}
            
            beta_chapeu_boot = produtos*y_estrela; // Estimativa dos betas estrela (bootstrap). \hat{beta^{*}}
			beta2_chapeu_boot_temp(k) = as_scalar(beta_chapeu_boot(1)); // Vetor com as estimativas b2 bootstrap de uma réplica MC. Em cada,
			                                                              // réplica de bootstrap (k) sera salvo a estimativa de beta2.
			mat matrixtemp_b(nobs,2); 			
			mat resid2_b = arma::pow((y_estrela - X*beta_chapeu_boot),2.0);	
			
			resid2_b = repmat(resid2_b,1,2); // p=2
			
			matrixtemp_b = resid2_b%Pt;
			mat weight4_b = repmat(weight4,1,2); //p=2

			HC4_b = P * (matrixtemp_b%weight4_b);
			
			z_estrela(k) = (as_scalar(beta2_chapeu_boot_temp(k) - temp(1)))/sqrt(HC4_b(1,1));
        
            epsilon_chapeu_boot_duplo = y_estrela - X*beta_chapeu_boot; // SERA UTILIZADO NO BOOTSTRAP DUPLO.
            
        /*    // Aqui comeca o bootstrap duplo
			for(int m=0; m<nrep_boot_duplo; m++){
				
				vec y_estrela_estrela(nobs); // Variavel resposta dentro do bootstrap
				vec t_estrela_estrela(nobs); // Número aleatório com média zero e variância 1.
			
				for(int a=0; a<nobs; a++){
					double numero = gsl_ran_gaussian(b, 1);
					t_estrela_estrela(a) = numero;	
				}

				for(int t=0; t<nobs;t++){
					y_estrela_estrela(t) =  as_scalar((X.row(t))*beta_chapeu_boot + t_estrela_estrela(t)*epsilon_chapeu_boot_duplo(t)/sqrt(1-h(t)));
					//y_estrela_estrela(t) =  as_scalar((X.row(t))*temp + t_estrela_estrela(t)*epsilon_chapeu(t)/sqrt(1-h(t)));
				}

				beta_chapeu_boot_duplo = produtos*y_estrela_estrela; // Estimativa dos betas estrela (bootstrap). \hat{beta^{*}}
				beta2_chapeu_boot_duplo_temp(m) = as_scalar(beta_chapeu_boot_duplo(1)); // Vetor com as estimativas b2 bootstrap de uma replica MC. Em cada,
			                                                              // replica de bootstrap (k) sera salvo a estimativa de beta2.
				//mat matrixtemp_b_duplo(nobs,p);			
				
				//mat resid2_b_duplo = arma::pow((y_estrela_estrela - X*beta_chapeu_boot_duplo),2.0);	
			    //mat resid2_b_duplo = arma::pow((y_estrela_estrela - X*temp),2.0);
				//resid2_b_duplo = repmat(resid2_b_duplo,1,p);
			
				//matrixtemp_b_duplo = resid2_b_duplo%Pt;
				//mat weight4_b_duplo = repmat(weight4,1,p);

				//HC4_b_duplo = P * (matrixtemp_b_duplo%weight4_b_duplo);
				
				//z_estrela_estrela(m) = (as_scalar(beta2_chapeu_boot_duplo_temp(m) - beta_chapeu_boot(1)))/sqrt(HC4_b_duplo(1,1)); 
						
		    } // Aqui termina o laco bootstrap duplo. 
           
           */
            
		}  // AQUI TERMINA O LACO BOOTSTRAP.

        //beta2_chapeu_boot_duplo_temp <= 2*beta2_chapeu_boot_temp(k) - temp(i);
        
        // Laco de MC.  
        // Calculo dos percentuais de cobertura, nao coberturas a esquerda, nao cobertura a direta
        // amplitude media para os niveis de confianca de 90%, 95% e 99% gerado pelo intervalo
        // bootstrap t-percentil.
        

        
        // Confiança de 95%.
        double quantil_inferior95 = myfunctions::quantil(z_estrela,0.975,nrep_boot);
        double quantil_superior95 = myfunctions::quantil(z_estrela,0.025,nrep_boot);
        	    
	    li95_t_percentil = temp(1,0) - quantil_inferior95*sqrt(HC4(1,1));
	    ls95_t_percentil = temp(1,0) - quantil_superior95*sqrt(HC4(1,1));
	    
	    if(beta(1)>= li95_t_percentil && beta(1)<=ls95_t_percentil){
			cob95_t_percentil(i) = 1;
	    } else cob95_t_percentil(i) = 0;

	    if(beta(1) < li95_t_percentil){
			ncobesq95_t_percentil(i) = 1;
        } else ncobesq95_t_percentil(i) = 0;
        
        if(beta(1) > ls95_t_percentil){
			ncobdi95_t_percentil(i) = 1;
        } else ncobdi95_t_percentil(i) = 0;
        
        ampl95_t_percentil(i) = ls95_t_percentil - li95_t_percentil; 
               	    
        // Calculo dos percentuais de cobertura, nao coberturas a esquerda, nao cobertura a direta
        // amplitude media para os niveis de confianca de 90%, 95% e 99% gerado pelo intervalo
        // bootstrap duplo t-percentil.	 

    } // AQUI TERMINA O LACO MONTE CARLO
    
    cout << "Cobertura 95% = " << (arma::sum(cob95_t_percentil)/nrep)*100 << endl;
   
    return 0;
    
}
	
