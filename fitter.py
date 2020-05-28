#https://docs.scipy.org/doc/scipy/reference/stats.html
#https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit

# testes de pertinencia a uma distribuicao
# Anderson-darling (para cauda) https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test
# kolmogorov-smirnoff (para corpo)

# testes de goodness 
# Kullback-Leibler Divergence (KLD) https://www.rdocumentation.org/packages/LaplacesDemon/versions/16.1.1/topics/KLD
# RMSE, MSE...

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9})
import numpy as np
import pandas as pd
from pprint import pprint
import scipy
import scipy.stats
import sys
import random
import math

#1/n * sum(yp - y)^2
def rmse(emp,fitted, minval, maxval):
	value = 0
	for idx in range(minval,maxval): 
		value = value + ((emp[idx] - fitted[idx])**2)
	value = value / len(emp)
	return math.sqrt(value)
#-------------------------------------------------------------------------------
#gera a cdf dos dados inseridos em csv
def ecdf(sample):

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

#goodness of fit
def errores(arr1,arr2,dst_name,pop_size):
	emp_d = np.asarray(arr1)
	theo_d = np.asarray(arr2)
	
	#definindo intervalos quantis
	quantils = np.arange(0.01,1.01,0.01)
	
	#sample of distributions
	#the size of sample depends on the population. You must look the shape of the curve to decide. For example: bad fit must not have a better goodness than a good fit.
	smp_emp = np.random.choice(emp_d,100,replace = False)
	smp_theo = np.random.choice(theo_d,100,replace = False)
	
	#gerando quantis
	quant_emp = scipy.stats.mstats.mquantiles(emp_d,  prob = quantils)
	quant_theo = scipy.stats.mstats.mquantiles(theo_d,  prob = quantils)
	
	#o gerador costuma gerar valores muito acima dos empiricos. 
	#por isso o MSE reportado pode ser muito alto apesar de a curva 
	#parecer visualmente boa. 
	kst = scipy.stats.ks_2samp(smp_emp,smp_theo)
	mserr = rmse(quant_emp,quant_theo,0,99)
	mserr95 = rmse(quant_emp,quant_theo,0,93)
	mserrTail = rmse(quant_emp,quant_theo,94,99)
	
	txt_err = "\nKS statistic = " + str(kst[0]) + "\np-value = " + str(kst[1]) + "\n"
	txt_err = txt_err + "[RMSE body (95%)] = " + str(round(mserr95,2)) + "\n"
	txt_err = txt_err + "[RMSE tail (5%)] = " + str(round(mserrTail,2)) + "\n"
	txt_err = txt_err + "[RMSE all] = " + str(round(mserr,2)) + "\n"
	txt_err = txt_err + "[Weigh avg RMSE] = " + str(round((mserr95 * 0.95) + (mserrTail * 0.05),2))
	return(txt_err)

#preprocess
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
np.random.seed(12343)
random.seed(12343)

#csv com os dados
inputcsv = sys.argv[1]
#pasta onde serao salvas a cdf, a imagem e os parametros
outputfolder = sys.argv[2]
#indico manualmente a feature (cabecalho do csv)
feature = sys.argv[3]
#leitura dos dados
my_data_df = pd.read_csv(inputcsv, header=0)
#cria uma variavel com os valores da feature desejada (ontime, offtime...)
y = my_data_df[feature]

#fitting
#-------------------------------------------------------------------------------
#quantidade de elementos
tam = len(y)

#indices
x = scipy.arange(tam)

#empirical cdf
cdf_quant, cdf_cum = ecdf(y)

#tipos de distribucao
#dist_names = ['alpha','anglit','arcsine','argus','beta','betaprime','bradford','burr','burr12','cauchy','chi','chi2','cosine','crystalball','dgamma','dweibull','erlang','expon','exponnorm','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy','foldnorm','frechet_r','frechet_l','genlogistic','gennorm','genpareto','genexpon','genextreme','gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant','invgamma','invgauss','invweibull','johnsonsb','johnsonsu','kappa4','kappa3','ksone','kstwobign','laplace','levy','levy_l','levy_stable','logistic','loggamma','loglaplace','lognorm','lomax','maxwell','mielke','moyal','nakagami','ncx2','ncf','nct','norm','norminvgauss','pareto','pearson3','powerlaw','powerlognorm','powernorm','rdist','reciprocal','rayleigh','rice','recipinvgauss','semicircular','skewnorm','t','trapz','triang','truncexpon','truncnorm','tukeylambda','uniform','vonmises','vonmises_line','wald','weibull_min','weibull_max','wrapcauchy'] 
#betaprime: beta type2, lomax: pareto type2, erlang: type of gamma

dist_names = ['norm','lognorm','powerlognorm','expon','exponpow','gamma','logistic','beta','weibull_min','exponweib','pareto'] 

for dist_name in dist_names:
	
	print("testing dist " + dist_name)
	
    #pega o valor do atributo ou ponteiro para a funcao pelo seu nome em string. ex: dist = scipy.stats.norm
	dist = getattr(scipy.stats, dist_name)
    
    #faz o fit
	param = dist.fit(y)
    
    #gera cdf: loc = fator de shift, scale = fator de escala
	cdf_fitted = dist.cdf(x, * param[:-2], loc=param[-2], scale=param[-1])

	#generating synth data
	synth_fitted = dist.rvs(* param[:-2], loc=param[-2], scale=param[-1], size = tam, random_state = 12343)

	#goodness (body and tail)
	rep_err = errores(y,synth_fitted,dist_name,tam)
	
	#gerando quantis pra imprimir cdf
	quantisCdf = scipy.stats.mstats.mquantiles(np.asarray(synth_fitted),  prob = np.arange(0,1,0.01))

	#plotagem
	plt.plot(cdf_quant, cdf_cum, 'r--', linewidth=2.0)
	plt.plot(cdf_fitted, label=dist_name+rep_err)
	plt.legend(loc='lower right')
	plt.xlim(min(y), max(y))
	plt.savefig(outputfolder + '/' + dist_name + '.png')
	plt.clf()
	plt.cla()
	plt.close()

	#save parameters data and goodness
	np.savetxt(outputfolder + '/param_' + dist_name + '.txt', param, delimiter=',')
	#save cdf
	np.savetxt(outputfolder + '/synthdata_' + dist_name + '.csv', quantisCdf, delimiter=',')
