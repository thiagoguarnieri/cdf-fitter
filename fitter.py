#libraries
import matplotlib
#allow generation of figures in OS which does not have graphical interface
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

#root mean squared error
def rmse(emp,fitted, minval, maxval):
	value = 0
	for idx in range(minval,maxval): 
		value = value + ((emp[idx] - fitted[idx])**2)
	value = value / len(emp)
	return math.sqrt(value)
#-------------------------------------------------------------------------------
#Generate CDF for fitting
def ecdf(sample):

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

#display goodness of fit
#root mean squared error 
#Kolmogorov-smirnov: lower values is better. p-value must be under 0.05
def errores(arr1,arr2,dst_name,pop_size):
	emp_d = np.asarray(arr1)
	theo_d = np.asarray(arr2)
	
	#definindo intervalos quantis
	quantils = np.arange(0.01,1.01,0.01)
	
	#sample of distributions
	#if you experience low p-values, try to change the samples size (second parameter)
	smp_emp = np.random.choice(emp_d,20,replace = False)
	smp_theo = np.random.choice(theo_d,20,replace = False)
	
	#generating percentiles
	quant_emp = scipy.stats.mstats.mquantiles(emp_d,  prob = quantils)
	quant_theo = scipy.stats.mstats.mquantiles(theo_d,  prob = quantils)
	
	kst = scipy.stats.ks_2samp(smp_emp,smp_theo)
	#The MSE has a problem: Errors in curve tail can increase error size
	mserr = rmse(quant_emp,quant_theo,0,99)
	mserr95 = rmse(quant_emp,quant_theo,0,93)
	mserrTail = rmse(quant_emp,quant_theo,94,99)
	
	txt_err = "\nKS statistic = " + str(kst[0]) + "\np-value = " + str(kst[1]) + "\n"
	txt_err = txt_err + "[RMSE body (95%)] = " + str(round(mserr95,4)) + "\n"
	txt_err = txt_err + "[RMSE tail (5%)] = " + str(round(mserrTail,4)) + "\n"
	txt_err = txt_err + "[RMSE all] = " + str(round(mserr,4)) + "\n"
	txt_err = txt_err + "[RMSE weighted] = " + str(round((mserr95 * 0.95) + (mserrTail * 0.05),2))
	return(txt_err)

#preprocess
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
np.random.seed(12343)
random.seed(12343)

#CSV with data
inputcsv = sys.argv[1]
#folder where output will be stored
outputfolder = sys.argv[2]
#indico manualmente a feature (cabecalho do csv)
feature = sys.argv[3]
#leitura dos dados
my_data = pd.read_csv(inputcsv, header=0)
#getting the desired attribute
y = my_data[feature]

#fitting
#-------------------------------------------------------------------------------
#getting number of elements
tam = len(y)

#indexes
x = scipy.arange(tam)

#empirical cdf
cdf_quant, cdf_cum = ecdf(y)

#dist_names = ['alpha','anglit','arcsine','argus','beta','betaprime','bradford','burr','burr12','cauchy','chi','chi2','cosine','crystalball','dgamma','dweibull',
#'erlang','expon','exponnorm','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy','foldnorm','frechet_r','frechet_l','genlogistic','gennorm','genpareto','genexpon',
#'genextreme','gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant',
#'invgamma','invgauss','invweibull','johnsonsb','johnsonsu','kappa4','kappa3','ksone','kstwobign','laplace','levy','levy_l','levy_stable','logistic','loggamma','loglaplace',
#'lognorm','lomax','maxwell','mielke','moyal','nakagami','ncx2','ncf','nct','norm','norminvgauss','pareto','pearson3','powerlaw','powerlognorm','powernorm','rdist','reciprocal',
#'rayleigh','rice','recipinvgauss','semicircular','skewnorm','t','trapz','triang','truncexpon','truncnorm','tukeylambda','uniform','vonmises','vonmises_line','wald','weibull_min',
#'weibull_max','wrapcauchy'] 


dist_names = ['alpha','anglit','arcsine','argus','beta','betaprime','bradford','burr','burr12','cauchy','chi','chi2','cosine','crystalball','dgamma','dweibull',
'erlang','expon','exponnorm','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy','foldnorm','frechet_r','frechet_l','genlogistic','gennorm','genpareto','genexpon',
'genextreme','gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant',
'invgamma','invgauss','invweibull','johnsonsb','johnsonsu','kappa4','kappa3','ksone','kstwobign','laplace','levy','levy_l','logistic','loggamma','loglaplace',
'lognorm','lomax','maxwell','mielke','moyal','nakagami','ncx2','ncf','nct','norm','norminvgauss','pareto','pearson3','powerlaw','rdist','reciprocal',
'rayleigh','rice','recipinvgauss','semicircular','skewnorm','t','trapz','triang','truncexpon','truncnorm','tukeylambda','uniform','vonmises','vonmises_line','wald','weibull_min',
'weibull_max'] 

#testing distributions
for dist_name in dist_names:
	
	print("testing dist " + dist_name)
	
    #select distribution
	dist = getattr(scipy.stats, dist_name)
    
   	#fitting
	param = dist.fit(y)
    
    #generating cdf for comparison
	cdf_fitted = dist.cdf(x, * param[:-2], loc=param[-2], scale=param[-1])

	#generating random variates
	synth_fitted = dist.rvs(* param[:-2], loc=param[-2], scale=param[-1], size = tam, random_state = 12343)

	#goodness (body and tail)
	rep_err = errores(y,synth_fitted,dist_name,tam)
	
	#generating quantiles to print cdf
	quantisCdf = scipy.stats.mstats.mquantiles(np.asarray(synth_fitted),  prob = np.arange(0,1,0.01))

	#ploting
	plt.plot(cdf_quant, cdf_cum, 'r--', linewidth=2.0, label="empirical")
	plt.plot(quantisCdf,np.arange(0,1,0.01), label='Distrib: '+dist_name+rep_err)
	plt.legend(loc='lower right')
	plt.xlim(min(y), max(quantisCdf))
	plt.savefig(outputfolder + '/' + dist_name + '.png')
	plt.clf()
	plt.cla()
	plt.close()

	#save parameters data and goodness (the two last parameters are position and scale, the other are dist. parameters)
	np.savetxt(outputfolder + '/parameters_' + dist_name + '.txt', param, delimiter=',')
	#saving data to allow Cdf plotting in other programs
	np.savetxt(outputfolder + '/synthdata_' + dist_name + '.csv', quantisCdf, delimiter=',')
