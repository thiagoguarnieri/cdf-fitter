#https://docs.scipy.org/doc/scipy/reference/stats.html
#https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9})
import scipy
import scipy.stats
import sys
from pprint import pprint
import pandas as pd
import numpy as np
import random
import math

#sys.exit()

#goodness of fit
def errores(arr1,arr2):
	emp_d = np.asarray(arr1)
	theo_d = np.asarray(arr2)
	
	smp_emp = np.random.choice(emp_d,100,replace = False)
	smp_theo = np.random.choice(theo_d,100,replace = False)
	
	adt = scipy.stats.anderson_ksamp([smp_emp,smp_theo])
	
	txt_err = "\nAD statistic = " + str(adt[0]) + "\ncritical = " + str(adt[1]) + "\np-value = " + str(adt[2]) + "\n"
    
	return(txt_err)

def epmf(sample):

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = counts.astype(np.double) / sample.size

    return quantiles, cumprob


def fitting(dist, idx_emp, val_emp, size, locat):
	
	c = 0
	idx_emp_tmp = list()
	val_emp_tmp = list()
	while (True):
		idx_emp_tmp.append(idx_emp[c])
		val_emp_tmp.append(val_emp[c])
		if(idx_emp[c] == size):
			break
		else:
			c = c + 1
		
	#pprint(idx_emp_tmp)
	#sys.exit()
	
	if(dist == 'nbinomial'):
		currval = 100000;
		
		for i in np.arange(0.01, max(idx_emp_tmp), 0.01): 
			for j in np.arange(0.01, 1, 0.01): 
				xa = np.arange(scipy.stats.nbinom.ppf(0.01, i, j,loc=locat), scipy.stats.nbinom.ppf(0.99, i, j,loc=locat))
				dst = scipy.stats.nbinom.pmf(xa, i, j,loc=locat)
				if(len(xa) == len(idx_emp_tmp) and max(xa) == max(idx_emp_tmp) and min(xa) == min(idx_emp_tmp)):
					calc_mse = mse(val_emp_tmp,dst)
					if(calc_mse < currval):
						currval = calc_mse
						opt_dst = dst
						opt_n = i
						opt_p = j
		return currval, opt_dst, idx_emp_tmp, val_emp_tmp, opt_n, opt_p 
		
	if(dist == 'geometric'):
		currval = 100000;
		
		for i in np.arange(0.01, 1, 0.01): 
			xa = np.arange(scipy.stats.geom.ppf(0.01, i,loc=locat), scipy.stats.geom.ppf(0.99, i, loc=locat))
			dst = scipy.stats.geom.pmf(xa, i, loc=locat)
			if(len(xa) == len(idx_emp_tmp) and max(xa) == max(idx_emp_tmp) and min(xa) == min(idx_emp_tmp)):
				calc_mse = mse(val_emp_tmp,dst)
				if(calc_mse < currval):
					currval = calc_mse
					opt_dst = dst
					opt_p = i
		return currval, opt_dst, idx_emp_tmp, val_emp_tmp, opt_p 
	
	if(dist == 'logser'):
		currval = 100000;
	
		for i in np.arange(0.01, 1, 0.01): 
			xa = np.arange(scipy.stats.logser.ppf(0.01, i,loc=locat), scipy.stats.logser.ppf(0.99, i, loc=locat))
			dst = scipy.stats.logser.pmf(xa, i, loc=locat)
			if(len(xa) == len(idx_emp_tmp) and max(xa) == max(idx_emp_tmp) and min(xa) == min(idx_emp_tmp)):
				calc_mse = mse(val_emp_tmp,dst)
				if(calc_mse < currval):
					currval = calc_mse
					opt_dst = dst
					opt_p = i
		return currval, opt_dst, idx_emp_tmp, val_emp_tmp, opt_p 
	
	if(dist == 'planck'):
		currval = 100000;
	
		for i in np.arange(0.01, 1, 0.01): 
			xa = np.arange(scipy.stats.planck.ppf(0.01, i,loc=locat), scipy.stats.planck.ppf(0.99, i, loc=locat))
			dst = scipy.stats.planck.pmf(xa, i, loc=locat)
			if(len(xa) == len(idx_emp_tmp) and max(xa) == max(idx_emp_tmp) and min(xa) == min(idx_emp_tmp)):
				calc_mse = mse(val_emp_tmp,dst)
				if(calc_mse < currval):
					currval = calc_mse
					opt_dst = dst
					opt_p = i
		return currval, opt_dst, idx_emp_tmp, val_emp_tmp, opt_p 
		
	if(dist == 'poisson'):
		currval = 100000;
	
		for i in np.arange(0.01,  max(idx_emp_tmp), 0.01): 
			xa = np.arange(scipy.stats.poisson.ppf(0.01, i,loc=locat), scipy.stats.poisson.ppf(0.99, i, loc=locat))
			dst = scipy.stats.poisson.pmf(xa, i, loc=locat)
			#se os limites teoricos casam com os limites empiricos
			if(len(xa) == len(idx_emp_tmp) and max(xa) == max(idx_emp_tmp) and min(xa) == min(idx_emp_tmp)):
				calc_mse = mse(val_emp_tmp,dst)
				if(calc_mse < currval):
					currval = calc_mse
					opt_dst = dst
					opt_p = i
		return currval, opt_dst, idx_emp_tmp, val_emp_tmp, opt_p 
				
		
#1/n * sum(yp - y)^2
def mse(emp,fitted):
	value = 0
	for idx in range(0, len(emp)): 
		value += (fitted[idx] - emp[idx])**2
	value = value / len(emp)
	return value
	
def rmse(emp,fitted):
	value = 0
	for idx in range(0, len(emp)): 
		value += (fitted[idx] - emp[idx])**2
	value = math.sqrt(value / len(emp))
	return value

#------------------------------------------------------------------------------#
#leitura dos dados
inputcsv = sys.argv[1]
outputfolder = sys.argv[4]
my_data_df = pd.read_csv(inputcsv, header = 0)
feature = sys.argv[5]

#feature estudada (header)
y = my_data_df[feature]

#quantidade de elementos
size = len(y)

#indices
x = scipy.arange(size)

#empirical pmf
pmf_quant, pmf_cum = epmf(y)

#------------------------------------------------------------------------------#
#maximo de sessoes
slic = int(sys.argv[3])
#inicio
globloc = int(sys.argv[2])

#goodness
np.random.seed(12343)
random.seed(12343)

for fittype in (["nbinomial","geometric","logser","planck","poisson"]):
	if(fittype == 'nbinomial'):
		location = globloc
		#fit
		best_mse, best_dist, sliced_emp_idx, sliced_emp, best_n, best_p = fitting(fittype, pmf_quant, pmf_cum, slic, location)		
		
		theo_d = scipy.stats.nbinom.rvs(best_n, best_p, size=len(y), loc=location, random_state = 12343)
		error_rep = errores(y,theo_d)

		#empirical
		fig, ax = plt.subplots(1, 1)
		ax.plot(sliced_emp_idx, sliced_emp,'bo', ms=5, color='r', label='empyrical')
		ax.vlines(sliced_emp_idx, 0, sliced_emp, colors='r', lw=5, alpha=0.5)

		xa = np.arange(scipy.stats.nbinom.ppf(0.01, best_n, best_p,loc=location), scipy.stats.nbinom.ppf(0.99, best_n, best_p,loc=location))
		ax.plot(xa, scipy.stats.nbinom.pmf(xa, best_n, best_p,loc=location), 'bo', ms=5, label=(fittype + "\n" + str(round(best_mse,5)) + error_rep))
		
		np.savetxt(outputfolder+'/pmf_'+fittype+'.csv', best_dist, delimiter=',')
		file = open(outputfolder+'/param_'+fittype+'.csv', 'w')
		file.write('%f\n' % best_n)
		file.write('%f' % best_p)
		file.close()

		#plot
		plt.legend(loc='best')
		plt.xlim(1,slic)
		plt.savefig(outputfolder+'/'+fittype+'.png')
		plt.clf()
		plt.cla()
		plt.close()
		
	if(fittype == 'geometric'):
		location = globloc - 1
		#fit
		best_mse, best_dist, sliced_emp_idx, sliced_emp, best_p = fitting(fittype, pmf_quant, pmf_cum, slic, location)		
		
		theo_d = scipy.stats.geom.rvs(best_p, size=len(y), loc=location, random_state = 12343)
		error_rep = errores(y,theo_d)

		#empirical
		fig, ax = plt.subplots(1, 1)
		ax.plot(sliced_emp_idx, sliced_emp,'bo', ms=5, color='r', label='empyrical')
		ax.vlines(sliced_emp_idx, 0, sliced_emp, colors='r', lw=5, alpha=0.5)

		#geometric test
		xa = np.arange(scipy.stats.geom.ppf(0.01, best_p,loc=location), scipy.stats.geom.ppf(0.99, best_p,loc=location))
		ax.plot(xa, scipy.stats.geom.pmf(xa, best_p,loc=location), 'bo', ms=5, label=(fittype + "\n" + str(round(best_mse,5)) + error_rep))
		
		np.savetxt(outputfolder+'/pmf_'+fittype+'.csv', best_dist, delimiter=',')
		file = open(outputfolder+'/param_'+fittype+'.csv', 'w')
		file.write('%f' % best_p)
		file.close()

		#plot
		plt.legend(loc='best')
		plt.xlim(1,slic)
		plt.savefig(outputfolder+'/'+fittype+'.png')
		plt.clf()
		plt.cla()
		plt.close()
		
	if(fittype == 'logser'):
		location = globloc - 1
		#fit
		best_mse, best_dist, sliced_emp_idx, sliced_emp, best_p = fitting(fittype, pmf_quant, pmf_cum, slic, location)	
		
		theo_d = scipy.stats.logser.rvs(best_p, size=len(y), loc=location, random_state = 12343)
		error_rep = errores(y,theo_d)	

		#empirical
		fig, ax = plt.subplots(1, 1)
		ax.plot(sliced_emp_idx, sliced_emp,'bo', ms=5, color='r', label='empyrical')
		ax.vlines(sliced_emp_idx, 0, sliced_emp, colors='r', lw=5, alpha=0.5)

		#nbinomial test
		xa = np.arange(scipy.stats.logser.ppf(0.01, best_p,loc=location), scipy.stats.logser.ppf(0.99, best_p,loc=location))
		ax.plot(xa, scipy.stats.logser.pmf(xa, best_p,loc=location), 'bo', ms=5, label=(fittype + "\n" + str(round(best_mse,5)) + error_rep))
		
		np.savetxt(outputfolder+'/pmf_'+fittype+'.csv', best_dist, delimiter=',')
		file = open(outputfolder+'/param_'+fittype+'.csv', 'w')
		file.write('%f' % best_p)
		file.close()

		#plot
		plt.legend(loc='best')
		plt.xlim(1,slic)
		plt.savefig(outputfolder+'/'+fittype+'.png')
		plt.clf()
		plt.cla()
		plt.close()
		
	if(fittype == 'planck'):
		location = globloc
		#fit
		best_mse, best_dist, sliced_emp_idx, sliced_emp, best_p = fitting(fittype, pmf_quant, pmf_cum, slic, location)
		
		theo_d = scipy.stats.planck.rvs(best_p, size=len(y), loc=location, random_state = 12343)
		error_rep = errores(y,theo_d)	

		#empirical
		fig, ax = plt.subplots(1, 1)
		ax.plot(sliced_emp_idx, sliced_emp,'bo', ms=5, color='r', label='empyrical')
		ax.vlines(sliced_emp_idx, 0, sliced_emp, colors='r', lw=5, alpha=0.5)

		#nbinomial test
		xa = np.arange(scipy.stats.planck.ppf(0.01, best_p,loc=location), scipy.stats.planck.ppf(0.99, best_p,loc=location))
		ax.plot(xa, scipy.stats.planck.pmf(xa, best_p,loc=location), 'bo', ms=5, label=(fittype + "\n" + str(round(best_mse,5)) + error_rep))
		
		np.savetxt(outputfolder+'/pmf_'+fittype+'.csv', best_dist, delimiter=',')
		file = open(outputfolder+'/param_'+fittype+'.csv', 'w')
		file.write('%f' % best_p)
		file.close()

		#plot
		plt.legend(loc='best')
		plt.xlim(1,slic)
		plt.savefig(outputfolder+'/'+fittype+'.png')
		plt.clf()
		plt.cla()
		plt.close()
	if(fittype == 'poisson'):
		location = globloc
		#fit
		best_mse, best_dist, sliced_emp_idx, sliced_emp, best_p = fitting(fittype, pmf_quant, pmf_cum, slic, location)
		
		theo_d = scipy.stats.poisson.rvs(best_p, size=len(y), loc=location, random_state = 12343)
		error_rep = errores(y,theo_d)	

		#empirical
		fig, ax = plt.subplots(1, 1)
		ax.plot(sliced_emp_idx, sliced_emp,'bo', ms=5, color='r', label='empyrical')
		ax.vlines(sliced_emp_idx, 0, sliced_emp, colors='r', lw=5, alpha=0.5)

		#nbinomial test
		xa = np.arange(scipy.stats.poisson.ppf(0.01, best_p,loc=location), scipy.stats.poisson.ppf(0.99, best_p,loc=location))
		ax.plot(xa, scipy.stats.poisson.pmf(xa, best_p,loc=location), 'bo', ms=5, label=(fittype + "\n" + str(round(best_mse,5)) + error_rep))
		
		np.savetxt(outputfolder+'/pmf_'+fittype+'.csv', best_dist, delimiter=',')
		file = open(outputfolder+'/param_'+fittype+'.csv', 'w')
		file.write('%f' % best_p)
		file.close()

		#plot
		plt.legend(loc='best')
		plt.xlim(1,slic)
		plt.savefig(outputfolder+'/'+fittype+'.png')
		plt.clf()
		plt.cla()
		plt.close()

	
		
#print("len",len(xa),"len",len(idx_emp_tmp))
#print("min",min(xa),"min",min(idx_emp_tmp))
#print("max",max(xa),"max",max(idx_emp_tmp))
