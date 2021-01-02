# Cummulative Distribuition Function Fitter
<p>Python script based on the scipy-stats library to fit an empirical distribution to an theorical statistical distribution.</p>

<p>It receives a CSV dataset with continuous and discrete data and find the parameters for more than 90 distributions. For each distribution, the script also employs</p>

<ul>
  <li> Kolmogorov-Smirnov and Anderson-Darling adherence tests</li>
  <li> RMSE error measurement for the entire curve and Weighted between curve body and tail</li>
</ul>

<p><b>How to call</b><br />
python -W ignore fitter.py [csv_data] [output_folder] [column_header];</p>

<p>E.g: Assuming you have a <i>dataset.csv</i> and wants to store in a folder called <i>results</i>:<br />
 
python -W ignore fitter.py dataset.csv results data;</p>

<b>The script outputs</b>
<ul>
<li>A png for each distribution
<li>CDF percentiles (synthdata_*) to plot in other program languages
<li>The distribution parameters (parameters_*). The two last parameters are related to curve position and scale. The remaining parameters are related to distribution parameters
</ul>

<b>Goodness</b><br/>
<p>The script outputs RMSE and Kolmogorov-Smirnov (KS). If you use RMSE, you must choose the dist. with the smallest value. However, there are cases where the curve has a good visual fit but high RMSE. It happens because some distributions overestimates the last percentile value. Therefore you may use the value up to 95% (curve body) or the weighted RMSE, which reduces the weight of the curve tail in RMSE calculation.</p>

<p>You can use KS instead. In this case you also must choose the smallest one. However, p-value must be above 0.05. P-value indicates if empirical and test samples are from the same theoretical distribution. If you encounter two distributions with similar KS values, then you can look to RMSE to decide which one you will pick. In fact, a good RMSE depends on your application. There are cases where curve tail does not matter and there are cases where making mistakes for values in the tail have a high impact in your system.</p>

<p>If you have problems with p-values even in distributions with a good visual fit, you can try to change the test sample sizer in lines 52 and 53 (second parameter):<br/>
smp_emp = np.random.choice(emp_d,50,replace = False)
smp_theo = np.random.choice(theo_d,50,replace = False)</p>
