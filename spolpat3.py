#Credit: Kishore C. Patra (Based on Aaron Barth's thesis work)
#Compatible with Python 3 (Sergiy Vasylyev)
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import ascii
from scipy import signal
from astropy.io import fits



## rebin into 50 Ang
def rebin(listt, binning=50, n_old=2.): # 25 because data already binned to 2 Ang
	new_list = []
	step = int(binning/n_old)
	for i in range(0,len(listt)-step,step):
		new_list.append(np.median(listt[i:i+step]))
	return np.array(new_list)


def rebin_err(err_array, binning=50, n_old=2.):
	return np.sqrt((n_old/(binning)) * rebin(err_array**2, binning, n_old))



def trim(trace_list, lower, upper):
    
    """
    inputs: a list of traces of the form: [b0,b22,b45,b67,t0,t22,t45,t67]
            lower and upper limits of wavelengths in Angstroms
    
    output: truncated list of traces within lower and 
            upper wavelength limits: [b0,b22,b45,b67,t0,t22,t45,t67]
    """
    
    b0 = trace_list[0][trace_list[0]["col1"] >= lower]
    b0 = b0[b0["col1"] <= upper]
    
    b22 = trace_list[1][trace_list[1]["col1"] >= lower]
    b22 = b22[b22["col1"] <= upper]
    
    b45 = trace_list[2][trace_list[2]["col1"] >= lower]
    b45 = b45[b45["col1"] <= upper]
    
    b67 = trace_list[3][trace_list[3]["col1"] >= lower]
    b67 = b67[b67["col1"] <= upper]
    
    t0 =  trace_list[4][trace_list[4]["col1"] >= lower]
    t0 =  t0[t0["col1"] <= upper]
    
    t22 = trace_list[5][trace_list[5]["col1"] >= lower]
    t22 = t22[t22["col1"] <= upper]
    
    t45 = trace_list[6][trace_list[6]["col1"] >= lower]
    t45 = t45[t45["col1"] <= upper]
    
    t67 = trace_list[7][trace_list[7]["col1"] >= lower]
    t67 = t67[t67["col1"] <= upper]
    
    
    return [b0,b22,b45,b67,t0,t22,t45,t67]



def see_all_traces(trace_list, offset, side="both"):

	# wavelength array
	lambdas = trace_list[0]["col1"] #assumes all lists of same length
	
	# flux array for each trace
	b0f,b22f, b45f, b67f = trace_list[0]["col2"], trace_list[1]["col2"], trace_list[2]["col2"], trace_list[3]["col2"]
	t0f,t22f, t45f, t67f = trace_list[4]["col2"], trace_list[5]["col2"], trace_list[6]["col2"], trace_list[7]["col2"]
	
	# uncertainty in flux array for each trace
	b0u,b22u, b45u, b67u = trace_list[0]["col3"], trace_list[1]["col3"], trace_list[2]["col3"], trace_list[3]["col3"]
	t0u,t22u, t45u, t67u = trace_list[4]["col3"], trace_list[5]["col3"], trace_list[6]["col3"], trace_list[7]["col3"]

	if side == "bottom":
		plt.plot(lambdas, b0f, label="b0f",alpha=0.5)
		plt.plot(lambdas, b45f+1.0*offset,label="b45f",alpha=0.5)
		plt.plot(lambdas, b22f+2.*offset,label="b22f",alpha=0.5)
		plt.plot(lambdas, b67f+3.0*offset,label="b67f",alpha=0.5)

	elif side == "top":
		plt.plot(lambdas, t0f+0.0*offset,label="t0f",alpha=0.5)
		plt.plot(lambdas, t45f+1.*offset,label="t45f",alpha=0.5)
		plt.plot(lambdas, t22f+2.*offset,label="t22f",alpha=0.5)
		plt.plot(lambdas, t67f+3.*offset,label="t67f",alpha=0.5)

	else:
		plt.plot(lambdas, b0f, label="b0f",alpha=0.5)
		plt.plot(lambdas, t0f+1.0*offset,label="t0f",alpha=0.5)
		plt.plot(lambdas, b45f+2.0*offset,label="b45f",alpha=0.5)
		plt.plot(lambdas, t45f+2.*offset,label="t45f",alpha=0.5)
		plt.plot(lambdas, b22f+4.*offset,label="b22f",alpha=0.5)
		plt.plot(lambdas, t22f+5.*offset,label="t22f",alpha=0.5)
		plt.plot(lambdas, b67f+6.0*offset,label="b67f",alpha=0.5)
		plt.plot(lambdas, t67f+7.*offset,label="t67f",alpha=0.5)


	plt.legend(fontsize="x-small")
	if offset < 0.01:
		plt.ylim(min(signal.medfilt(b0f,99)), max(signal.medfilt(b0f,99)))
	else:
		plt.ylim(0,8.*offset)
	plt.xlabel("wavelength (angstroms)")
	plt.ylabel("scaled flux")
	plt.show()




def stokes_and_pol(trace_list, inst_pa, fname, polarizance=False, do_gw=False, do_rebin=True, binning=50, n_old=2., debias=True, flipq=False, correct_pa=True):
	"""
	takes in a list of bottom and top traces
	in the form: [b0,b22,b45,b67,t0,t22,t45,t67]
	Use truncate function to ensure all input 
	list elements have the same lengths
	
	inputs: trace_list
	if polarizance, will calculate polarization as a fraction out of 1
	if do_gw, applies corrective g and w factors
	if do_rebin, will rebin polarization percentange while plotting
	binning is the bin size in Ang
	if debias, will debias the polarization percentage
	if correct_pa, will recalculate q and u by correcting for inst_pa
	inst_pa is an array of smoothed pa output from polarizance test
	returns: [wavelengths,q,u,q_err,u_err,p_percent,p_percent_err, theta, theta_err, total_flux, total_flux_err]
	"""


	# wavelength array
	lambdas = trace_list[0]["col1"] #assumes all lists of same length
	
	# flux array for each trace
	b0f,b22f, b45f, b67f = trace_list[0]["col2"], trace_list[1]["col2"], trace_list[2]["col2"], trace_list[3]["col2"]
	t0f,t22f, t45f, t67f = trace_list[4]["col2"], trace_list[5]["col2"], trace_list[6]["col2"], trace_list[7]["col2"]
	
	# uncertainty in flux array for each trace
	b0u,b22u, b45u, b67u = trace_list[0]["col3"], trace_list[1]["col3"], trace_list[2]["col3"], trace_list[3]["col3"]
	t0u,t22u, t45u, t67u = trace_list[4]["col3"], trace_list[5]["col3"], trace_list[6]["col3"], trace_list[7]["col3"]
	
	
	## q: 0 and 45
	## u: 22 and 67
	
	if  do_gw:
	### w is a corrective factor for flux variations across exposures
		w_q = np.sqrt((b0f*t0f)/(b45f*t45f))
		w_u = np.sqrt((b22f*t22f)/(b67f*t67f))
	
	### g corrects for difference in response between top and bottom trace
		g_q = np.sqrt((b0f*b45f)/(t0f*t45f))
		g_u = np.sqrt((b22f*b67f)/(t22f*t67f))

	### smooth the w and g curve and plot it    
		w_q_filt = signal.medfilt(w_q,99) ## with 99, smoothing over 198 Angstroms
		g_q_filt = signal.medfilt(g_q,99)
		w_u_filt = signal.medfilt(w_u,99) 
		g_u_filt = signal.medfilt(g_u,99)
		
	else:
		### w is a corrective factor for flux variations across exposures
		w_q = np.full(len(lambdas), 1.0)
		w_u = np.full(len(lambdas), 1.0)
	
	### g corrects for difference in response between top and bottom trace
		g_q = np.full(len(lambdas), 1.0)
		g_u = np.full(len(lambdas), 1.0)

	### smooth the w and g curve and plot it    
		w_q_filt = signal.medfilt(w_q,99) ## with 99, smoothing over 198 Angstroms
		g_q_filt = signal.medfilt(g_q,99)
		w_u_filt = signal.medfilt(w_u,99) 
		g_u_filt = signal.medfilt(g_u,99)
	
	
	### plotting g and w vs wavelength
	
	plt.figure(1111)
	plt.plot(lambdas, g_q, alpha=.3,c="g",label="g_q")
	plt.plot(lambdas, g_q_filt, lw=2, c="g",label="filtered g_q")
	plt.plot(lambdas, w_q, alpha=.3, c="r",label="w_q")
	plt.plot(lambdas, w_q_filt, lw=2, c="r",label="filtered w_q")
	plt.legend(loc="best")
	plt.xlabel("wavelength (angstrom)")
	plt.ylabel("dimensionless w and g")
	plt.xlim(min(lambdas), max(lambdas))
	plt.show()
	
	plt.figure(2)
	plt.plot(lambdas, g_u, alpha=.3,c="g",label="g_u")
	plt.plot(lambdas, g_u_filt, lw=2, c="g",label="filtered g_u")
	plt.plot(lambdas, w_u, alpha=.3, c="r",label="w_u")
	plt.plot(lambdas, w_u_filt, lw=2, c="r",label="filtered w_u")
	plt.legend(loc="best")
	plt.xlabel("wavelength (angstrom)")
	plt.ylabel("dimensionless w and g")
	plt.xlim(min(lambdas), max(lambdas))
	plt.show()
	
	
	## set w and g to the smoother function
	w_q = w_q_filt
	g_q = g_q_filt
	w_u = w_u_filt
	g_u = g_u_filt
	


	
	###### now calculate q and u
	#q_b = (b0f - w_q*b45f)/(b0f + w_q*b45f)
	q_b = (b0f - b45f)/(b0f + b45f)
	#q_b_err = np.sqrt(((4.*w_q**2)/(b0f + w_q*b45f)**4.) * (b45f*b45f*b0u*b0u + b0f*b0f*b45u*b45u))
	q_b_err = np.sqrt( (2./(b0f + b45f)**4) * (b45f*b45f*b0u*b0u + b0f*b0f*b45u*b45u + b0f*b0f*b0u*b0u + b45f*b45f*b45u*b45u) )

	#q_t = (w_q*t45f - t0f)/(w_q*t45f + t0f)
	q_t = (t45f - t0f)/(t45f + t0f)
	#q_t_err = np.sqrt(((4.*w_q**2)/(t0f + w_q*t45f)**4.) * (t45f*t45f*t0u*t0u + t0f*t0f*t45u*t45u))
	q_t_err = np.sqrt( (2./(t0f + t45f)**4) * (t45f*t45f*t0u*t0u + t0f*t0f*t45u*t45u + t0f*t0f*t0u*t0u + t45f*t45f*t45u*t45u) )


	if flipq:
		q = -0.5*(q_b + g_q*q_t)
	else:
		q = 0.5*(q_b + g_q*q_t)
	#q = 0.5*(q_b + q_t)
	q_err = np.sqrt(0.25*(q_b_err*q_b_err + g_q*g_q*q_t_err*q_t_err))
	
	

	#u_b = (b22f - w_u*b67f)/(b22f + w_u*b67f)
	u_b = (b22f - b67f)/(b22f + b67f)
	#u_b_err = np.sqrt(((4.*w_u**2)/(b22f + w_u*b67f)**4.) * (b67f*b67f*b22u*b22u + b22f*b22f*b67u*b67u))
	u_b_err = np.sqrt( (2./(b22f + b67f)**4) * (b67f*b67f*b22u*b22u + b22f*b22f*b67u*b67u + b22f*b22f*b22u*b22u + b67f*b67f*b67u*b67u) )

	
	#u_t = (w_u*t67f - t22f)/(w_u*t67f + t22f)
	u_t = (t67f - t22f)/(t67f + t22f)
	#u_t_err = np.sqrt(((4.*w_u**2)/(t22f + w_u*t67f)**4.) * (t67f*t67f*t22u*t22u + t22f*t22f*t67u*t67u))
	u_t_err = np.sqrt( (2./(t22f + t67f)**4) * (t67f*t67f*t22u*t22u + t22f*t22f*t67u*t67u + t22f*t22f*t22u*t22u + t67f*t67f*t67u*t67u) )
	
	u = 0.5*(u_b + g_u*u_t)
	#u = 0.5*(u_b + u_t)
	u_err = np.sqrt(0.25*(u_b_err*u_b_err + g_u*g_u*u_t_err*u_t_err))
	
	
	### plot q and u vs wavelength
	plt.figure(3)
	plt.plot(lambdas, q, label="q")
	plt.plot(lambdas, u, label="u")
	plt.xlabel("wavelength (angstrom)")
	plt.ylabel("q and u ")
	plt.legend(loc="best")
	plt.xlim(min(lambdas), max(lambdas))
	plt.show()

	
	## calculate percentage polarized
	p = np.sqrt(q**2. + u**2.)
	p_err = (1./p)*np.sqrt(q*q*q_err*q_err + u*u*u_err*u_err) #does not account for covariance between q and u
	if polarizance:
		p_percent = p
		p_percent_err = p_err
	else:
		p_percent = 100.*p
		p_percent_err = 100.*p_err
	
 
	## calculate theta
	
	if polarizance:
		theta = 0.5*np.rad2deg(np.arctan(u/q))
	else:
		theta1 = np.rad2deg(np.arctan(np.abs(u/q)))
		theta = []
		for i in range(len(theta1)):
			if u[i]>=0 and q[i]>=0:
				theta.append(theta1[i])
			elif u[i]>0 and q[i]<0:
				theta.append(180.-theta1[i])
			elif u[i]<0 and q[i]<0:
				theta.append(180.+theta1[i])
			elif u[i]<0 and q[i]>0:
				theta.append(360.-theta1[i])
		theta = 0.5*np.array(theta)

	theta_err = (90./np.pi)* (p**(-2))*np.sqrt(q*q*u_err*u_err + u*u*q_err*q_err)



	## correct for inst_pa

	if correct_pa:
		#print(theta, inst_pa)
		qsky = p*np.cos(np.deg2rad(2*(theta - inst_pa)))
		usky = p*np.sin(np.deg2rad(2*(theta - inst_pa)))
		q = np.array(qsky[:])
		u = np.array(usky[:])
		p = np.sqrt(q**2. + u**2.)
		p_err = (1./p)*np.sqrt(q*q*q_err*q_err + u*u*u_err*u_err) #does not account for covariance between q and u
		theta1 = np.rad2deg(np.arctan(np.abs(u/q)))
		theta = []
		for i in range(len(theta1)):
			if u[i]>=0 and q[i]>=0:
				theta.append(theta1[i])
			elif u[i]>0 and q[i]<0:
				theta.append(180.-theta1[i])
			elif u[i]<0 and q[i]<0:
				theta.append(180.+theta1[i])
			elif u[i]<0 and q[i]>0:
				theta.append(360.-theta1[i])
		theta = 0.5*np.array(theta)
		theta_err = (90./np.pi)* (p**(-2))*np.sqrt(q*q*u_err*u_err + u*u*q_err*q_err)


	"""
	if debias:
		deb_pol = []
		for i in range(len(p)):
			sign = q[i]**2. + u[i]**2. - q_err[i]**2 - u_err[i]**2
			if sign < 0:
				deb_pol.append(-np.sqrt(abs(sign)))
			else:
				deb_pol.append(np.sqrt(abs(sign)))
		p = np.array(deb_pol[:])
	"""

	if debias: 
		deb_pol = []
		for i in range(len(p)):
			if p[i] - p_err[i] > 0:
				h = 1.
			else:
				h = 0
			deb_pol.append(p[i] - ((p_err[i]**2)/p[i])*h)
		p_percent = np.array(deb_pol[:])*100.


	## plot polarization
	plt.figure(4, figsize=(20,10))
	
	if do_rebin:
		#pol_err = np.sqrt((2./(2.*binning)) * rebin(p_percent_err**2, binning))
		pol_err = rebin_err(p_percent_err, binning, n_old)
		upper = rebin(p_percent, binning, n_old) + pol_err
		lower = rebin(p_percent, binning, n_old) - pol_err
			
		plt.step(rebin(lambdas, binning, n_old), rebin(p_percent, binning, n_old), "r",lw=2)
		plt.fill_between(rebin(lambdas,binning, n_old), y1=lower, y2=upper,step="pre", color="r",alpha=.2 )
		#plt.plot(lambdas, p_percent, "g",alpha=.3)
		
	else:
		plt.plot(lambdas, p_percent, "k")

	plt.minorticks_on()
	plt.xlabel("Wavelength [$\AA$]")
	if polarizance:
		plt.ylabel("polarizance")
	else:
		plt.ylabel("% polarization")
	if debias:
		plt.title("debiased")
	plt.xlim(min(lambdas), max(lambdas))
	plt.savefig(fname+" polarization.pdf")
	plt.show()

	

	## plot PA
	pa_err = rebin_err(theta_err, binning, n_old)
	pa_upper = rebin(theta, binning, n_old) + pa_err
	pa_lower = rebin(theta, binning, n_old) - pa_err

	plt.figure(5)  
	plt.step(rebin(lambdas, binning, n_old), rebin(theta, binning, n_old), "r",lw=2)
	plt.fill_between(rebin(lambdas,binning, n_old), y1=pa_lower, y2=pa_upper,step="pre", color="r",alpha=.2 )
	plt.xlabel("Wavelength [$\AA$]")
	plt.ylabel("PA (deg)")
	plt.show()



	## total flux spectrum

	"""
	f_q = 0.25*(b0f+t0f+b45f+t45f)
	f_q_err = np.sqrt((b0u**2+t0u**2+b45u**2+t45u**2)/(3))
	f_u = 0.25*(b22f+t22f+b67f+t67f)
	f_u_err = np.sqrt((b22u**2+t22u**2+b67u**2+t67u**2)/(3))
	total_flux = 0.5*(f_u+f_q)
	total_flux_err = np.sqrt(f_q_err**2 + f_u_err**2)
	"""

	f_q = np.median([b0f,t0f,b45f,t45f], axis=0)
	f_q_err = 0.25*np.sqrt(b0u**2+t0u**2+b45u**2+t45u**2)
	f_u = np.median([b22f,t22f,b67f,t67f], axis=0)
	f_u_err = 0.25*np.sqrt(b22u**2+t22u**2+b67u**2+t67u**2)
	total_flux = 0.5*(f_u+f_q)
	total_flux_err = 0.5*np.sqrt(f_q_err**2 + f_u_err**2)
	

	plt.figure(6)
	#plt.plot(rebin(lambdas, 2), rebin(total_flux, 2))
	plt.plot(lambdas, total_flux)
	#plt.xlim(4800, 7500)
	#plt.ylim(10, 90)
	plt.minorticks_on()
	#plt.yscale("log")
	plt.xlabel("Wavelength [$\AA$]")
	plt.ylabel("$F_{\lambda}$")
	plt.xlim(min(lambdas), max(lambdas))
	plt.savefig(fname+" total_flux.pdf")
	plt.show()
	
	return ([lambdas,q,u,q_err,u_err,p_percent,p_percent_err, theta, theta_err, total_flux, total_flux_err])





def write_stokes(array):
	f = open("pol_reduction.txt", "w")
	f.write("# wavelength q u q_err u_err, deb_pol, pol_err, theta, theta_err, total_flux, total_flux_err" + "\n")
	for k in range(len(array[0])):
		f.write(str(array[0][k]) + "    " +
				str(array[1][k]) + "    " +
				str(array[2][k]) + "    " +
				str(array[3][k]) + "    " +
				str(array[4][k]) + "    " +
				str(array[5][k]) + "    " +
				str(array[6][k]) + "    " +
				str(array[7][k]) + "    " +
				str(array[8][k]) + "    " +
				str(array[9][k]) + "    " +
				str(array[10][k]) + "    " +"\n") 
	f.close() 
	

  

def subtract_null(loop_null, loop_obj, nullstdlevel=0, binning=99):
	"""
	nullstdlevel = %polarization of the null std

	return null subtracted obj_loop
	"""

	loop_null = np.array(loop_null)
	loop_obj = np.array(loop_obj)

	q_sub = loop_obj[1] - signal.medfilt(loop_null[1], binning) + nullstdlevel
	u_sub = loop_obj[2] - signal.medfilt(loop_null[2], binning) + nullstdlevel

	return [loop_obj[0], q_sub, u_sub, loop_obj[3], loop_obj[4], loop_obj[5], loop_obj[6], loop_obj[7], loop_obj[8], loop_obj[9], loop_obj[10]]




def combine_loops(loops, debias=True, do_cov=True, do_rebin=False, binning=50, n_old=2., probe=False, probe_qu=None):
	"""
	loops is a list of stokes_and_pol arrays
	returns and writes [lambdas,q,u,q_err,u_err,p_percent,p_percent_err, theta, theta_err, total_flux, total_flux_err]

	"""

	loops = np.array(loops)

	"""
	loops_fid = []
	for i in range(len(loops)):
		binnedloop = []
		for j in range(len(loops[i])):
			if j not in [3,4,6,8,10]:
				binnedloop.append(rebin(loops[i][j], binning=binning, n_old=n_old))
			else:
				binnedloop.append(rebin_err(loops[i][j], binning=binning, n_old=n_old))
		loops_fid.append(binnedloop)

	loops = np.array(loops_fid)
	"""

	print("*********************************************") #
	print("combining number of loops: " + str(len(loops))) #
	print("*********************************************") #

	if len(loops) > 1:

		q_int = []
		u_int = []
		q_err = np.zeros(len(loops[0][0]))
		u_err = np.zeros(len(loops[0][0]))
		f_int = []
		f_err = np.zeros(len(loops[0][0]))


		for i in range(len(loops)):

			q_int.append(loops[i][1])
			u_int.append(loops[i][2])
			f_int.append(loops[i][9])

	
			q_err += loops[i][3]**2.
			u_err += loops[i][4]**2.
			f_err += loops[i][10]**2.



		q_int = np.array(q_int)
		u_int = np.array(u_int)
		f_int = np.array(f_int)

		q_com = np.median(q_int, axis=0)
		u_com = np.median(u_int, axis=0)
		f_com = np.median(f_int, axis=0)
		q_err = np.sqrt(q_err)/(len(loops))
		u_err = np.sqrt(u_err)/(len(loops))
		f_err = np.sqrt(f_err)/(len(loops))
	

	else: 

		#print(type(loops))
		#print(loops[0])
		q_com = loops[0][1]
		u_com = loops[0][2]
		q_err = loops[0][3]
		u_err = loops[0][4]
		f_com = loops[0][9]
		f_err = loops[0][10]

	#### if probe star provided, subtract q and u from probe. 
	if probe: 
		q_com = q_com - probe_qu[0]
		u_com = u_com - probe_qu[1]

	
	if do_rebin:
		q_com = rebin(q_com, binning, n_old)
		u_com = rebin(u_com, binning, n_old)
		q_err = rebin_err(q_err, binning, n_old)
		u_err = rebin_err(u_err, binning, n_old)



	cov = np.cov(q_com, u_com)[0][1] ## covariance between q and u
	cov_norm = np.corrcoef(q_com, u_com)[0][1]
	print("covariance is: "+str(cov)) #
	print("covariance norm is: "+str(cov_norm)) #
	plt.figure(109)
	plt.plot(q_com, u_com, "k.")
	plt.show()

	## calculate pol percentage
	p = np.sqrt(q_com**2. + u_com**2.)
	if do_cov:
		p_err = (1./p)*np.sqrt(q_com*q_com*q_err*q_err + u_com*u_com*u_err*u_err + 2*cov*q_com*u_com)
	else:
		p_err = (1./p)*np.sqrt(q_com*q_com*q_err*q_err + u_com*u_com*u_err*u_err)



	################################################ do complete theta calculation
	## calculate theta
	#theta = np.rad2deg(0.5*np.arctan(u_com/q_com))
	#theta_err = (90./np.pi)* (p**(-2))*np.sqrt(q_com*q_com*u_err*u_err + u_com*u_com*q_err*q_err)
	u = np.array(u_com[:])
	q = np.array(q_com[:])

	theta1 = np.rad2deg(np.arctan(np.abs(u/q)))
	theta = []
	for i in range(len(theta1)):
		if u[i]>=0 and q[i]>=0:
			theta.append(theta1[i])
		elif u[i]>0 and q[i]<0:
			theta.append(180.-theta1[i])
		elif u[i]<0 and q[i]<0:
			theta.append(180.+theta1[i])
		elif u[i]<0 and q[i]>0:
			theta.append(360.-theta1[i])

	theta = 0.5*np.array(theta)
	theta_err = (90./np.pi)* (p**(-2))*np.sqrt(q*q*u_err*u_err + u*u*q_err*q_err)


	"""
	deb_pol = []
	if debias:
		for i in range(len(p)):
			sign = q_com[i]**2. + u_com[i]**2. - q_err[i]**2 - u_err[i]**2
			if sign < 0:
				deb_pol.append(-np.sqrt(abs(sign)))
			else:
				deb_pol.append(np.sqrt(abs(sign)))
		p = np.array(deb_pol[:])
	"""


	if debias: 
		deb_pol = []
		for i in range(len(p)):
			if p[i] - p_err[i] > 0:
				h = 1.
			else:
				h = 0
			deb_pol.append(p[i] - ((p_err[i]**2)/p[i])*h)
		p = np.array(deb_pol[:])


	return [loops[0][0],q_com,u_com,q_err,u_err,p,p_err, theta, theta_err, f_com, f_err]






def blotch(spec, low, high, forward=True):
	## spec is of the form [wavelength, q, q_err]

	ranj = int(high-low)

	corr = []
	for i in range(len(spec[0])):
		if (low < spec[0][i]) and (spec[0][i] < high):
			if forward:
				corr.append(np.random.normal(spec[1][i+ranj], spec[2][i+ranj]))
			else:
				corr.append(np.random.normal(spec[1][i-ranj], spec[2][i-ranj]))

		else:
			corr.append(spec[1][i])

	return corr











################################# deprecated functions ##################################################################################################################3



def test_stokes_and_pol(trace_list, inst_pa, fname, polarizance=False, do_gw=False, do_rebin=True, binning=50, n_old=2., debias=True, flipq=False, correct_pa=True):
	"""
	takes in a list of bottom and top traces
	in the form: [b0,b22,b45,b67,t0,t22,t45,t67]
	Use truncate function to ensure all input 
	list elements have the same lengths
	
	inputs: trace_list
	if polarizance, will calculate polarization as a fraction out of 1
	if do_gw, applies corrective g and w factors
	if do_rebin, will rebin polarization percentange while plotting
	binning is the bin size in Ang
	if debias, will debias the polarization percentage
	if correct_pa, will recalculate q and u by correcting for inst_pa
	inst_pa is an array of smoothed pa output from polarizance test
	returns: [wavelengths,q,u,q_err,u_err,p_percent,p_percent_err, theta, theta_err, total_flux, total_flux_err]
	"""


	# wavelength array
	lambdas = trace_list[0]["col1"] #assumes all lists of same length
	
	# flux array for each trace
	b0f,b22f, b45f, b67f = trace_list[0]["col2"], trace_list[1]["col2"], trace_list[2]["col2"], trace_list[3]["col2"]
	t0f,t22f, t45f, t67f = trace_list[4]["col2"], trace_list[5]["col2"], trace_list[6]["col2"], trace_list[7]["col2"]
	
	# uncertainty in flux array for each trace
	b0u,b22u, b45u, b67u = trace_list[0]["col3"], trace_list[1]["col3"], trace_list[2]["col3"], trace_list[3]["col3"]
	t0u,t22u, t45u, t67u = trace_list[4]["col3"], trace_list[5]["col3"], trace_list[6]["col3"], trace_list[7]["col3"]
	
	
	## q: 0 and 45
	## u: 22 and 67
	
	if  do_gw:
	### w is a corrective factor for flux variations across exposures
		w_q = np.sqrt((b0f*t0f)/(b45f*t45f))
		w_u = np.sqrt((b22f*t22f)/(b67f*t67f))
	
	### g corrects for difference in response between top and bottom trace
		g_q = np.sqrt((b0f*b45f)/(t0f*t45f))
		g_u = np.sqrt((b22f*b67f)/(t22f*t67f))

	### smooth the w and g curve and plot it    
		w_q_filt = signal.medfilt(w_q,99) ## with 99, smoothing over 198 Angstroms
		g_q_filt = signal.medfilt(g_q,99)
		w_u_filt = signal.medfilt(w_u,99) 
		g_u_filt = signal.medfilt(g_u,99)
		
	else:
		### w is a corrective factor for flux variations across exposures
		w_q = np.full(len(lambdas), 1.0)
		w_u = np.full(len(lambdas), 1.0)
	
	### g corrects for difference in response between top and bottom trace
		g_q = np.full(len(lambdas), 1.0)
		g_u = np.full(len(lambdas), 1.0)

	### smooth the w and g curve and plot it    
		w_q_filt = signal.medfilt(w_q,99) ## with 99, smoothing over 198 Angstroms
		g_q_filt = signal.medfilt(g_q,99)
		w_u_filt = signal.medfilt(w_u,99) 
		g_u_filt = signal.medfilt(g_u,99)
	
	
	### plotting g and w vs wavelength
	
	plt.figure(1111)
	plt.plot(lambdas, g_q, alpha=.3,c="g",label="g_q")
	plt.plot(lambdas, g_q_filt, lw=2, c="g",label="filtered g_q")
	plt.plot(lambdas, w_q, alpha=.3, c="r",label="w_q")
	plt.plot(lambdas, w_q_filt, lw=2, c="r",label="filtered w_q")
	plt.legend(loc="best")
	plt.xlabel("wavelength (angstrom)")
	plt.ylabel("dimensionless w and g")
	plt.xlim(min(lambdas), max(lambdas))
	plt.show()
	
	plt.figure(2)
	plt.plot(lambdas, g_u, alpha=.3,c="g",label="g_u")
	plt.plot(lambdas, g_u_filt, lw=2, c="g",label="filtered g_u")
	plt.plot(lambdas, w_u, alpha=.3, c="r",label="w_u")
	plt.plot(lambdas, w_u_filt, lw=2, c="r",label="filtered w_u")
	plt.legend(loc="best")
	plt.xlabel("wavelength (angstrom)")
	plt.ylabel("dimensionless w and g")
	plt.xlim(min(lambdas), max(lambdas))
	plt.show()
	
	
	## set w and g to the smoother function
	w_q = w_q_filt
	g_q = g_q_filt
	w_u = w_u_filt
	g_u = g_u_filt
	


	
	###### now calculate q and u
	q_b = (b0f - w_q*t0f)/(b0f + w_q*t0f)
	#q_b = (b0f - b45f)/(b0f + b45f)
	q_b_err = np.sqrt(((4.*w_q**2)/(b0f + w_q*b45f)**4.) * (b45f*b45f*b0u*b0u + b0f*b0f*b45u*b45u))
	
	q_t = (w_q*t45f - b45f)/(w_q*t45f + b45f)
	#q_t = (t45f - t0f)/(t45f + t0f)
	q_t_err = np.sqrt(((4.*w_q**2)/(t0f + w_q*t45f)**4.) * (t45f*t45f*t0u*t0u + t0f*t0f*t45u*t45u))
	


	if flipq:
		#q = -0.5*(q_b + g_q*q_t)
		q = q_b #-0.5*(q_b + g_q*q_t)
	else:
		q = 0.5*(q_b + g_q*q_t)
	#q = 0.5*(q_b + q_t)
	q_err = np.sqrt(0.25*(q_b_err*q_b_err + g_q*g_q*q_t_err*q_t_err))
	
	

	u_b = (b22f - w_u*t22f)/(b22f + w_u*t22f)
	#u_b = (b22f - b67f)/(b22f + b67f)
	u_b_err = np.sqrt(((4.*w_u**2)/(b22f + w_u*b67f)**4.) * (b67f*b67f*b22u*b22u + b22f*b22f*b67u*b67u))
	
	u_t = (w_u*t67f - b67f)/(w_u*t67f + b67f)
	#u_t = (t67f - t22f)/(t67f + t22f)
	u_t_err = np.sqrt(((4.*w_u**2)/(t22f + w_u*t67f)**4.) * (t67f*t67f*t22u*t22u + t22f*t22f*t67u*t67u))
	
	#u = 0.5*(u_b + g_u*u_t)
	u = u_b
	#u = 0.5*(u_b + u_t)
	u_err = np.sqrt(0.25*(u_b_err*u_b_err + g_u*g_u*u_t_err*u_t_err))
	#u_err = np.sqrt(0.25*(u_b_err*u_b_err)


	
	### plot q and u vs wavelength
	plt.figure(3)
	plt.plot(lambdas, q, label="q")
	plt.plot(lambdas, u, label="u")
	plt.xlabel("wavelength (angstrom)")
	plt.ylabel("q and u ")
	plt.legend(loc="best")
	plt.xlim(min(lambdas), max(lambdas))
	plt.show()

	
	## calculate percentage polarized
	p = np.sqrt(q**2. + u**2.)
	p_err = (1./p)*np.sqrt(q*q*q_err*q_err + u*u*u_err*u_err) #does not account for covariance between q and u
	if polarizance:
		p_percent = p
		p_percent_err = p_err
	else:
		p_percent = 100.*p
		p_percent_err = 100.*p_err
	
 
	## calculate theta
	
	if polarizance:
		theta = 0.5*np.rad2deg(np.arctan(u/q))
	else:
		theta1 = np.rad2deg(np.arctan(np.abs(u/q)))
		theta = []
		for i in range(len(theta1)):
			if u[i]>=0 and q[i]>=0:
				theta.append(theta1[i])
			elif u[i]>0 and q[i]<0:
				theta.append(180.-theta1[i])
			elif u[i]<0 and q[i]<0:
				theta.append(180.+theta1[i])
			elif u[i]<0 and q[i]>0:
				theta.append(360.-theta1[i])
		theta = 0.5*np.array(theta)

	theta_err = (90./np.pi)* (p**(-2))*np.sqrt(q*q*u_err*u_err + u*u*q_err*q_err)



	## correct for inst_pa

	if correct_pa:
		#print(theta, inst_pa)
		qsky = p*np.cos(np.deg2rad(2*(theta - inst_pa)))
		usky = p*np.sin(np.deg2rad(2*(theta - inst_pa)))
		q = np.array(qsky[:])
		u = np.array(usky[:])
		p = np.sqrt(q**2. + u**2.)
		p_err = (1./p)*np.sqrt(q*q*q_err*q_err + u*u*u_err*u_err) #does not account for covariance between q and u
		theta1 = np.rad2deg(np.arctan(np.abs(u/q)))
		theta = []
		for i in range(len(theta1)):
			if u[i]>=0 and q[i]>=0:
				theta.append(theta1[i])
			elif u[i]>0 and q[i]<0:
				theta.append(180.-theta1[i])
			elif u[i]<0 and q[i]<0:
				theta.append(180.+theta1[i])
			elif u[i]<0 and q[i]>0:
				theta.append(360.-theta1[i])
		theta = 0.5*np.array(theta)
		theta_err = (90./np.pi)* (p**(-2))*np.sqrt(q*q*u_err*u_err + u*u*q_err*q_err)


	"""
	if debias:
		deb_pol = []
		for i in range(len(p)):
			sign = q[i]**2. + u[i]**2. - q_err[i]**2 - u_err[i]**2
			if sign < 0:
				deb_pol.append(-np.sqrt(abs(sign)))
			else:
				deb_pol.append(np.sqrt(abs(sign)))
		p = np.array(deb_pol[:])
	"""

	if debias: 
		deb_pol = []
		for i in range(len(p)):
			if p[i] - p_err[i] > 0:
				h = 1.
			else:
				h = 0
			deb_pol.append(p[i] - ((p_err[i]**2)/p[i])*h)
		p_percent = np.array(deb_pol[:])*100.


	## plot polarization
	plt.figure(4)
	
	if do_rebin:
		#pol_err = np.sqrt((2./(2.*binning)) * rebin(p_percent_err**2, binning))
		pol_err = rebin_err(p_percent_err, binning, n_old)
		upper = rebin(p_percent, binning, n_old) + pol_err
		lower = rebin(p_percent, binning, n_old) - pol_err
			
		plt.step(rebin(lambdas, binning, n_old), rebin(p_percent, binning, n_old), "r",lw=2)
		plt.fill_between(rebin(lambdas,binning, n_old), y1=lower, y2=upper,step="pre", color="r",alpha=.2 )
		#plt.plot(lambdas, p_percent, "g",alpha=.3)
		
	else:
		plt.plot(lambdas, p_percent, "k")

	plt.minorticks_on()
	plt.xlabel("Wavelength [$\AA$]")
	if polarizance:
		plt.ylabel("polarizance")
	else:
		plt.ylabel("% polarization")
	if debias:
		plt.title("debiased")
	plt.xlim(min(lambdas), max(lambdas))
	plt.savefig(fname+" polarization.pdf")
	plt.show()

	

	## plot PA
	pa_err = rebin_err(theta_err, binning, n_old)
	pa_upper = rebin(theta, binning, n_old) + pa_err
	pa_lower = rebin(theta, binning, n_old) - pa_err

	plt.figure(5)  
	plt.step(rebin(lambdas, binning, n_old), rebin(theta, binning, n_old), "r",lw=2)
	plt.fill_between(rebin(lambdas,binning, n_old), y1=pa_lower, y2=pa_upper,step="pre", color="r",alpha=.2 )
	plt.xlabel("Wavelength [$\AA$]")
	plt.ylabel("PA (deg)")
	plt.show()



	## total flux spectrum

	f_q = 0.25*(b0f+t0f+b45f+t45f)
	f_q_err = np.sqrt((b0u**2+t0u**2+b45u**2+t45u**2)/(3))
	f_u = 0.25*(b22f+t22f+b67f+t67f)
	f_u_err = np.sqrt((b22u**2+t22u**2+b67u**2+t67u**2)/(3))
	total_flux = 0.5*(f_u+f_q)
	total_flux_err = np.sqrt(f_q_err**2 + f_u_err**2)

	
	plt.figure(6)
	#plt.plot(rebin(lambdas, 2), rebin(total_flux, 2))
	plt.plot(lambdas, total_flux)
	#plt.xlim(4800, 7500)
	#plt.ylim(10, 90)
	plt.minorticks_on()
	#plt.yscale("log")
	plt.xlabel("Wavelength [$\AA$]")
	plt.ylabel("$F_{\lambda}$")
	plt.xlim(min(lambdas), max(lambdas))
	plt.savefig(fname+" total_flux.pdf")
	plt.show()
	
	return ([lambdas,q,u,q_err,u_err,p_percent,p_percent_err, theta, theta_err, total_flux, total_flux_err])





"""
### function to convert .fits to .flm if needed

def fits2flm(files):
	
	for i in range(len(files)):
		
		sp = fits.open(files[i])
		header = sp[0].header
		index = np.arange(header['NAXIS1'])
		cdelt1 = header["CDELT1"]
		crval1 = header["CRVAL1"]
		print(cdelt1) #
		print(crval1) #
		wave = index*cdelt1 + crval1
		flux = np.array(sp[0].data[0])
		err = np.array(sp[0].data[1])
		
		plt.figure(i)
		plt.plot(wave, flux)
		plt.show()
		
		
		f = open(files[i][:-4]+"flm","w") 
		for k in range(len(wave)):
			f.write(str(wave[k]) + "    " + str(flux[k]) + "    " + str(err[k]) + "\n") 
		f.close() 

		
#files = ["bot0.0lp1.fits","bot0.0lp2.fits","bot0.0lp3.fits","bot22.5lp1.fits","bot22.5lp2.fits","bot22.5lp3.fits",
		#"bot45.0lp1.fits","bot45.0lp2.fits","bot45.0lp3.fits", "bot67.5lp1.fits", "bot67.5lp2.fits", "bot67.5lp3.fits"]

#fits2flm(files)





filename = "botfilter0.0.flm"
f = ascii.read(filename)
#print(f["col1"])

g = open("top" + filename[3:],"w") 
for k in range(len(f["col1"])):
	g.write(str(f["col1"][k]) + "    " + str(0.0)+ "    " + str(0.0) + "\n") 
g.close() 


filename = "topfilter45.0.flm"
f = ascii.read(filename)
#print(f["col1"])

g = open("bot" + filename[3:],"w") 
for k in range(len(f["col1"])):
	g.write(str(f["col1"][k]) + "    " + str(0.0)+ "    " + str(0.0) + "\n") 
g.close() 

"""
