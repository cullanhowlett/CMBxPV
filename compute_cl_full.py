import os
import numpy as np 
import scipy as sp 
import pandas as pd 
import camb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import integrate, special, interpolate
from camb.sources import GaussianSourceWindow, SplinedSourceWindow

LightSpeed = 299792.458

# Neat stackoverflow function to generate pseudo log-spaced integers (i.e., sort of linear up to some ell, then more logarithmic after that)
# https://stackoverflow.com/questions/12418234/logarithmically-spaced-integers)
def gen_log_space(limit, n):
    result = [1]
    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)

# Generate the camb power spectrum, and redshift distance lookup table, the growth rate/factor and Hubble factor
def get_camb(redshifts, kmax=10.0, dndz=None):

	# Generate the matter power spectrum
	pars = camb.CAMBparams(max_l = 1000)
	pars.InitPower.set_params(As=2.145e-9, ns=0.96)
	pars.set_matter_power(
	    redshifts=redshifts, kmax=kmax, nonlinear=False
	)
	pars.set_cosmology(
	    H0=70.0,
	    omch2=0.11662,
	    ombh2=0.02352,
	    omk=0.0,
	    mnu=0.0,
	    neutrino_hierarchy='degenerate',
	)
	pars.NonLinear = camb.model.NonLinear_both
	#pars.NonLinearModel.set_params(halofit_version='takahashi')
	if dndz is not None:
		pars.SourceWindows = [SplinedSourceWindow(z=dndz["z"].to_numpy(), W=dndz["cz_hist"].to_numpy(), dlog10Ndm=0.0)]
	pars.SourceTerms.counts_density = True
	pars.SourceTerms.counts_redshift = True
	pars.SourceTerms.counts_lensing = False
	pars.SourceTerms.counts_velocity = True
	pars.SourceTerms.counts_radial = False
	pars.SourceTerms.counts_timedelay = True
	pars.SourceTerms.counts_ISW = True
	pars.SourceTerms.counts_potential = True
	pars.SourceTerms.counts_evolve = False
	pars.SourceTerms.line_phot_dipole = False
	pars.SourceTerms.line_phot_quadrupole = False
	pars.SourceTerms.line_basic = True
	pars.SourceTerms.line_distortions = True
	pars.SourceTerms.line_extra = False
	pars.SourceTerms.line_reionization = False
	pars.SourceTerms.use_21cm_mK = True

	# Run CAMB
	results = camb.get_results(pars)

	# Get the power spectrum
	Plin, zin, kin = camb.get_matter_power_interpolator(pars,zmin=redshifts[0],zmax=redshifts[-1],nz_step=len(redshifts),kmax=kmax,nonlinear=True,extrap_kmax=kmax,return_z_k=True,hubble_units=False,k_hunit=False)

	# Get some derived quantities
	Dc = results.comoving_radial_distance(redshifts)
	H = results.hubble_parameter(redshifts)
	fsigma8 = results.get_fsigma8()[::-1]
	sigma8 = results.get_sigmaR(8.0, var1="delta_nonu", var2="delta_nonu")[::-1]
	D = sigma8 / sigma8[0]
	f = fsigma8 / sigma8
	derived = results.get_derived_params()
	Omega_m0 = results.get_Omega('baryon') + results.get_Omega('cdm') + results.get_Omega('nu')

	Cls = None
	if dndz is not None:
		Cls = results.get_source_cls_dict()

	return kin, Plin, Dc, D, f, H, derived["zstar"], Omega_m0, Cls

def WGfunc(dist, ell, ks, dndz):

	bessel = sp.special.spherical_jn(ell, np.outer(ks, dist))
	w = sp.interpolate.splev(dist, dndz, ext=0)

	return w*bessel 

def WUfunc(dist, ell, ks, dndz):

	bessel = sp.special.spherical_jn(ell, np.outer(ks, dist), derivative=True)
	w = sp.interpolate.splev(dist, dndz, ext=0)

	return w*bessel 

def WKfunc(dist, ell, ks, cmbspline):

	bessel = sp.special.spherical_jn(ell, np.outer(ks, dist))
	w = sp.interpolate.splev(dist, cmbspline, ext=0)

	return w*bessel 

def Cellfunc(k, Wspline1, Wspline2, Pspline):

	w1 = sp.interpolate.splev(k, Wspline1, ext=0)
	w2 = sp.interpolate.splev(k, Wspline2, ext=0)
	P = Pspline.P(0.0, k)

	return w1*w2*P

def get_SDSS(Dcspline = None):

	if Dcspline is None:
		zin = np.logspace(-2.5, 1.0, 150)
		kin, Plin, Dc, D, f, H, zstar, Omega_m0, Cls = get_camb(zin)
		Dcspline = sp.interpolate.splrep(zin, Dc)

	skyarea = 7016.41 * (np.pi/180.0)**2
	dndz = pd.read_csv("/Volumes/Work/UQ/SDSS_dists/data/SDSS_nbar_coarse.dat", delim_whitespace = True, names=["cz", "cz_hist", "nbar"], header=None, skiprows=1)
	dndz["z"] = dndz["cz"]/LightSpeed
	dndz["dz"] = sp.interpolate.splev(dndz["z"].to_numpy(), Dcspline)
	#print(dndz["z"], dndz["cz_hist"], np.sum(dndz["cz_hist"]))

	return skyarea, dndz

def get_DESI(Dcspline = None):

	if Dcspline is None:
		zin = np.logspace(-2.5, 1.0, 150)
		kin, Plin, Dc, D, f, H, zstar, Omega_m0, Cls = get_camb(zin)
		Dcspline = sp.interpolate.splrep(zin, Dc)

	skyarea = 14000.0 * (np.pi/180.0)**2
	dndz = pd.read_csv("/Volumes/Work/UQ/DESI/PV/DESI_PV_fa_nbar_fine.dat", delim_whitespace = True, names=["z", "nbar"], header=None, skiprows=1)
	dndz = dndz.drop(dndz[dndz["z"] > 0.15].index)
	dndz["nbar"] *= 0.7**3/1.0e6 
	dndz["dz"] = sp.interpolate.splev(dndz["z"].to_numpy(), Dcspline)
	dzlow = sp.interpolate.splev(dndz["z"].to_numpy() - (dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/2.0, Dcspline)
	dzhi = sp.interpolate.splev(dndz["z"].to_numpy() + (dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/2.0, Dcspline)
	dndz["cz_hist"] = dndz["nbar"] * skyarea*(dzhi**3 - dzlow**3)/3.0
	#print(dndz["z"], dndz["cz_hist"], np.sum(dndz["cz_hist"]))

	return skyarea, dndz

def get_4HS(Dcspline = None):

	if Dcspline is None:
		zin = np.logspace(-2.5, 1.0, 150)
		kin, Plin, Dc, D, f, H, zstar, Omega_m0, Cls = get_camb(zin)
		Dcspline = sp.interpolate.splrep(zin, Dc)

	skyarea = 17000.0 * (np.pi/180.0)**2
	dndz = pd.read_csv("/Volumes/Work/UQ/4HS/4HS_nbar_pv_all.dat", delim_whitespace = True, names=["z", "nbar"], header=None, skiprows=1)
	dndz = dndz.drop(dndz[dndz["z"] > 0.15].index)
	dndz["nbar"] *= 0.7**3/1.0e6 
	dndz["dz"] = sp.interpolate.splev(dndz["z"].to_numpy(), Dcspline)
	dzlow = sp.interpolate.splev(dndz["z"].to_numpy() - (dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/2.0, Dcspline)
	dzhi = sp.interpolate.splev(dndz["z"].to_numpy() + (dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/2.0, Dcspline)
	dndz["cz_hist"] = dndz["nbar"] * skyarea*(dzhi**3 - dzlow**3)/3.0
	#print(dndz["z"], dndz["cz_hist"], np.sum(dndz["cz_hist"]))
	dndz["z"].iloc[-1] = 0.15

	return skyarea, dndz

def get_LSST(Dcspline = None):

	if Dcspline is None:
		zin = np.logspace(-2.5, 1.0, 150)
		kin, Plin, Dc, D, f, H, zstar, Omega_m0, Cls = get_camb(zin)
		Dcspline = sp.interpolate.splrep(zin, Dc)

	skyarea = 18000.0 * (np.pi/180.0)**2
	dndz = pd.read_csv("/Volumes/Work/ICRAR/TAIPAN/Mocks/Lightcone_SN-Optical_Lagos12Mill1_simulator_field1_1_nbar_vel_Jlt19p0_new.dat", delim_whitespace = True, names=["z", "nbar"], header=None, skiprows=1)
	dndz = dndz.drop(dndz[dndz["z"] > 0.5].index)
	dndz["nbar"] *= 0.7**3/1.0e6 
	dndz["dz"] = sp.interpolate.splev(dndz["z"].to_numpy(), Dcspline)
	dzlow = sp.interpolate.splev(dndz["z"].to_numpy() - (dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/2.0, Dcspline)
	dzhi = sp.interpolate.splev(dndz["z"].to_numpy() + (dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/2.0, Dcspline)
	dndz["cz_hist"] = dndz["nbar"] * skyarea*(dzhi**3 - dzlow**3)/3.0
	#print(dndz["z"], dndz["cz_hist"], np.sum(dndz["cz_hist"]))

	return skyarea, dndz

def compute_cl(skyarea, dndz, outputfile):

	if os.path.exists(outputfile):
		cl_full = np.loadtxt(outputfile).T

	else:

		# Get cosmology stuff for CMB lensing (CAMB only allows for up to 150 redshifts, so we spline to a narrower redshift binning)
		kin, Plin, Dc, D, f, H, zstar, Omega_m0, Cls = get_camb(np.array([0.0]))
		zin = np.logspace(-2.5, np.log10(zstar), 150)
		kin, Plin, Dc, D, f, H, zstar, Omega_m0, Cls = get_camb(zin, dndz=dndz)
		zspline = sp.interpolate.splrep(Dc, zin)
		Dspline = sp.interpolate.splrep(Dc, D)
		Hspline = sp.interpolate.splrep(Dc, H)
		fspline = sp.interpolate.splrep(Dc, f)
		Dcspline = sp.interpolate.splrep(zin, Dc)
		cmbspline = sp.interpolate.splrep(Dc, 3.0/2.0*Omega_m0*(H[0]/LightSpeed)**2*Dc*D*(1.0 + zin)*(Dc[-1] - Dc)/Dc[-1])   # All the terms in the CMB convergence kernel
		C_ell_kk_spline = sp.interpolate.splrep(np.linspace(0,1000,1001),Cls["PxP"]/(2.0/np.pi))

		dndz["H"] = sp.interpolate.splev(dndz["dz"].to_numpy(), Hspline)
		dndz["f"] = sp.interpolate.splev(dndz["dz"].to_numpy(), fspline)
		dndz["D"] = sp.interpolate.splev(dndz["dz"].to_numpy(), Dspline)
		dndz["dudz"] = dndz["H"] * dndz["f"] * dndz["cz_hist"] / (1.0 + dndz["z"])
		dnspline = sp.interpolate.splrep(dndz["dz"].to_numpy(), dndz["cz_hist"].to_numpy() * dndz["D"].to_numpy() * dndz["H"].to_numpy()/LightSpeed/(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/np.sum(dndz["cz_hist"].to_numpy()))   # Normalised dn/dz * D(z) * H(z)/c
		duspline = sp.interpolate.splrep(dndz["dz"].to_numpy(), dndz["dudz"].to_numpy() * dndz["D"].to_numpy() * dndz["H"].to_numpy()/LightSpeed/(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/np.sum(dndz["cz_hist"].to_numpy()))   # Normalised dn/dz * H^2(z) * f(z) * D(z) / (1 + z) / c

		# Set up ell and k binning
		ks = np.logspace(-4.0, np.log10(4.0), 2000)
		#ells = gen_log_space(501, 100).astype(int) + 1
		ells = np.linspace(0, 500, 101).astype(int)
		ells[0] = 1
		print(ells)

		# Integral over redshift
		zmax = dndz["z"].to_numpy()[-1]+(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/2.0
		Dmin, Dmax, Dcmb = 0.0, sp.interpolate.splev(zmax, Dcspline), Dc[-1]
		C_gg = np.zeros(len(ells))
		C_uu = np.zeros(len(ells))
		C_ug = np.zeros(len(ells))
		C_kk = np.zeros(len(ells))
		C_gk = np.zeros(len(ells))
		C_uk = np.zeros(len(ells))
		for i, ell in enumerate(ells):
			WG = np.zeros(len(ks))
			WU = np.zeros(len(ks))
			WK = np.zeros(len(ks))
			for k, kval in enumerate(ks):
				WG[k] = sp.integrate.quad(WGfunc, Dmin, Dmax, args=(ell, kval, dnspline), limit=50000, epsrel=1.0e-6, epsabs=1.0e-12)[0]
				WU[k] = sp.integrate.quad(WUfunc, Dmin, Dmax, args=(ell, kval, duspline), limit=50000, epsrel=1.0e-6, epsabs=1.0e-12)[0]
				WK[k] = sp.integrate.quad(WKfunc, Dmin, Dcmb, args=(ell, kval, cmbspline), limit=50000, epsrel=1.0e-6, epsabs=1.0e-12)[0]
			WGspline = sp.interpolate.splrep(ks, ks*WG)
			WUspline = sp.interpolate.splrep(ks, WU)
			WKspline = sp.interpolate.splrep(ks, ks*WK)
			#C_gg[i] = 2.0/np.pi * sp.integrate.quad(Cellfunc, ks[0], ks[-1], args=(WGspline, WGspline, Plin), limit=1000)[0]
			C_uu[i] = 2.0/np.pi * sp.integrate.quad(Cellfunc, ks[0], ks[-1], args=(WUspline, WUspline, Plin), limit=50000, epsrel=1.0e-6, epsabs=1.0e-12)[0]
			#C_ug[i] = 2.0/np.pi * sp.integrate.quad(Cellfunc, ks[0], ks[-1], args=(WUspline, WGspline, Plin), limit=1000)[0]
			C_kk[i] = 2.0/np.pi * sp.integrate.quad(Cellfunc, ks[0], ks[-1], args=(WKspline, WKspline, Plin), limit=50000, epsrel=1.0e-6, epsabs=1.0e-12)[0]
			#C_gk[i] = 2.0/np.pi * sp.integrate.quad(Cellfunc, ks[0], ks[-1], args=(WGspline, WKspline, Plin), limit=1000)[0]
			C_uk[i] = 2.0/np.pi * sp.integrate.quad(Cellfunc, ks[0], ks[-1], args=(WUspline, WKspline, Plin), limit=50000, epsrel=1.0e-6, epsabs=1.0e-12)[0]
			print(ell, C_uu[i], C_uk[i], C_kk[i], sp.interpolate.splev(ell,C_ell_kk_spline))

		# Save to a file
		cl_full = np.c_[ells, C_uu, C_uk, C_kk]
		np.savetxt(outputfile, cl_full, fmt="%d %g %g %g", header="ell    C_uu    C_uk    C_kk") 

	return cl_full

def compute_cl_recon(zmax, outputfile):

	if os.path.exists(outputfile):
		cl_full = np.loadtxt(outputfile).T

	else:

		# Get cosmology stuff for CMB lensing (CAMB only allows for up to 150 redshifts, so we spline to a narrower redshift binning)
		kin, Plin, Dc, D, f, H, zstar, Omega_m0, Cls = get_camb(np.array([0.0]))
		zin = np.logspace(-2.5, np.log10(zstar), 150)
		kin, Plin, Dc, D, f, H, zstar, Omega_m0, Cls = get_camb(zin)
		zspline = sp.interpolate.splrep(Dc, zin)
		Dspline = sp.interpolate.splrep(Dc, D)
		Hspline = sp.interpolate.splrep(Dc, H)
		fspline = sp.interpolate.splrep(Dc, f)
		Dcspline = sp.interpolate.splrep(zin, Dc)
		cmbspline = sp.interpolate.splrep(Dc, 3.0/2.0*Omega_m0*(H[0]/LightSpeed)**2*Dc*D*(1.0 + zin)*(Dc[-1] - Dc)/Dc[-1])   # All the terms in the CMB convergence kernel

		dnspline = sp.interpolate.splrep(Dc, 3.0 * Dc**2 * D)
		duspline = sp.interpolate.splrep(Dc, 3.0 * Dc**2 * D * f*H/(1.0 + zin))

		# Set up ell and k binning
		ks = np.logspace(-4.0, np.log10(4.0), 2000)
		#ells = gen_log_space(501, 100).astype(int) + 1
		ells = np.linspace(0, 500, 101).astype(int)
		ells[0] = 1
		print(ells)

		# Integral over redshift
		Dmin, Dmax, Dcmb = 0.0, sp.interpolate.splev(zmax, Dcspline), Dc[-1]
		C_gg = np.zeros(len(ells))
		C_uu = np.zeros(len(ells))
		C_ug = np.zeros(len(ells))
		C_kk = np.zeros(len(ells))
		C_gk = np.zeros(len(ells))
		C_uk = np.zeros(len(ells))
		for i, ell in enumerate(ells):
			WG = np.zeros(len(ks))
			WU = np.zeros(len(ks))
			WK = np.zeros(len(ks))
			for k, kval in enumerate(ks):
				WG[k] = sp.integrate.quad(WGfunc, Dmin, Dmax, args=(ell, kval, dnspline), limit=5000, epsrel=1.0e-8, epsabs=1.0e-10)[0]
				WU[k] = sp.integrate.quad(WUfunc, Dmin, Dmax, args=(ell, kval, duspline), limit=5000, epsrel=1.0e-8, epsabs=1.0e-10)[0]
				WK[k] = sp.integrate.quad(WKfunc, Dmin, Dcmb, args=(ell, kval, cmbspline), limit=5000, epsrel=1.0e-8, epsabs=1.0e-10)[0]
			WGspline = sp.interpolate.splrep(ks, ks*WG/Dmax**3)
			WUspline = sp.interpolate.splrep(ks, WU/Dmax**3)
			WKspline = sp.interpolate.splrep(ks, ks*WK)
			C_gg[i] = 2.0/np.pi * sp.integrate.quad(Cellfunc, ks[0], ks[-1], args=(WGspline, WGspline, Plin), limit=5000, epsrel=1.0e-8, epsabs=1.0e-10)[0]
			C_uu[i] = 2.0/np.pi * sp.integrate.quad(Cellfunc, ks[0], ks[-1], args=(WUspline, WUspline, Plin), limit=5000, epsrel=1.0e-8, epsabs=1.0e-10)[0]
			#C_ug[i] = 2.0/np.pi * sp.integrate.quad(Cellfunc, ks[0], ks[-1], args=(WUspline, WGspline, Plin), limit=1000)[0]
			C_kk[i] = 2.0/np.pi * sp.integrate.quad(Cellfunc, ks[0], ks[-1], args=(WKspline, WKspline, Plin), limit=5000, epsrel=1.0e-8, epsabs=1.0e-10)[0]
			C_gk[i] = 2.0/np.pi * sp.integrate.quad(Cellfunc, ks[0], ks[-1], args=(WGspline, WKspline, Plin), limit=5000, epsrel=1.0e-8, epsabs=1.0e-10)[0]
			C_uk[i] = 2.0/np.pi * sp.integrate.quad(Cellfunc, ks[0], ks[-1], args=(WUspline, WKspline, Plin), limit=5000, epsrel=1.0e-8, epsabs=1.0e-10)[0]
			print(ell, C_uu[i], C_uk[i], C_kk[i], C_gg[i], C_gk[i])

		# Save to a file
		cl_full = np.c_[ells, C_uu, C_uk, C_kk, C_gg, C_gk]
		np.savetxt(outputfile, cl_full, fmt="%d %g %g %g %g %g", header="ell    C_uu    C_uk    C_kk     C_gg    C_gk") 

	return cl_full

if __name__ == "__main__":

	#skyarea, dndz = get_SDSS()
	#compute_cl(skyarea, dndz, "./C_ell_SDSS.txt")

	#skyarea, dndz = get_DESI()
	#compute_cl(skyarea, dndz, "./C_ell_DESI.txt")

	#skyarea, dndz = get_4HS()
	#compute_cl(skyarea, dndz, "./C_ell_4HS.txt")

	#skyarea, dndz = get_LSST()
	#compute_cl(skyarea, dndz, "./C_ell_LSSTJlt19.txt")

	compute_cl_recon(0.067, "./C_ell_recon_2MRS.txt")


