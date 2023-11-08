from compute_cl_full import *

def Nuufunc(dist, dnspline, alpha):

	return (alpha * 70.0 * dist)*sp.interpolate.splev(dist, dnspline)

def plot_dndz():

	redvals = np.linspace(0.0, 0.70, 280)
	colors = ['r', 'b', 'g', 'purple']
	#labels = [r"$\mathrm{SDSS\,(Howlett\!+\!22):\,}N=34,\!059$",
	labels = [r"$\mathrm{DESI:\,}N \approx 186,\!000$",
			  r"$\mathrm{4HS:\,}N \approx 450,\!000$",
			  r"$\mathrm{LSST+J<19:\,}N \approx 161,\!000$",
			  r"$\mathrm{Reconstruction}:\,z \leq 0.15$"]

	# Compute the CMB lensing kernel
	kin, Plin, Dc, D, f, H, zstar, Omega_m0, Cls = get_camb(np.array([0.0]))
	zin = np.logspace(-2.5, np.log10(zstar), 150)
	kin, Plin, Dc, D, f, H, zstar, Omega_m0, Cls = get_camb(zin)
	zspline = sp.interpolate.splrep(Dc, zin)
	Dspline = sp.interpolate.splrep(Dc, D)
	Hspline = sp.interpolate.splrep(Dc, H)
	fspline = sp.interpolate.splrep(Dc, f)
	Dcspline = sp.interpolate.splrep(zin, Dc)

	cmbkernel = 3.0/2.0*Omega_m0*H[0]**2/LightSpeed/H*Dc*(1.0 + zin)*(Dc[-1] - Dc)/Dc[-1]
	cmbspline = sp.interpolate.splrep(zin, cmbkernel)   # All the terms in the CMB convergence kernel
	#cmbnorm = sp.integrate.simps(cmbkernel, zin)
	cmbnorm = np.amax(cmbkernel)
	print(cmbnorm)

	dnnorms, dnsplines = [], []
	for (skyarea, dndz) in [get_DESI(Dcspline=Dcspline), get_4HS(Dcspline=Dcspline), get_LSST(Dcspline=Dcspline)]:
		dndz_normalised = dndz["cz_hist"].to_numpy()/np.sum(dndz["cz_hist"].to_numpy())/(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])    # Normalised dn/dz
		dnnorms.append(np.amax(dndz_normalised))
		dnsplines.append(sp.interpolate.splrep(dndz["z"].to_numpy(), dndz_normalised))

	dnnorm = np.max(dnnorms)

	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.13, 0.83, 0.85])
	for dnspline, color, label in zip(dnsplines, colors, labels):
		ax.errorbar(redvals, sp.interpolate.splev(redvals, dnspline, ext=1)/dnnorm, color=color, ls='-', lw=1.3, label=label)
	index = np.where(zin < 0.15)
	dnspline = sp.interpolate.splrep(zin[index], 3.0 * Dc[index]**2 * LightSpeed / H[index] / sp.interpolate.splev(0.15, Dcspline)**3)
	ax.errorbar(redvals, sp.interpolate.splev(redvals, dnspline, ext=1)/dnnorm, color=colors[-1], ls='-', lw=1.3, label=labels[-1])
	ax.errorbar(redvals, sp.interpolate.splev(redvals, cmbspline, ext=1)/cmbnorm, color='k', ls='--', lw=1.3)
	ax.set_xlim(0.0, 0.52)
	#ax.set_ylim(0.0, 1.05)
	ax.set_xlabel(r"$z$", fontsize=14)
	ax.set_ylabel(r"$dn/dz\,\,\mathrm{(Normalised)}$", fontsize=14)
	ax.tick_params('both',length=10, which='major', width=1.3, direction='in')
	ax.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for axis in ['top','left','bottom','right']:
		ax.spines[axis].set_linewidth(1.3)
	for tick in ax.xaxis.get_ticklabels():
		tick.set_fontsize(12)
	for tick in ax.yaxis.get_ticklabels():
		tick.set_fontsize(12)
	ax.yaxis.set_ticks_position('both')
	plt.legend(fontsize=12, loc='upper right')
	plt.show()

def plot_delta():

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

	# Read in the DESI PV survey dn/dz
	skyarea, dndz = get_DESI(Dcspline=Dcspline)
	dndz["cz_hist"] = 2400.0
	dndz["H"] = sp.interpolate.splev(dndz["dz"].to_numpy(), Hspline)
	dndz["f"] = sp.interpolate.splev(dndz["dz"].to_numpy(), fspline)
	dndz["D"] = sp.interpolate.splev(dndz["dz"].to_numpy(), Dspline)
	dndz["cz_hist"] = 2400.0 * dndz["H"][0] / dndz["H"] * dndz["D"][0] / dndz["D"]
	dndz["dudz"] = dndz["H"] * dndz["f"] * dndz["cz_hist"] / (1.0 + dndz["z"])
	dnspline = sp.interpolate.splrep(dndz["dz"].to_numpy(), dndz["cz_hist"].to_numpy() * dndz["D"].to_numpy() * dndz["H"].to_numpy()/LightSpeed/(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/np.sum(dndz["cz_hist"].to_numpy()))   # Normalised dn/dz * D(z) * H(z)/c
	duspline = sp.interpolate.splrep(dndz["dz"].to_numpy(), dndz["dudz"].to_numpy() * dndz["D"].to_numpy() * dndz["H"].to_numpy()/LightSpeed/(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/np.sum(dndz["cz_hist"].to_numpy()))   # Normalised dn/dz * H^2(z) * f(z) * D(z) / (1 + z) / c

	# Set up some k and ell binning
	ks = np.logspace(np.log10(1.0e-4), np.log10(4.0), 1000)
	ells = [4, 45, 180]
	labels = [r"$\ell = 4$", r"$\ell = 45$", r"$\ell = 180$"]
	zmax = dndz["z"].to_numpy()[-1]+(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/2.0
	Dmin, Dmax, Dcmb = 0.0, sp.interpolate.splev(zmax, Dcspline), Dc[-1]
	outfiles = [str("./Delta_k_constdn_%03d.dat" % ell) for ell in ells]

	WG = np.zeros((len(ells), len(ks)))
	WU = np.zeros((len(ells), len(ks)))
	WK = np.zeros((len(ells), len(ks)))

	for i, ell in enumerate(ells):

		# Compute the delta functions unless they are already saved in a file
		if os.path.exists(outfiles[i]):
			Win = np.loadtxt(outfiles[i])
			WG[i] = Win[:,1]
			WU[i] = Win[:,2]
			WK[i] = Win[:,3]
		else:
			for k, kval in enumerate(ks):
				print(ell, k)
				WG[i, k] = sp.integrate.quad(WGfunc, Dmin, Dmax, args=(ell, kval, dnspline), limit=50000, epsabs=1.0e-10)[0]
				WU[i, k] = sp.integrate.quad(WUfunc, Dmin, Dmax, args=(ell, kval, duspline), limit=50000, epsabs=1.0e-10)[0]
				WK[i, k] = sp.integrate.quad(WKfunc, Dmin, Dcmb, args=(ell, kval, cmbspline), limit=50000, epsabs=1.0e-10)[0]

			# Save the deltas to a file
			np.savetxt(outfiles[i], np.c_[ks, WG[i,:], WU[i,:], WK[i,:]], fmt="%g %g %g %g", header="k    Delta_g    Delta_u    Delta_k")

	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.13, 0.83, 0.82])
	for i, (ell, c) in enumerate(zip(ells, ['r', 'g', 'b'])):
		# Remove points that seem numerically wrong
		index = np.where(WG[i,1:-1]*WK[i,1:-1] > 3.0*(ks[1:-1]-ks[:-2])*(WG[i,2:]*WK[i,2:] - WG[i,:-2]*WK[i,:-2])/(ks[2:]-ks[:-2]) + WG[i,:-2]*WK[i,:-2])[0]
		ax.errorbar(ks, ell**(1.0/2.0)*ks**2*WG[i,:]*WK[i,:], color=c, ls='-', lw=1.3)
		ax.axvline(x=(ell+0.5)/Dmax, color=c, ls='--', lw=1.0)
	ax.errorbar(kin, Plin.P(0.0, kin)*np.amax(ells[0]**(1.0/2.0)*ks**2*WG[0,:]*WK[0,:])/np.amax(Plin.P(0.0, kin)), color='k', ls=':', lw=1.3)
	ax.set_xscale('log')
	ax.set_xlim(1.0e-4, 3.99)
	ax.set_xlabel(r"$k\quad[h_{70}\,\mathrm{Mpc^{-1}}]$", fontsize=18)
	ax.set_ylabel(r"$\ell^{1/2}\,\times\,k^{2}\,\Delta^{g}_{\ell}\,\,\Delta^{\kappa}_{\ell}\quad[h_{70}^{2}\,\mathrm{Mpc^{-2}}]$", fontsize=18)
	ax.tick_params('both',length=10, which='major', width=1.3, direction='in')
	ax.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for axis in ['top','left','bottom','right']:
		ax.spines[axis].set_linewidth(1.3)
	for tick in ax.xaxis.get_ticklabels():
		tick.set_fontsize(16)
	for tick in ax.yaxis.get_ticklabels():
		tick.set_fontsize(16)
	ax.yaxis.set_ticks_position('both')
	ax.yaxis.offsetText.set_fontsize(16)
	#plt.legend(fontsize=12, loc='upper right')

	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.13, 0.83, 0.82])
	for i, (ell, c) in enumerate(zip(ells, ['r', 'g', 'b'])):
		ax.errorbar(ks, ell**(4.0/2.0)*ks*WU[i,:]*WK[i,:]/LightSpeed, color=c, ls='-', lw=1.3)
		ax.axvline(x=(ell+0.5)/Dmax, color=c, ls='--', lw=1.0)
	ax.errorbar(kin, Plin.P(0.0, kin)*np.amax(ells[0]**(4.0/2.0)*ks*WU[0,:]*WK[0,:]/LightSpeed)/np.amax(Plin.P(0.0, kin)), color='k', ls=':', lw=1.3)
	ax.set_xscale('log')
	ax.set_xlim(1.0e-4, 3.99)
	ax.set_xlabel(r"$k\quad[h_{70}\,\mathrm{Mpc^{-1}}]$", fontsize=18)
	ax.set_ylabel(r"$\frac{\ell^{2}}{c}\,\times\,k^{2}\,\Delta^{u}_{\ell}\,\,\Delta^{\kappa}_{\ell}\quad[h_{70}^{2}\,\mathrm{Mpc}^{-2}]$", fontsize=18)
	ax.tick_params('both',length=10, which='major', width=1.3, direction='in')
	ax.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for axis in ['top','left','bottom','right']:
		ax.spines[axis].set_linewidth(1.3)
	for tick in ax.xaxis.get_ticklabels():
		tick.set_fontsize(16)
	for tick in ax.yaxis.get_ticklabels():
		tick.set_fontsize(16)
	ax.yaxis.set_ticks_position('both')
	ax.yaxis.offsetText.set_fontsize(16)
	#plt.legend(fontsize=12, loc='upper right')

	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.13, 0.83, 0.82])
	for i, (ell, c, label) in enumerate(zip(ells, ['r', 'g', 'b'], labels)):
		ax.errorbar(ks, ell**(6.0/2.0)*WU[i,:]*WU[i,:]/LightSpeed**2, color=c, ls='-', lw=1.3, label=label)
		ax.axvline(x=(ell+0.5)/Dmax, color=c, ls='--', lw=1.0)
	ax.errorbar(kin, Plin.P(0.0, kin)*np.amax(ells[0]**(6.0/2.0)*WU[0,:]*WU[0,:]/LightSpeed**2)/np.amax(Plin.P(0.0, kin)), color='k', ls=':', lw=1.3)
	ax.set_xscale('log')
	ax.set_xlim(1.0e-4, 3.99)
	ax.set_xlabel(r"$k\quad[h_{70}\,\mathrm{Mpc^{-1}}]$", fontsize=18)
	ax.set_ylabel(r"$\frac{\ell^{3}}{c^{2}}\,\times\,k^{2}\,\Delta^{u}_{\ell}\,\,\Delta^{u}_{\ell}\quad[h_{70}^{2}\,\mathrm{Mpc}^{-2}]$", fontsize=18)
	ax.tick_params('both',length=10, which='major', width=1.3, direction='in')
	ax.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for axis in ['top','left','bottom','right']:
		ax.spines[axis].set_linewidth(1.3)
	for tick in ax.xaxis.get_ticklabels():
		tick.set_fontsize(16)
	for tick in ax.yaxis.get_ticklabels():
		tick.set_fontsize(16)
	ax.yaxis.set_ticks_position('both')
	ax.yaxis.offsetText.set_fontsize(16)
	plt.legend(fontsize=16, loc='upper right')

	plt.show()

def plot_delta_limber():

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

	# Read in the DESI PV survey dn/dz
	skyarea, dndz = get_DESI(Dcspline=Dcspline)
	dndz["H"] = sp.interpolate.splev(dndz["dz"].to_numpy(), Hspline)
	dndz["f"] = sp.interpolate.splev(dndz["dz"].to_numpy(), fspline)
	dndz["D"] = sp.interpolate.splev(dndz["dz"].to_numpy(), Dspline)
	dndz["cz_hist"] = 2400.0 * dndz["H"][0] / dndz["H"]
	dndz["dudz"] = dndz["H"] * dndz["f"] * dndz["cz_hist"] / (1.0 + dndz["z"])
	dnspline = sp.interpolate.splrep(dndz["dz"].to_numpy(), dndz["cz_hist"].to_numpy() * dndz["H"].to_numpy()/LightSpeed/(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/np.sum(dndz["cz_hist"].to_numpy()))   # Normalised dn/dz * H(z)/c
	duspline = sp.interpolate.splrep(dndz["dz"].to_numpy(), dndz["dudz"].to_numpy() * dndz["H"].to_numpy()/LightSpeed/(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/np.sum(dndz["cz_hist"].to_numpy()))   # Normalised dn/dz * H^2(z) * f(z) / (1 + z) / c

	# Set up some k and ell binning
	ks = np.logspace(np.log10(1.0e-4), np.log10(4.0), 1000)
	ells = np.array([4, 45, 180])
	labels = [r"$\ell = 4$", r"$\ell = 45$", r"$\ell = 180$"]
	zmax = dndz["z"].to_numpy()[-1]+(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/2.0
	Dmin, Dmax, Dcmb = 0.0, sp.interpolate.splev(zmax, Dcspline), Dc[-1]

	WG = sp.interpolate.splev(np.outer(ells+0.5, 1.0/ks), dnspline, ext=1)
	WUlow = sp.interpolate.splev(np.outer(ells-0.5, 1.0/ks), duspline, ext=1)
	WUhigh = sp.interpolate.splev(np.outer(ells+0.5, 1.0/ks), duspline, ext=1)
	WU = (WUlow.T / np.sqrt(ells-0.5) - (ells+1.0)/(np.sqrt(ells+0.5)*(ells+0.5))*WUhigh.T).T
	WK = sp.interpolate.splev(np.outer(ells+0.5, 1.0/ks), cmbspline, ext=1)

	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.13, 0.83, 0.83])
	for i, (ell, c) in enumerate(zip(ells, ['r', 'g', 'b'])):
		ax.errorbar(ks, WG[i,:]*WK[i,:], color=c, ls='-', lw=1.3)
		ax.axvline(x=(ell+0.5)/Dmax, color=c, ls='--', lw=1.0)
	ax.set_xscale('log')
	ax.set_xlim(1.0e-4, 3.99)
	ax.set_xlabel(r"$k\quad[h_{70}\,\mathrm{Mpc^{-1}}]$", fontsize=14)
	ax.set_ylabel(r"$\ell^{1/2}\,\times\,k^{2}\,\Delta^{g}_{\ell}\,\,\Delta^{\kappa}_{\ell}\quad[h_{70}^{2}\,\mathrm{Mpc^{-2}}]$", fontsize=14)
	ax.tick_params('both',length=10, which='major', width=1.3, direction='in')
	ax.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for axis in ['top','left','bottom','right']:
		ax.spines[axis].set_linewidth(1.3)
	for tick in ax.xaxis.get_ticklabels():
		tick.set_fontsize(12)
	for tick in ax.yaxis.get_ticklabels():
		tick.set_fontsize(12)
	ax.yaxis.set_ticks_position('both')
	#plt.legend(fontsize=12, loc='upper right')

	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.13, 0.83, 0.83])
	for i, (ell, c) in enumerate(zip(ells, ['r', 'g', 'b'])):
		ax.errorbar(ks, ks*WU[i,:]*WK[i,:], color=c, ls='-', lw=1.3)
		ax.axvline(x=(ell+0.5)/Dmax, color=c, ls='--', lw=1.0)
	ax.set_xscale('log')
	ax.set_xlim(1.0e-4, 3.99)
	ax.set_xlabel(r"$k\quad[h_{70}\,\mathrm{Mpc^{-1}}]$", fontsize=14)
	ax.set_ylabel(r"$\frac{\ell^{2}}{c}\,\times\,k^{2}\,\Delta^{u}_{\ell}\,\,\Delta^{\kappa}_{\ell}\quad[h_{70}^{2}\,\mathrm{Mpc}^{-2}]$", fontsize=14)
	ax.tick_params('both',length=10, which='major', width=1.3, direction='in')
	ax.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for axis in ['top','left','bottom','right']:
		ax.spines[axis].set_linewidth(1.3)
	for tick in ax.xaxis.get_ticklabels():
		tick.set_fontsize(12)
	for tick in ax.yaxis.get_ticklabels():
		tick.set_fontsize(12)
	ax.yaxis.set_ticks_position('both')
	#plt.legend(fontsize=12, loc='upper right')

	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.13, 0.83, 0.83])
	for i, (ell, c, label) in enumerate(zip(ells, ['r', 'g', 'b'], labels)):
		ax.errorbar(ks, WU[i,:]*WU[i,:], color=c, ls='-', lw=1.3, label=label)
		ax.axvline(x=(ell+0.5)/Dmax, color=c, ls='--', lw=1.0)
	ax.set_xscale('log')
	ax.set_xlim(1.0e-4, 3.99)
	ax.set_xlabel(r"$k\quad[h_{70}\,\mathrm{Mpc^{-1}}]$", fontsize=14)
	ax.set_ylabel(r"$\frac{\ell^{3}}{c^{2}}\,\times\,k^{2}\,\Delta^{u}_{\ell}\,\,\Delta^{u}_{\ell}\quad[h_{70}^{2}\,\mathrm{Mpc}^{-2}]$", fontsize=14)
	ax.tick_params('both',length=10, which='major', width=1.3, direction='in')
	ax.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for axis in ['top','left','bottom','right']:
		ax.spines[axis].set_linewidth(1.3)
	for tick in ax.xaxis.get_ticklabels():
		tick.set_fontsize(12)
	for tick in ax.yaxis.get_ticklabels():
		tick.set_fontsize(12)
	ax.yaxis.set_ticks_position('both')
	plt.legend(fontsize=12, loc='upper right')

	plt.show()


def plot_Cell():

	# Plot C_uk for the 4 different surveys we consider
	colors = ['r', 'purple', 'g', 'b']
	labels = [r"$\mathrm{DESI}\,$",
			  r"$\mathrm{4HS}\,$",
			  r"$\mathrm{LSST}\,$",
			  r"$\mathrm{Recon}\,z \leq 0.15\,$"]
	files = ["./C_ell_DESI.txt", "./C_ell_4HS.txt", "./C_ell_LSSTJlt19.txt", "./C_ell_recon_2MRS.txt"]

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

	# Planck noise curves
	planck_mv = pd.read_csv("./nlkk_mv.dat", delim_whitespace=True, header=None)

	# SO noise curves
	SO_mv = pd.read_csv("./nlkk_v3_1_0_deproj0_SENS1_fsky0p4_it_lT30-3000_lP30-5000.dat", delim_whitespace=True, header=None)

	# S4 noise curves
	S4_realistic_mv = pd.read_csv("./nlkk_cmb_s4_realistic.dat", delim_whitespace=True, escapechar="#")
	S4_ideal_mv = pd.read_csv("./nlkk_cmb_s4_ideal.dat", delim_whitespace=True, escapechar="#")

	fig1, fig2, fig3, fig4 = plt.figure(), plt.figure(), plt.figure(), plt.figure()
	ax1, ax2, ax3, ax4 = fig1.add_axes([0.18, 0.13, 0.80, 0.82]), fig2.add_axes([0.16, 0.13, 0.82, 0.82]), fig3.add_axes([0.15, 0.13, 0.83, 0.82]), fig4.add_axes([0.15, 0.13, 0.82, 0.82])
	for i, (file, color, label, (skyarea, dndz), alpha) in enumerate(zip(files, colors, labels, [get_DESI(Dcspline=Dcspline), get_4HS(Dcspline=Dcspline), get_LSST(Dcspline=Dcspline), get_LSST(Dcspline=Dcspline)], [0.2, 0.2, 0.05, 0.0])):
		cell = np.loadtxt(file).T
		newell = np.linspace(3, 500, 498).astype(int)
		C_uu = 10.0**sp.interpolate.interp1d(np.log10(cell[0]), np.log10(cell[1]))(np.log10(newell))
		C_uk = 10.0**sp.interpolate.interp1d(np.log10(cell[0]), np.log10(cell[2]))(np.log10(newell))
		C_kk = 10.0**sp.interpolate.interp1d(np.log10(cell[0]), np.log10(cell[3]))(np.log10(newell))
		if np.shape(cell)[1] == 6:
			C_gg = sp.interpolate.interp1d(cell[0], cell[4])(newell)
			C_gk = sp.interpolate.interp1d(cell[0], cell[5])(newell)

		ax1.errorbar(newell, C_uu*newell**3/LightSpeed**2, color=color, ls='-', lw=1.3, label=label)
		#ax1.errorbar(newell, C_uu/C_uk, color=color, ls='-', lw=1.3, label=label)
		ax2.errorbar(newell, C_uk*newell**2/LightSpeed, color=color, ls='-', lw=1.3, label=label)
		if i == 0:
			ax3.errorbar(newell, C_kk, color='k', ls='-', lw=1.3, label=r"$\mathrm{Theory}$")

		planck_noise = sp.interpolate.splev(newell, sp.interpolate.splrep(planck_mv[0][:1001],planck_mv[1][:1001]))
		SO_noise = sp.interpolate.splev(newell, sp.interpolate.splrep(SO_mv[0][:1001],SO_mv[7][:1001]))
		S4_realistic_noise = sp.interpolate.splev(newell, sp.interpolate.splrep(S4_realistic_mv[" L"].to_numpy(),S4_realistic_mv["MV"].to_numpy()))
		S4_ideal_noise = sp.interpolate.splev(newell, sp.interpolate.splrep(S4_ideal_mv[" L"].to_numpy(),S4_ideal_mv["MV"].to_numpy()))

		dndz["H"] = sp.interpolate.splev(dndz["dz"].to_numpy(), Hspline)
		dndz["f"] = sp.interpolate.splev(dndz["dz"].to_numpy(), fspline)
		dndz["D"] = sp.interpolate.splev(dndz["dz"].to_numpy(), Dspline)
		dndz["dudz"] = dndz["H"] * dndz["f"] * dndz["cz_hist"] / (1.0 + dndz["z"])
		dnspline = sp.interpolate.splrep(dndz["dz"].to_numpy(), dndz["cz_hist"].to_numpy() * dndz["H"].to_numpy()/LightSpeed/(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/np.sum(dndz["cz_hist"].to_numpy()))   # Normalised dn/dz * H(z)/c
		duspline = sp.interpolate.splrep(dndz["dz"].to_numpy(), dndz["dudz"].to_numpy() * dndz["H"].to_numpy()/LightSpeed/(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/np.sum(dndz["cz_hist"].to_numpy()))   # Normalised dn/dz * H^2(z) * f(z) / (1 + z) / c

		if i == 3:
			skyarea = 4.0*np.pi
			N_uu = 3.0*250.0**2*4.0**3/sp.interpolate.splev(0.15, Dcspline)**3
			print(sp.interpolate.splev(0.15, Dcspline), sp.interpolate.splev(0.15, Dcspline)**3/(3.0 * 4.0**3))
		else:
			zmax = dndz["z"].to_numpy()[-1]+(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/2.0
			Dmin, Dmax, Dcmb = 0.0, sp.interpolate.splev(zmax, Dcspline), Dc[-1]
			v_std = sp.integrate.quad(Nuufunc, Dmin, Dmax, args=(dnspline, alpha), limit=1000, epsrel=1.0e-4)[0]**2
			N_uu = skyarea*v_std/np.sum(dndz["cz_hist"].to_numpy())
			print(np.sqrt(v_std), skyarea*np.sqrt(v_std)/np.sum(dndz["cz_hist"].to_numpy()))

		N_ell_uu = (C_uu + N_uu) * np.sqrt(4.0*np.pi/(skyarea * (2.0 * newell + 1)))
		N_ell_uk_planck = np.sqrt((C_uu + N_uu)*(C_kk + planck_noise) + C_uk**2) * np.sqrt(4.0*np.pi/(np.amin([skyarea,0.67*4.0*np.pi]) * (2.0 * newell + 1)))
		N_ell_uk_SO = np.sqrt((C_uu + N_uu)*(C_kk + SO_noise) + C_uk**2) * np.sqrt(4.0*np.pi/(np.amin([skyarea,0.4*4.0*np.pi]) * (2.0 * newell + 1)))
		N_ell_uk_S4_realistic = np.sqrt((C_uu + N_uu)*(C_kk + S4_realistic_noise) + C_uk**2) * np.sqrt(4.0*np.pi/(np.amin([skyarea,0.67*4.0*np.pi]) * (2.0 * newell + 1)))
		N_ell_uk_S4_ideal = np.sqrt((C_uu + N_uu)*(C_kk + S4_ideal_noise) + C_uk**2) * np.sqrt(4.0*np.pi/(np.amin([skyarea,0.67*4.0*np.pi]) * (2.0 * newell + 1)))
		
		N_ell_uk_cv = np.sqrt(C_uu*C_kk + C_uk**2) * np.sqrt(4.0*np.pi/(np.amin([skyarea,0.67*4.0*np.pi]) * (2.0 * newell + 1)))

		ax1.errorbar(newell, np.ones(len(newell))*N_uu*newell**3/LightSpeed**2, color=color, ls=':', lw=1.3)
		if i == 0:
			N_ell_kk_planck = (C_kk + planck_noise) * np.sqrt(4.0*np.pi/(2.67*np.pi * (2.0 * newell + 1)))
			N_ell_kk_S4_realistic = (C_kk + S4_realistic_noise) * np.sqrt(4.0*np.pi/(2.67*np.pi * (2.0 * newell + 1)))
			ax3.errorbar(newell, planck_noise, color='r', ls=':', lw=1.3, label=r"$\mathrm{Planck\,Noise}$")
			ax3.errorbar(newell, SO_noise, color='b', ls=':', lw=1.3, label=r"$\mathrm{SO\,Noise}$")
			ax3.errorbar(newell, S4_realistic_noise, color='g', ls=':', lw=1.3, label=r"$\mathrm{CMB-S4\,Noise}$")
			#ax3.errorbar(newell, S4_ideal_noise, color='purple', ls='--', lw=1.3, label=r"$\mathrm{CMB-S4\,(Ideal)\,Noise}$")

		if i == 0 or i == 3:
			ax4.errorbar(newell, np.cumsum(C_uk/N_ell_uk_planck), color=color, ls=':', lw=1.3, label=label+r"$ + \mathrm{Planck}$")
			ax4.errorbar(newell, np.cumsum(C_uk/N_ell_uk_cv), color=color, ls='-', lw=1.3)
			ax4.errorbar(newell, np.cumsum(C_uk/N_ell_uk_S4_realistic), color=color, ls='--', lw=1.3, label=label+r"$ + \mathrm{CMB-S4}$")

		lmin = 8
		print(newell[lmin-3])
		print(np.cumsum(C_uk[lmin-3:198]/N_ell_uk_planck[lmin-3:198])[-1])
		print(np.cumsum(C_uk[lmin-3:198]/N_ell_uk_SO[lmin-3:198])[-1])
		print(np.cumsum(C_uk[lmin-3:198]/N_ell_uk_S4_realistic[lmin-3:198])[-1])
		print(np.cumsum(C_uk[lmin-3:198]/N_ell_uk_S4_ideal[lmin-3:198])[-1])

	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.set_ylim(5.0e-11, 1.0e-5)
	ax1.set_xlabel(r"$\ell$", fontsize=18)
	ax1.set_ylabel(r"$\frac{\ell^{3}}{c^{2}}\,(C_{\ell}^{u u} \,\,\, || \,\,\, N_{\ell}^{u u})$", fontsize=18)
	#ax1.set_ylabel(r"$C_{\ell}^{u u} / C_{\ell}^{u k}$", fontsize=14)
	ax1.tick_params('both',length=10, which='major', width=1.3, direction='in')
	ax1.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for axis in ['top','left','bottom','right']:
		ax1.spines[axis].set_linewidth(1.3)
	for tick in ax1.xaxis.get_ticklabels():
		tick.set_fontsize(16)
	for tick in ax1.yaxis.get_ticklabels():
		tick.set_fontsize(16)
	ax1.yaxis.set_ticks_position('both')
	ax1.legend(fontsize=18, loc='lower left', bbox_to_anchor=(0.03, 0.03))

	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.set_ylim(5.0e-10, 5.0e-7)
	ax2.set_xlabel(r"$\ell$", fontsize=18)
	ax2.set_ylabel(r"$\frac{\ell^{2}}{c}\,C_{\ell}^{u k} \quad$", fontsize=18)
	ax2.tick_params('both',length=10, which='major', width=1.3, direction='in')
	ax2.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for axis in ['top','left','bottom','right']:
		ax2.spines[axis].set_linewidth(1.3)
	for tick in ax2.xaxis.get_ticklabels():
		tick.set_fontsize(16)
	for tick in ax2.yaxis.get_ticklabels():
		tick.set_fontsize(16)
	ax2.yaxis.set_ticks_position('both')
	ax2.legend(fontsize=18, loc='lower left', bbox_to_anchor=(0.03, 0.03))

	ax3.set_xscale('log')
	ax3.set_yscale('log')
	ax3.set_ylim(1.0e-11, 1.0e-6)
	ax3.set_xlabel(r"$\ell$", fontsize=18)
	ax3.set_ylabel(r"$C_{\ell}^{\kappa \kappa} \,\,\, || \,\,\, N_{\ell}^{\kappa \kappa}$", fontsize=18, labelpad=-5)
	ax3.tick_params('both',length=10, which='major', width=1.3, direction='in')
	ax3.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for axis in ['top','left','bottom','right']:
		ax3.spines[axis].set_linewidth(1.3)
	for tick in ax3.xaxis.get_ticklabels():
		tick.set_fontsize(16)
	for tick in ax3.yaxis.get_ticklabels():
		tick.set_fontsize(16)
	ax3.yaxis.set_ticks_position('both')
	ax3.legend(fontsize=18, loc='lower left', bbox_to_anchor=(0.03, 0.03))

	#ax4.set_yscale('log')
	ax4.set_xlim(0.0, 100.0)
	ax4.set_ylim(0.1, 50.0)
	ax4.set_xlabel(r"$\ell$", fontsize=14)
	ax4.set_ylabel(r"$\mathrm{Cumulative\,S/N}$", fontsize=14)
	#ax4.set_ylabel(r"$\mathrm{S/N}$", fontsize=14)
	ax4.tick_params('both',length=10, which='major', width=1.3, direction='in')
	ax4.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for axis in ['top','left','bottom','right']:
		ax4.spines[axis].set_linewidth(1.3)
	for tick in ax4.xaxis.get_ticklabels():
		tick.set_fontsize(12)
	for tick in ax4.yaxis.get_ticklabels():
		tick.set_fontsize(12)
	ax4.yaxis.set_ticks_position('both')
	#ax4.legend(fontsize=12, loc='lower right', bbox_to_anchor=(0.98, 0.03))
	ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

	plt1 = ax4.errorbar([], [], color=colors[0], label=labels[0], ls='-')
	plt2 = ax4.errorbar([], [], color=colors[1], label=labels[1], ls='-')
	plt3 = ax4.errorbar([], [], color=colors[2], label=labels[2], ls='-')
	plt4 = ax4.errorbar([], [], color=colors[3], label=labels[3], ls='-')
	#leg = ax4.legend(handles=[plt1, plt4], loc='lower right', bbox_to_anchor=(0.98, 0.03), fontsize=12, bbox_transform=ax4.transAxes)
	leg = ax4.legend(handles=[plt1, plt4], loc='upper left', bbox_to_anchor=(0.03, 0.75), fontsize=12, bbox_transform=ax4.transAxes)
	ax4.add_artist(leg)

	plt1 = ax4.errorbar([], [], color='k', label=r"$\mathrm{Cosmic\,Variance}$", ls='-')
	plt2 = ax4.errorbar([], [], color='k', label=r"$\mathrm{Planck}$", ls=':')
	plt3 = ax4.errorbar([], [], color='k', label=r"$\mathrm{CMB-S4}$", ls='--')
	ax4.legend(handles=[plt1, plt2, plt3], loc='upper left', bbox_to_anchor=(0.03, 0.98), fontsize=12, bbox_transform=ax4.transAxes)
	#ax4.legend(handles=[plt1, plt2, plt3], loc='lower left', bbox_to_anchor=(0.03, 0.03), fontsize=12, bbox_transform=ax4.transAxes)

	#ax5.set_yscale('log')
	#ax5.set_xlim(0.0, 100.0)
	#ax5.set_ylim(0.1, 500.0)
	#ax5.set_xlabel(r"$\ell$", fontsize=14)
	#ax5.set_ylabel(r"$\mathrm{Cumulative\,S/N}$", fontsize=14)
	#ax5.axvline(x = 8.0, color='k', ls=':', lw=1.0)
	#ax5.axvline(x = 200.0, color='k', ls=':', lw=1.0)
	#ax5.tick_params('both',length=10, which='major', width=1.3, direction='in')
	#ax5.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	#for axis in ['top','left','bottom','right']:
	#	ax5.spines[axis].set_linewidth(1.3)
	#for tick in ax5.xaxis.get_ticklabels():
	#	tick.set_fontsize(12)
	#for tick in ax5.yaxis.get_ticklabels():
	#	tick.set_fontsize(12)
	#ax5.yaxis.set_ticks_position('both')
	#ax5.legend(fontsize=12, loc='lower right', bbox_to_anchor=(0.98, 0.03))
	#ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))


	plt.show()


	#ax.set_xlim(0.0, 0.70)
	#ax.set_ylim(0.0, 1.05)
	#ax.set_xlabel(r"$z$", fontsize=14)
	#ax.set_ylabel(r"$W(z)$", fontsize=14)
	#ax.tick_params('both',length=10, which='major', width=1.3, direction='in')
	#ax.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	#for axis in ['top','left','bottom','right']:
	#	ax.spines[axis].set_linewidth(1.3)
	#for tick in ax.xaxis.get_ticklabels():
	#	tick.set_fontsize(12)
	#for tick in ax.yaxis.get_ticklabels():
	#	tick.set_fontsize(12)
	#ax.yaxis.set_ticks_position('both')
	#plt.legend(fontsize=12, loc='upper right')
	#plt.show()



def plot_SN_vs_sigma_rec():

	lmin = 3
	newell = np.linspace(3, 500, 498).astype(int)
	print(newell[lmin-3])

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

	# Planck noise curves
	planck_mv = pd.read_csv("./nlkk_mv.dat", delim_whitespace=True, header=None)
	planck_noise = sp.interpolate.splev(newell, sp.interpolate.splrep(planck_mv[0][:1001],planck_mv[1][:1001]))

	# SO noise curves
	SO_mv = pd.read_csv("./nlkk_v3_1_0_deproj0_SENS1_fsky0p4_it_lT30-3000_lP30-5000.dat", delim_whitespace=True, header=None)
	SO_noise = sp.interpolate.splev(newell, sp.interpolate.splrep(SO_mv[0][:1001],SO_mv[7][:1001]))

	# S4 noise curves
	S4_realistic_mv = pd.read_csv("./nlkk_cmb_s4_realistic.dat", delim_whitespace=True, escapechar="#")
	S4_ideal_mv = pd.read_csv("./nlkk_cmb_s4_ideal.dat", delim_whitespace=True, escapechar="#")

	S4_realistic_noise = sp.interpolate.splev(newell, sp.interpolate.splrep(S4_realistic_mv[" L"].to_numpy(),S4_realistic_mv["MV"].to_numpy()))
	S4_ideal_noise = sp.interpolate.splev(newell, sp.interpolate.splrep(S4_ideal_mv[" L"].to_numpy(),S4_ideal_mv["MV"].to_numpy()))

	# Arrays of sigma_recs and alphas
	sigma_rec = np.logspace(np.log10(10.0), np.log10(10000.0), 100)
	alphas = np.logspace(np.log10(0.01), np.log10(0.25), 100)

	# Recon C_ells
	skyarea = 4.0*np.pi
	cell = np.loadtxt("./C_ell_recon_z0p15.txt").T
	C_uu = 10.0**sp.interpolate.interp1d(np.log10(cell[0]), np.log10(cell[1]))(np.log10(newell))
	C_uk = 10.0**sp.interpolate.interp1d(np.log10(cell[0]), np.log10(cell[2]))(np.log10(newell))
	C_kk = 10.0**sp.interpolate.interp1d(np.log10(cell[0]), np.log10(cell[3]))(np.log10(newell))
	if np.shape(cell)[1] == 6:
		C_gg = sp.interpolate.interp1d(cell[0], cell[4])(newell)
		C_gk = sp.interpolate.interp1d(cell[0], cell[5])(newell)



	SN_sigma_rec, SN_alphas = np.zeros((4, 100)), np.zeros((4, 100))
	for i, s in enumerate(sigma_rec):

		N_uu = 3.0*s**2*4.0**3/sp.interpolate.splev(0.15, Dcspline)**3
		N_ell_uu = (C_uu + N_uu) * np.sqrt(4.0*np.pi/(skyarea * (2.0 * newell + 1)))
		N_ell_uk_planck = np.sqrt((C_uu + N_uu)*(C_kk + planck_noise) + C_uk**2) * np.sqrt(4.0*np.pi/(np.amin([skyarea,0.67*4.0*np.pi]) * (2.0 * newell + 1)))
		N_ell_uk_SO = np.sqrt((C_uu + N_uu)*(C_kk + SO_noise) + C_uk**2) * np.sqrt(4.0*np.pi/(np.amin([skyarea,0.4*4.0*np.pi]) * (2.0 * newell + 1)))
		N_ell_uk_S4_realistic = np.sqrt((C_uu + N_uu)*(C_kk + S4_realistic_noise) + C_uk**2) * np.sqrt(4.0*np.pi/(np.amin([skyarea,0.67*4.0*np.pi]) * (2.0 * newell + 1)))
		N_ell_uk_S4_ideal = np.sqrt((C_uu + N_uu)*(C_kk + S4_ideal_noise) + C_uk**2) * np.sqrt(4.0*np.pi/(np.amin([skyarea,0.67*4.0*np.pi]) * (2.0 * newell + 1)))
		
		SN_sigma_rec[0, i] = np.cumsum(C_uk[lmin-3:198]/N_ell_uk_planck[lmin-3:198])[-1]
		SN_sigma_rec[1, i] = np.cumsum(C_uk[lmin-3:198]/N_ell_uk_SO[lmin-3:198])[-1]
		SN_sigma_rec[2, i] = np.cumsum(C_uk[lmin-3:198]/N_ell_uk_S4_realistic[lmin-3:198])[-1]
		SN_sigma_rec[3, i] = np.cumsum(C_uk[lmin-3:198]/N_ell_uk_S4_ideal[lmin-3:198])[-1]

		print(s, N_uu)

	fig1 = plt.figure(figsize=[6.4, 5.4])
	ax1 = fig1.add_axes([0.11, 0.13, 0.82, 0.76])
	ax1.errorbar(sigma_rec, SN_sigma_rec[0], color='b', ls=':', lw=1.3, label=r"$\mathrm{Recon}\,z \leq 0.15\,$" + r"$ + \mathrm{Planck}$")
	ax1.errorbar(sigma_rec, SN_sigma_rec[2], color='b', ls='--', lw=1.3, label=r"$\mathrm{Recon}\,z \leq 0.15\,$"+r"$ + \mathrm{CMB-S4}$")
	ax1.axvline(x=250.0, color='k', ls='-', lw=0.8)
	ax1.set_xscale('log')
	ax1.set_xlim(10.0, 10000.0)
	ax1.set_ylim(0.1, 50.0)
	ax1.set_xlabel(r"$\sigma_{\mathrm{rec}}\,(\mathrm{km\,s^{-1}})$", fontsize=14)
	ax1.set_ylabel(r"$\mathrm{Cumulative\,S/N}$", fontsize=14)
	ax1.tick_params('both',length=10, which='major', width=1.3, direction='in')
	ax1.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for axis in ['top','left','bottom','right']:
		ax1.spines[axis].set_linewidth(1.3)
	for tick in ax1.xaxis.get_ticklabels():
		tick.set_fontsize(12)
	for tick in ax1.yaxis.get_ticklabels():
		tick.set_fontsize(12)
	ax1.yaxis.set_ticks_position('both')
	ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
	ax1.legend(fontsize=14, loc='upper left')

	def N_uu_forward(x):
		return 3.0*x**2*4.0**3/sp.interpolate.splev(0.15, Dcspline)**3

	def N_uu_backward(x):
		return np.sqrt(x*sp.interpolate.splev(0.15, Dcspline)**3/(3.0*4.0**3))

	secax = ax1.secondary_xaxis('top', functions=(N_uu_forward, N_uu_backward))
	secax.set_xlabel(r"$N_{uu}\,(\mathrm{km^{2}\,s^{-2}})$", fontsize=14, labelpad=8)
	secax.tick_params('both',length=10, which='major', width=1.3, direction='in')
	secax.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for tick in secax.xaxis.get_ticklabels():
		tick.set_fontsize(12)
	secax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))


	# DESI C_ells
	skyarea, dndz = get_DESI(Dcspline=Dcspline)
	cell = np.loadtxt("./C_ell_recon_z0p15.txt").T
	newell = np.linspace(3, 500, 498).astype(int)
	C_uu = 10.0**sp.interpolate.interp1d(np.log10(cell[0]), np.log10(cell[1]))(np.log10(newell))
	C_uk = 10.0**sp.interpolate.interp1d(np.log10(cell[0]), np.log10(cell[2]))(np.log10(newell))
	C_kk = 10.0**sp.interpolate.interp1d(np.log10(cell[0]), np.log10(cell[3]))(np.log10(newell))
	if np.shape(cell)[1] == 6:
		C_gg = sp.interpolate.interp1d(cell[0], cell[4])(newell)
		C_gk = sp.interpolate.interp1d(cell[0], cell[5])(newell)

	dndz["H"] = sp.interpolate.splev(dndz["dz"].to_numpy(), Hspline)
	dndz["f"] = sp.interpolate.splev(dndz["dz"].to_numpy(), fspline)
	dndz["D"] = sp.interpolate.splev(dndz["dz"].to_numpy(), Dspline)
	dndz["dudz"] = dndz["H"] * dndz["f"] * dndz["cz_hist"] / (1.0 + dndz["z"])
	dnspline = sp.interpolate.splrep(dndz["dz"].to_numpy(), dndz["cz_hist"].to_numpy() * dndz["H"].to_numpy()/LightSpeed/(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/np.sum(dndz["cz_hist"].to_numpy()))   # Normalised dn/dz * H(z)/c
	duspline = sp.interpolate.splrep(dndz["dz"].to_numpy(), dndz["dudz"].to_numpy() * dndz["H"].to_numpy()/LightSpeed/(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/np.sum(dndz["cz_hist"].to_numpy()))   # Normalised dn/dz * H^2(z) * f(z) / (1 + z) / c

	v_std = np.empty(len(alphas))
	zmax = dndz["z"].to_numpy()[-1]+(dndz["z"].to_numpy()[1] - dndz["z"].to_numpy()[0])/2.0
	Dmin, Dmax, Dcmb = 0.0, sp.interpolate.splev(zmax, Dcspline), Dc[-1]
	for i, alpha in enumerate(alphas):

		v_std[i] = sp.integrate.quad(Nuufunc, Dmin, Dmax, args=(dnspline, alpha), limit=1000, epsrel=1.0e-4)[0]**2
		N_uu = skyarea*v_std[i]/np.sum(dndz["cz_hist"].to_numpy())

		N_ell_uu = (C_uu + N_uu) * np.sqrt(4.0*np.pi/(skyarea * (2.0 * newell + 1)))
		N_ell_uk_planck = np.sqrt((C_uu + N_uu)*(C_kk + planck_noise) + C_uk**2) * np.sqrt(4.0*np.pi/(np.amin([skyarea,0.67*4.0*np.pi]) * (2.0 * newell + 1)))
		N_ell_uk_SO = np.sqrt((C_uu + N_uu)*(C_kk + SO_noise) + C_uk**2) * np.sqrt(4.0*np.pi/(np.amin([skyarea,0.4*4.0*np.pi]) * (2.0 * newell + 1)))
		N_ell_uk_S4_realistic = np.sqrt((C_uu + N_uu)*(C_kk + S4_realistic_noise) + C_uk**2) * np.sqrt(4.0*np.pi/(np.amin([skyarea,0.67*4.0*np.pi]) * (2.0 * newell + 1)))
		N_ell_uk_S4_ideal = np.sqrt((C_uu + N_uu)*(C_kk + S4_ideal_noise) + C_uk**2) * np.sqrt(4.0*np.pi/(np.amin([skyarea,0.67*4.0*np.pi]) * (2.0 * newell + 1)))
		
		SN_alphas[0, i] = np.cumsum(C_uk[lmin-3:198]/N_ell_uk_planck[lmin-3:198])[-1]
		SN_alphas[1, i] = np.cumsum(C_uk[lmin-3:198]/N_ell_uk_SO[lmin-3:198])[-1]
		SN_alphas[2, i] = np.cumsum(C_uk[lmin-3:198]/N_ell_uk_S4_realistic[lmin-3:198])[-1]
		SN_alphas[3, i] = np.cumsum(C_uk[lmin-3:198]/N_ell_uk_S4_ideal[lmin-3:198])[-1]

		print(alpha, np.sqrt(v_std[i]), N_uu)

	v_std_spline = sp.interpolate.splrep(alphas, v_std)
	alpha_spline = sp.interpolate.splrep(v_std, alphas)

	fig2 = plt.figure(figsize=[6.4, 5.4])
	ax1 = fig2.add_axes([0.11, 0.13, 0.82, 0.76])
	ax1.errorbar(alphas, SN_alphas[0], color='r', ls=':', lw=1.3, label=r"$\mathrm{DESI}$" + r"$ + \mathrm{Planck}$")
	ax1.errorbar(alphas, SN_alphas[2], color='r', ls='--', lw=1.3, label=r"$\mathrm{DESI}$"+r"$ + \mathrm{CMB-S4}$")
	ax1.axvline(x=0.05, color='k', ls='-', lw=0.8)
	ax1.axvline(x=0.20, color='k', ls='-', lw=0.8)
	ax1.set_xscale('log')
	ax1.set_xlim(0.01, 0.25)
	ax1.set_ylim(0.1, 18.0)
	ax1.set_xlabel(r"$\alpha$", fontsize=14)
	ax1.set_ylabel(r"$\mathrm{Cumulative\,S/N}$", fontsize=14)
	ax1.tick_params('both',length=10, which='major', width=1.3, direction='in')
	ax1.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for axis in ['top','left','bottom','right']:
		ax1.spines[axis].set_linewidth(1.3)
	for tick in ax1.xaxis.get_ticklabels():
		tick.set_fontsize(12)
	for tick in ax1.yaxis.get_ticklabels():
		tick.set_fontsize(12)
	ax1.yaxis.set_ticks_position('both')
	ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
	ax1.legend(fontsize=14, loc='upper left')

	def N_uu_forward(x):
		v_std = sp.interpolate.splev(x, v_std_spline)
		return skyarea*v_std/np.sum(dndz["cz_hist"].to_numpy())

	def N_uu_backward(x):
		v_std = x * np.sum(dndz["cz_hist"].to_numpy())/skyarea
		return sp.interpolate.splev(v_std, alpha_spline)

	secax = ax1.secondary_xaxis('top', functions=(N_uu_forward, N_uu_backward))
	secax.set_xlabel(r"$N_{uu}\,(\mathrm{km^{2}\,s^{-2}})$", fontsize=14, labelpad=8)
	secax.tick_params('both',length=10, which='major', width=1.3, direction='in')
	secax.tick_params('both',length=5, which='minor', width=1.3, direction='in')
	for tick in secax.xaxis.get_ticklabels():
		tick.set_fontsize(12)
	secax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

	plt.show()


if __name__ == "__main__":

	#plot_dndz()
	#plot_delta()
	#plot_delta_limber()
	#plot_Cell()
	plot_SN_vs_sigma_rec()

