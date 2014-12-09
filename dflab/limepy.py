# -*- coding: utf-8 -*-
from __future__ import division, absolute_import

import numpy
from numpy import exp, sqrt, pi, sin
from scipy.interpolate import PiecewisePolynomial
from scipy.special import gamma, gammainc, dawsn
from scipy.integrate import ode
from math import factorial

#     Authors: Mark Gieles, Alice Zocchi (Surrey 2014)

class limepy:
    def __init__(self, W0, n,**kwargs):
        r"""

        *(MM, A)limepy*

        *(Multi-Mass, Anisotropic) Lowered Isothermal Model Explorer in Python*

        Class to generate solutions to lowered isothermal models.

        **Parameters:**

        W0 : scalar
           Central dimensionless potential
        n : int
          Order of truncation [1=Woolley, 2=King, 3=Wilson]; default=2
        mj : list, required for multi-mass system
          Mean mass of each component; default=None
        Mj : list, required for multi-mass system
           Total mass of each component; default=None
        delta : scalar, optional
              Index in sig_j = v_0*mu_j**-delta; default=0.5
        eta : scalar, optional
              Index in ra_j = ra*mu_j**eta; default=0.5
        MS : scalar, optional
           Final scaled mass; default=10^5 [Msun]
        RS : scalar, optional
           Final scaled mass; default=3 [pc]
        GS : scalar, optional
           Final scaled mass; default=0.004302 [(km/s)^2 pc/Msun]
        scale_radius : str, optional
                     Radius to scale ['rv' or 'rh']; default='rh'
        scale : bool, optional
              Scale model to desired G=GS, M=MS, R=RS; default=False
        potonly : bool, optional
                Fast solution by solving potential only; default=False
        max_step : scalar, optional
                 Maximum step size for ode output; default=1e4
        verbose : bool, optional
                Print diagnostics; default=False

        **Examples:**

        Construct a Woolley model with :math:`W_0 = 7` and print
        :math:`r_{\rm t}/r_0` and :math:`r_{\rm v}/r_{\rm h}`

        >>> k = limepy(7, 1)
        >>> print k.rt/k.r0, k.rv/k.rh

        Construct a Michie-King model and print :math:`r_{\rm
        a}/r_{\rm h}`

        >>> a = limepy(7, 2, ra=2)
        >>> print a.ra/a.rh

        Create a Wilson model with :math:`W_0 = 12` in Henon/N-body
        units: :math:`G=M=r_{\rm v}=1` and print the normalisation
        constant :math:`A` of the DF and the DF in the centre:

        >>> w = limepy(12, 3, scale=True, GS=1, MS=1, RS=1, scale_radius='rv')
        >>> print w.A, w.df(0,0,0)

        Multi-mass King model in physical units with :math:`r_{\rm h}
        = 1\,{\rm pc}` and :math:`M = 10^5\,{\rm M_{\odot}}`

        >>> m = limepy(7, 2, mj=[0.3,1,5], Mj=[9,3,1], scale=True, MS=1e5, RS=1)

        
        **Description of the distribution function:**

        The isotropic distribution functions are defined as

        .. math::
            f_n(E) = \displaystyle \begin{cases}
            A\exp(-E), &n=1 \\
            \displaystyle A\left[\exp(-E) - \sum_{m=0}^{n-2} \frac{(-E)^m}{m!} \right], &n>1
            \end{cases}

        where :math:`\displaystyle E = \frac{v^2/2 - \phi +
        \phi(r_{\rm t})}{\sigma^2}` and :math:`\sigma` is a velocity
        scale and :math:`0 < \phi-\phi(r_{\rm t}) <W_0/\sigma^2`

         *  n = 1 : `Woolley (1954) <http://adsabs.harvard.edu/abs/1954MNRAS.114..191W>`_
         *  n = 2 : `King (1966) <http://adsabs.harvard.edu/abs/1966AJ.....71...64K>`_
         *  n = 3 : `Wilson (1975) <http://adsabs.harvard.edu/abs/1975AJ.....80..175W>`_

        Radial anisotropy a la `Michie (1963)
        <http://adsabs.harvard.edu/abs/1963MNRAS.125..127M>`_ can be
        included

        .. math::
            f_n(E, J^2) = \exp(-J^2)f_n(E),

        where :math:`J^2 = (rv_t)^2/(2r_{\rm a}^2\sigma^2)`, here
        :math:`r_{\rm a}` is the anisotropy radius

        Multi-mass models are found by summing the DFs of individual
        mass components and adopting for each component following
        `Gunn & Griffin (1979)
        <http://adsabs.harvard.edu/abs/1979AJ.....84..752G>`_

        .. math::
             \sigma_j       &\propto  \mu_j^{-\delta}\\
             r_{{\rm a},j}  &\propto  \mu_j^{\eta}

        where :math:`\mu_j = m_j/\bar{m}` and :math:`\bar{m}` is the
        central density weighted mean mass.

        """

        self._set_kwargs(W0, n, **kwargs)

        if (self.multi):
            self._init_multi(self.mj, self.Mj)
            while self.diff > self.diffcrit:
                self._poisson(True)
                self._set_alpha()
                if self.niter > 100:
                    self.converged=False

        self.r0 = 3./sqrt(self._rho(self.W0, 0, self.ramax))

        if (self.multi): self.r0j = sqrt(self.sig2)*self.r0

        self._poisson(self.potonly)
        if (self.multi): self.Mj = self._Mjtot
        if (self.scale): self._scale()

        if (self.verbose):
            print "\n Model properties: "
            print " ----------------- "
            print " W0 = %5.2f; n = %1i"%(self.W0, self.n)
            if (self.potonly):
                print " M = %10.3f; U = %10.4f "%(self.M, self.U)
            else:
                out1=(self.M,self.U,self.K,self.K/self.U)
                print " M = %10.3e; U = %10.4e; K = %10.4e; Q = %6.4f"%out1
            out2=(self.rv/self.rh,self.rh/self.r0,self.rt/self.r0)
            print " rv/rh = %4.3f; rh/r0 = %6.3f; rt/r0 = %7.3f"%out2

    def _set_kwargs(self, W0, n, **kwargs):
        self.W0, self.n = W0, n

        self.MS, self.RS, self.GS = 1e5, 3, 0.004302
        self.scale_radius = 'rh'
        self.scale=False
        self.max_step = 1e4
        self.diffcrit = 1e-10
        self.xcrit = 3e-4*10**self.n # criterion to switch to dawsn approx
        self.phicrit = 3e-2/10**self.n # criterion to switch to v2 approx

        self.nmbin, self.delta, self.eta=1,0.5,0.5

        self.G = 1/(4*pi)
        self.mu, self._ah = numpy.array([1.0]), numpy.array([1.0])
        self.sig2 = numpy.array([1.0])
        self.niter = 0

        self.potonly, self.multi, self.verbose = [False]*3
        self.ra, self.ramax = 1e6, 100

        self.converged=False
        self._interpolator_set=False

        if kwargs is not None:
            for key, value in kwargs.iteritems():
                setattr(self, key, value)
            if 'mj' in kwargs and 'Mj' in kwargs:
                self.multi=True
                if len(self.Mj) is not len(self.mj):
                    raise ValueError("Error: Mj and mj must have same length")
            if ('mj' not in kwargs and 'Mj' in kwargs) or \
               ('Mj' not in kwargs and 'mj' in kwargs):
                raise ValueError("Error: Supply both mj and Mj")
        self.raj = [self.ra]

        return

    def _logcheck(self, t, y):
        """ Logs steps and checks for final values """
        if (t>0):
            self.r, self.y = numpy.r_[self.r, t], numpy.c_[self.y, y]
        return 0 if (y[0]>1e-6) else -1

    def _set_mass_function_variables(self):
        self.mmean = sum(self.mj*self.alpha)
        self.mu = self.mj/self.mmean
        self.sig2 = self.mu**(-2*self.delta)
        self.raj = self.ra*self.mu**self.eta

        self.W0j = self.W0/self.sig2
        self._ah = self.alpha*self._rho(self.W0,0,self.ramax)
        self._ah /= self._rho(self.W0j,0,self.ramax)

    def _init_multi(self, mj, Mj):
        """ Initialise parameters and arrays for multi-mass system"""
        self.multi=True
        self.mj = numpy.array(mj)
        self.Mj = numpy.array(Mj)
        self.nmbin = len(mj)

        # Set trial value for alpha_j array, will be updated in iterations
        self.alpha = self.Mj/sum(self.Mj)
        self.alpha/=sum(self.alpha)
        self._set_mass_function_variables()
        self.diff = 1

    def _set_alpha(self):
        """ Set central rho_j for next iteration """
        self.alpha *= self.Mj/self._Mjtot
        self.alpha/=sum(self.alpha)

        self._set_mass_function_variables()
        self.diff = sum((self._Mjtot/sum(self._Mjtot) -
                         self.Mj/sum(self.Mj))**2)/len(self._Mjtot)
        self.niter+=1
        if (self.verbose):
            Mjin,Mjit="", ""
            for j in range(self.nmbin):
                Mjin=Mjin+"%12.8f "%(self.Mj[j]/sum(self.Mj))
                Mjit=Mjit+"%12.8f "%(self._Mjtot[j]/sum(self._Mjtot))
            out=(self.niter, self.diff, self.converged, Mjin, Mjit)

            print " Iter %3i; diff = %10.3e; conv = %s; Mj=%s; Mjtot=%s"%out

    def _poisson(self, potonly):
        """ Solves Poisson equation """
        # y = [phi, u_j, U, K_j], where u = -M(<r)/G

        # Initialize
        self.r = numpy.array([0])
        self.y = numpy.r_[self.W0, numpy.zeros(self.nmbin+1)]
        if (not potonly): self.y = numpy.r_[self.y, numpy.zeros(self.nmbin)]

        # Ode solving
        max_step = 1e4 if (potonly) else self.max_step
        sol = ode(self._odes)
        sol.set_integrator('dopri5',nsteps=1e6,max_step=max_step,atol=1e-8)
        sol.set_solout(self._logcheck)
        sol.set_f_params(potonly)
        sol.set_initial_value(self.y,0)
        sol.integrate(1e4)

        # Extrapolate to r_t: phi(r) =~ a(r_t -r), a = GM/r_t^2
        p = 2*sol.y[0]*self.r[-1]/(self.G*-sol.y[1]/self.G)
        if (p<0.5):
            rtfac = (1 - sqrt(1-2*p))/p
            self.rt = rtfac*self.r[-1] if (rtfac > 1) else self.r[-1]
        else:
            self.rt = self.r[-1]

        if (self.rt < 1e4):
            self.converged=True
        else:
            self.converged=False

        # Fill arrays needed if potonly=True
        self.r = numpy.r_[self.r, self.rt]
        self.phi = numpy.r_[self.y[0,:], 0]
        self._Mjtot = -sol.y[1:1+self.nmbin]/self.G

        self.M = sum(self._Mjtot)
        self.dp1 = numpy.r_[0, self.y[1,1:]/self.r[1:-1]**2,
                            -self.G*self.M/self.rt**2]

        self.A = self._ah/(2*pi*self.sig2)**1.5

        if (not self.multi):
            self.mc = -numpy.r_[self.y[1,:],self.y[1,-1]]/self.G

        if (self.multi):
            self.mc = sum(-self.y[1:1+self.nmbin,:]/self.G)
            self.mc = numpy.r_[self.mc, self.mc[-1]]

        # Compute radii to be able to scale in case potonly=True
        self.U = self.y[1+self.nmbin,-1]  - 0.5*self.G*self.M**2/self.rt
        self.rh = numpy.interp(0.5*self.mc[-1],self.mc,self.r)
        self.rv = -0.5*self.G*self.M**2/self.U

        # Additional stuff
        if (not potonly):
            self.K = numpy.sum(sol.y[2+self.nmbin:2+2*self.nmbin])

            if (not self.multi):
                self.rho = self._rho(self.phi, self.r, self.ra)
                self.v2, self.v2r, self.v2t = \
                        self._get_v2(self.phi, self.r, self.rho, self.ra)

            if (self.multi):
                for j in range(self.nmbin):
                    phi, ra = self.phi/self.sig2[j], self.raj[j]
                    rhoj = self._rho(phi, self.r, ra)
                    v2j, v2rj, v2tj = self._get_v2(phi, self.r, rhoj, ra)
                    v2j, v2rj, v2tj = (q*self.sig2[j] for q in [v2j,v2rj,v2tj])
                    betaj = self._beta(self.r, v2rj, v2tj)

                    kj = self.y[2+self.nmbin,:]

                    mcj = -numpy.r_[self.y[1+j,:], self.y[1+j,-1]]/self.G
                    rhj = numpy.interp(0.5*mcj[-1], mcj, self.r)

                    if (j==0):
                        self.rhoj = rhoj
                        self.rho = self._ah[j]*rhoj
                        self.v2j, self.v2rj, self.v2tj = v2j, v2rj, v2tj
                        self.v2 = self._Mjtot[j]*v2j/self.M
                        self.v2r = self._Mjtot[j]*v2rj/self.M
                        self.v2t = self._Mjtot[j]*v2tj/self.M

                        self.betaj, self.kj, self.Kj = betaj, kj, kj[-1]
                        self.rhj, self.mcj = rhj, mcj
                    else:
                        self.rhoj = numpy.vstack((self.rhoj, self._ah[j]*rhoj))
                        self.rho += self._ah[j]*rhoj

                        self.v2j = numpy.vstack((self.v2j, v2j))
                        self.v2rj = numpy.vstack((self.v2rj, v2rj))
                        self.v2tj = numpy.vstack((self.v2tj, v2tj))
                        self.v2 += self._Mjtot[j]*v2j/self.M
                        self.v2r += self._Mjtot[j]*v2rj/self.M
                        self.v2t += self._Mjtot[j]*v2tj/self.M

                        self.betaj = numpy.vstack((self.betaj, betaj))
                        self.kj = numpy.vstack((self.kj, kj))
                        self.Kj = numpy.r_[self.Kj, kj[-1]]
                        self.rhj = numpy.r_[self.rhj,rhj]
                        self.mcj = numpy.vstack((self.mcj, mcj))
            self.beta = self._beta(self.r, self.v2r, self.v2t)

    def _dawsn_t(self, p, phi):
        """ Anisotropic term in rho and v2"""
        n = self.n
        x = p*sqrt(phi)
        if (x < self.xcrit):
            dawsn_t = p**2*phi**(n+0.5)/gamma(n+1.5)
            dawsn_t*=(1 - p**2*phi/(n+1.5) + p**4*phi**2/((n+1.5)*(n+2.5)))
        else:
            P  = (-1)**n * p**(1-2*n)/gamma(1.5)
            dawsn_series = sum((-1)**m * x**(2*m+1) * gamma(1.5)/gamma(m+1.5)
                               for m in range(n))
            dawsn_t = P*(dawsn(x) - dawsn_series)
        return dawsn_t

    def _rho(self, phi, r, ra):
        """ Wrapper for _rhofunc when either phi or r, or both, are arrays """
        if not hasattr(phi,"__len__"): phi = numpy.array([phi])
        if not hasattr(r,"__len__"): r = numpy.array([r])

        n = max([phi.size, r.size])
        rho = numpy.zeros(n)

        for i in range(n):
            if (phi.size==n)&(r.size==n):
                rho[i] = self._rhofunc(phi[i], r[i], ra)
            if (phi.size==n)&(r.size==1): rho[i] = self._rhofunc(phi[i], r, ra)
        return rho

    def _rhofunc(self, phi, r, ra):
        """ Dimensionless density as a function of phi and r (scalars only) """
        # Isotropic case first
        rho = exp(phi)*gammainc(self.n+0.5, phi)

        # Add anisotropy
        if (self.ra < self.ramax)&(phi>0)&(r>0):
            p = r/ra
            rho += self._dawsn_t(p, phi)
            rho /= (1+p**2)
        return rho

    def _get_v2(self, phi, r, rho, ra):
        v2, v2r, v2t = numpy.zeros(r.size), numpy.zeros(r.size), numpy.zeros(r.size)
        for i in range(r.size-1):
            v2[i], v2r[i], v2t[i] = self._rhov2func(phi[i], r[i], ra)/rho[i]

        return v2, v2r, v2t

    def _rhov2func(self, phi, r, ra):
        """ Product of density and mean square velocity """

        # Isotropic case first
        rhov2r = exp(phi)*gammainc(self.n+1.5, phi)
        rhov2 = rhov2r
        rhov2t = rhov2r

       # Add anisotropy
        if (ra < self.ramax)&(r>0):
            p, n = r/ra, self.n
            p2 = p**2
            p12 = 1+p2
            if(phi>self.phicrit):

                B, K, C = phi**(n+0.5)/gamma(n+1.5), n-1/p12, self._dawsn_t(p, phi)
                Z = 2*K + 1
                rhov2 *= (3+p2)/p12
                rhov2 += -2*B*K + 2*C*(K+p2*phi)/p2
                rhov2 /= p12

                rhov2r += B - C/p2
                rhov2r /= p12

                rhov2t *= 2./p12
                rhov2t += -Z*B + C*(Z + 2*p2*phi)/p2
                rhov2t /= p12

            elif (phi>0):
                t1 = phi**(n+1.5)/gamma(n+2.5)
                t2 = t1*phi/(n+2.5)
                rhov2 = 3*t1 + (-5*p2+3)*t2
                rhov2r = t1 + (1-p2)*t2
                rhov2t = 2*(t1 -(2*p2**2 + p2 -1)*t2/p12)
        else:
            rhov2 *= 3
            rhov2t *= 2
        return rhov2, rhov2r, rhov2t

    def _beta(self, r, v2r, v2t):
        beta = numpy.zeros(r.size)
        if (self.ra < self.ramax):
            c = (v2r>0)
            beta[c] = 1.0 - 0.5*v2t[c]/v2r[c]
        return beta

    def _odes(self, x, y, potonly):
        """ Solve ODEs """
        # y = [phi, u_j, U, K_j], where u = -M(<r)/G
        if (self.multi):
            derivs = [numpy.sum(y[1:1+self.nmbin])/x**2] if (x>0) else [0]
            for j in range(self.nmbin):
                phi, ra = y[0]/self.sig2[j], self.raj[j]
                derivs.append(-x**2*self._ah[j]*self._rhofunc(phi, x, ra))
            dUdx  = 2.0*pi*numpy.sum(derivs[1:1+self.nmbin])*y[0]
        else:
            derivs = [y[1]/x**2] if (x>0) else [0]
            derivs.append(-x**2*self._rhofunc(y[0], x, self.ra))
            dUdx  = 2.0*pi*derivs[1]*y[0]
        derivs.append(dUdx)

        if (not potonly): #dK_j/dx
            for j in range(self.nmbin):
                rhov2j = self._rhov2func(y[0]/self.sig2[j], x, self.raj[j])[0]
                derivs.append(self._ah[j]*self.sig2[j]*2*pi*x**2*rhov2j)
        return derivs

    def _setup_phi_interpolator(self):
        """ Setup interpolater for phi, works on scalar and arrays """
        # Generate piecewise 3th order polynomials to connect phi, using phi'
        self._interpolator_set = True
        phi_and_derivs = numpy.vstack([[self.phi],[self.dp1]]).T
        self._phi_poly =PiecewisePolynomial(self.r,phi_and_derivs,direction=1)


    def _scale(self):
        """
        Scales the model to the units set in the input: GS, MS, RS
        """
        Mstar = self.MS/self.M
        Gstar = self.GS/self.G
        if (self.scale_radius=='rh'): Rstar = self.RS/self.rh
        if (self.scale_radius=='rv'): Rstar = self.RS/self.rv
        v2star =  Gstar*Mstar/Rstar

        # Update the 3 scales and params that define the system
        self.G *= Gstar
        self.rs = Rstar
        self.sig2 *= v2star
        self.ra *= Rstar
        self.ramax *= Rstar

        # Scale all variable needed when run with potonly=True
        self.r, self.r0, self.rt = (q*Rstar for q in [self.r,self.r0,self.rt])
        self.rh, self.rv = (q*Rstar for q in [self.rh,self.rv])

        self.M *= Mstar
        self.phi *= v2star
        self.dp1 *= v2star/Rstar
        self.mc *= Mstar
        self.U *= Mstar*v2star
        self.A *= Mstar/(v2star**1.5*Rstar**3)

        if (self.multi):
            self.Mj *= Mstar
            self.raj *= Rstar
            self.r0j *= Rstar

        # All other stuff
        if (not self.potonly):
            self.rho *= Mstar/Rstar**3
            self.v2, self.v2r, self.v2t = (q*v2star for q in [self.v2,
                                                          self.v2r,self.v2t])
            self.K *= Mstar*v2star

            if (self.multi):
                self.rhoj *= Mstar/Rstar**3
                self.mcj *= Mstar
                self.rhj *= Rstar
                self.v2j,self.v2rj,self.v2tj=(q*v2star for q in
                                              [self.v2j, self.v2rj,self.v2tj])
                self.kj,self.Kj=(q*Mstar*v2star for q in [self.kj, self.Kj])

    def tonp(self, q):
        q = numpy.array([q]) if not hasattr(q,"__len__") else numpy.array(q)
        return q

    def interp_phi(self, r):
        """ Interpolate potential at r, works on scalar and arrays """

        if not hasattr(r,"__len__"): r = numpy.array([r])
        if (not self._interpolator_set): self._setup_phi_interpolator()

        phi = numpy.zeros([r.size])
        inrt = (r<self.rt)
        # Use 3th order polynomials to interp, using phi'
        if (sum(inrt)>0): phi[inrt] = self._phi_poly(r[inrt])
        return phi

    def df(self, *arg):
        """
        Returns the normalised DF, can only be called after Poisson solver
        Arguments can be:
          - r, v, j                (isotropic models)
          - r, v, theta, j         (anisotropic models)
          - x, y, z, vx, vy, vz, j (all models)
        Here j specifies the mass bin, j=0 for single mass
        Works with scalar and ndarray input
        """
        if len(arg) == 3:
            r, v = (self.tonp(q) for q in arg[:-1])
            j = arg[-1]
            r2, v2 = r**2, v**2

        if len(arg) == 4:
            r, v, theta = (self.tonp(q) for q in arg[:-1])
            j = arg[-1]
            r2, v2 = r**2, v**2

        if len(arg) == 7:
            x, y, z, vx, vy, vz = (self.tonp(q) for q in arg[:-1])
            j = arg[-1]
            r2 = x**2 + y**2 + z**2
            v2 = vx**2 + vy**2 + vz**2
            r, v = sqrt(r2), sqrt(v2)

        phi = self.interp_phi(r)
        vesc2 = 2.0*phi                            # Note: phi > 0

        DF = numpy.zeros([max(r.size, v.size)])
        c = (r<self.rt)&(v2<vesc2)

        E = (0.5*v2 - phi)/self.sig2[j]            # Dimensionless energy
        DF[c] = exp(-E[c])

        for i in range(self.n-1):
            DF[c] -= (-E[c])**i/factorial(i)

        if (self.raj[j] < self.ramax):
            if (len(arg)==7): J2 = v2*r2 - (x*vx + y*vy + z*vz)**2
            if (len(arg)==4): J2 = sin(theta)**2*v2*r2

            DF[c] *= exp(-J2[c]/(2*self.raj[j]**2*self.sig2[j]))

        DF[c] *= self.A[j]

        return DF

