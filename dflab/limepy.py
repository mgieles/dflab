# -*- coding: utf-8 -*-
from __future__ import division, absolute_import

import numpy
from numpy import exp, sqrt, pi, sin
from scipy.interpolate import PiecewisePolynomial
from scipy.special import gamma, gammainc, dawsn, hyp1f1 
from scipy.integrate import ode, quad
from math import factorial

#     Authors: Mark Gieles, Alice Zocchi (Surrey 2014)

class limepy:
    def __init__(self, W0, g, **kwargs):
        r"""

        (MM, A)limepy

        (Multi-Mass, Anisotropic) Lowered Isothermal Model Explorer in Python


        Parameters:

        W0 : scalar
           Central dimensionless potential
        g : scalar
          Order of truncation [0=Woolley, 1=King, 2=Wilson]; default=1
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


        Examples:
        
        Construct a Woolley model with W0 = 7 and print r_t/r_0 and r_v/r_h

        >>> k = limepy(7, 0)
        >>> print k.rt/k.r0, k.rv/k.rh
        >>> 19.1293450185 1.17562140501
        
        Construct a Michie-King model and print r_a/r_h

        >>> a = limepy(7, 1, ra=2)
        >>> print a.ra/a.rh
        >>> 5.55543555675
        
        Create a Wilson model with W_0 = 12 in Henon/N-body units: G = M = 
        r_v = 1 and print the normalisation constant A of the DF and the 
        value of the DF in the centre:
        
        >>> w = limepy(12, 2, scale=True, GS=1, MS=1, RS=1, scale_radius='rv')
        >>> print w.A, w.df(0,0)
        >>> [ 0.00800902] [ 1303.40165523]

        Multi-mass model in physical units with r_h = 1 pc and M = 10^5 M_sun
        and print central densities of each bin over the total central density

        >>> m = limepy(7, 1, mj=[0.3,1,5], Mj=[9,3,1], scale=True, MS=1e5,RS=1)
        >>> print m.alpha
        >>> array([ 0.3072483 ,  0.14100799,  0.55174371])

        """

        self._set_kwargs(W0, g, **kwargs)

        if (self.multi):
            self._init_multi(self.mj, self.Mj)
            while self.diff > self.diffcrit:
                self._poisson(True)
                self._set_alpha()
                if self.niter > 100:
                    self.converged=False

        self.r0 = 3./sqrt(self._rho(self.W0, 0, self.ramax))[0]

        if (self.multi): self.r0j = sqrt(self.sig2)*self.r0

        self._poisson(self.potonly)
        if (self.multi): self.Mj = self._Mjtot
        if (self.scale): self._scale()

        if (self.verbose):
            print "\n Model properties: "
            print " ----------------- "
            print " W0 = %5.2f; n = %1i"%(self.W0, self.g)
            if (self.potonly):
                print " M = %10.3f; U = %10.4f "%(self.M, self.U)
            else:
                out1=(self.M,self.U,self.K,2*self.Kr/self.Kt,self.K/self.U)
                print " M = %10.3e; U = %10.4e; K = %10.4e; 2Kr/Kt = %5.3f; Q = %6.4f"%out1
            out2=(self.rv/self.rh,self.rh/self.r0,self.rt/self.r0)
            print " rv/rh = %4.3f; rh/r0 = %6.3f; rt/r0 = %7.3f"%out2

    def _set_kwargs(self, W0, g, **kwargs):
        if (g<0): raise ValueError("Error: g must be larger or equal to 0")
        
        self.W0, self.g = W0, g

        self.MS, self.RS, self.GS = 1e5, 3, 0.004302
        self.scale_radius = 'rh'
        self.scale = False
        self.max_step = 1e4
        self.diffcrit = 1e-10

        self.nmbin, self.delta, self.eta = 1, 0.5, 0.5

        self.G = 1.0/(4.0*pi)
        self.mu, self._ah = numpy.array([1.0]), numpy.array([1.0])
        self.sig2 = numpy.array([1.0])
        self.niter = 0

        self.potonly, self.multi, self.verbose = [False]*3
        self.ra, self.ramax = 1e6, 100

        self.nstep=0
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
        self.raj = numpy.array([self.ra])

        return

    def _logcheck(self, t, y):
        """ Logs steps and checks for final values """
        if (t>0): self.r, self.y = numpy.r_[self.r, t], numpy.c_[self.y, y]
        self.nstep+=1
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
        if (not potonly): self.y = numpy.r_[self.y, numpy.zeros(2*self.nmbin)]

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

        if (self.rt < 1e4)&(sol.successful()):
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
            self.Kr = numpy.sum(sol.y[2+2*self.nmbin:2+3*self.nmbin])
            self.Kt = self.K - self.Kr

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

                    kj = self.y[2+self.nmbin+j,:]
                    krj = self.y[2+2*self.nmbin+j,:]

                    mcj = -numpy.r_[self.y[1+j,:], self.y[1+j,-1]]/self.G
                    rhj = numpy.interp(0.5*mcj[-1], mcj, self.r)

                    if (j==0):
                        self.rhoj = rhoj
                        self.rho = self._ah[j]*rhoj
                        self.v2j, self.v2rj, self.v2tj = v2j, v2rj, v2tj
                        self.v2 = self._Mjtot[j]*v2j/self.M
                        self.v2r = self._Mjtot[j]*v2rj/self.M
                        self.v2t = self._Mjtot[j]*v2tj/self.M

                        self.betaj = betaj
                        self.kj, self.krj, self.Kj, self.Krj = kj, krj, kj[-1], krj[-1]
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
                        self.krj = numpy.vstack((self.krj, krj))
                        self.Kj = numpy.r_[self.Kj, kj[-1]]
                        self.Krj = numpy.r_[self.Krj, krj[-1]]
                        self.rhj = numpy.r_[self.rhj,rhj]
                        self.mcj = numpy.vstack((self.mcj, mcj))
            self.beta = self._beta(self.r, self.v2r, self.v2t)


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
        rho = exp(phi)*gammainc(self.g + 1.5, phi)

        # Add anisotropy
        if (self.ra < self.ramax)&(phi>0)&(r>0):
            p = r/ra
            p2 = p**2
            g = self.g
            g3, g5, g7, fp2 = g+1.5, g+2.5, g+3.5, phi*p2
            P2 = 1 - hyp1f1(1, g3, -fp2)
            if fp2 < 1e-2: P2 = fp2/g3 - fp2**2/(g3*g5) + fp2**3/(g3*g5*g7)
            if fp2 > 500: P2 = 1 - (g3-1)/fp2 
            rho += phi**(g+0.5)/gamma(g+1.5)*P2
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
        rhov2r = exp(phi)*gammainc(self.g + 2.5, phi)
        rhov2  = rhov2r
        rhov2t = rhov2r

       # Add anisotropy
        if (ra < self.ramax)&(r>0)&(phi>0):
            p, g = r/ra, self.g
            p2 = p**2
            p12 = 1+p2

            g3, g5, g7, fp2 = g + 1.5, g + 2.5, g + 3.5, phi*p2
            G3 = gamma(g3)
            P1 = phi**g3/G3/p12
            P2 = (1 - hyp1f1(1, g3, -fp2) )/fp2

            # Taylor series near 0 for small fp2, accurate to 1e-8
            if fp2 < 1e-2: P2 = 1/g3 - fp2/(g3*g5) + fp2**2/(g3*g5*g7)
            if fp2 > 500: P2 = 1/fp2 - (g3-1)/fp2**2

            rhov2r /= p12
            rhov2r += P1*(1/g3 - P2)

            P3 = fp2*hyp1f1(2, g7, -fp2)/(g3*g5)
            P4 = hyp1f1(1, g5, -fp2)/g3
            rhov2t *= 2/p12**2 
            rhov2t += 2*P1*(1/g3/p12 + p2/p12*P2 + P3 - P4)

            P5 = (3+p2)/p12
            rhov2 *= P5/p12
            rhov2 += P1*(P5/g3 + (p2-1)/p12*P2 + 2*P3 - 2*P4)
        else:
            rhov2 *= 3
            rhov2t *= 2
        return rhov2, rhov2r, rhov2t

    def _beta(self, r, v2r, v2t):
        beta = numpy.zeros(r.size)
        if (self.ra < self.ramax):
            c = (v2r>0.)
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
            rhov2j, rhov2rj = [], []
            for j in range(self.nmbin):
                rv2, rv2r, rv2t = self._rhov2func(y[0]/self.sig2[j], x, self.raj[j])
                rhov2j.append(self._ah[j]*self.sig2[j]*2*pi*x**2*rv2)
                rhov2rj.append(self._ah[j]*self.sig2[j]*2*pi*x**2*rv2r)

            for j in range(self.nmbin):
                derivs.append(rhov2j[j])
            for j in range(self.nmbin):
                derivs.append(rhov2rj[j])

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

        # Anisotropy radii
        self.ra, self.raj, self.ramax = (q*Rstar for q in [self.ra,self.raj,self.ramax])

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
            self.r0j *= Rstar

        # All other stuff
        if (not self.potonly):
            self.rho *= Mstar/Rstar**3
            self.v2, self.v2r, self.v2t = (q*v2star for q in [self.v2,
                                                          self.v2r,self.v2t])
            self.K,self.Kr,self.Kt=(q*Mstar*v2star for q in [self.K, self.Kr, self.Kt])

            if (self.multi):
                self.rhoj *= Mstar/Rstar**3
                self.mcj *= Mstar
                self.rhj *= Rstar
                self.v2j,self.v2rj,self.v2tj=(q*v2star for q in
                                              [self.v2j, self.v2rj,self.v2tj])
                self.kj,self.Krj,self.Kj=(q*Mstar*v2star for q in [self.kj, self.Krj, self.Kj])

    def _tonp(self, q):
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
        if (len(arg)<2)|(len(arg)==5)|(len(arg)==6)|(len(arg)>7):
            raise ValueError("Error: df needs 2, 3, 4 or 7 arguments")

        if len(arg) == 2:
            r, v = (self._tonp(q) for q in arg)
            j = 0

        if len(arg) == 3:
            r, v = (self._tonp(q) for q in arg[:-1])
            j = arg[-1]

        if len(arg) == 4:
            r, v, theta = (self._tonp(q) for q in arg[:-1])
            j = arg[-1]

        if len(arg) < 7: r2, v2 = r**2, v**2

        if len(arg) == 7:
            x, y, z, vx, vy, vz = (self._tonp(q) for q in arg[:-1])
            j = arg[-1]
            r2 = x**2 + y**2 + z**2
            v2 = vx**2 + vy**2 + vz**2
            r, v = sqrt(r2), sqrt(v2)

        phi = self.interp_phi(r)
        vesc2 = 2.0*phi                        # Note: phi > 0

        DF = numpy.zeros([max(r.size, v.size)])
        c = (r<self.rt)&(v2<vesc2)

        E = (phi-0.5*v2)/self.sig2[j]          # Dimensionless positive energy
        DF[c] = exp(E[c])

        # Float truncation parameter 
        # Following Gomez-Leyton & Velazquez 2014, J. Stat. Mech. 4, 6

        if (self.g>0): DF[c] *= gammainc(self.g, E[c])

        if (self.raj[j] < self.ramax):
            if (len(arg)==7): J2 = v2*r2 - (x*vx + y*vy + z*vz)**2
            if (len(arg)==4): J2 = sin(theta)**2*v2*r2

            DF[c] *= exp(-J2[c]/(2*self.raj[j]**2*self.sig2[j]))

        DF[c] *= self.A[j]

        return DF



