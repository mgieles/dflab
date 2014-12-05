========
Usage
========

To use dflab in a project::

    import dflab


Description of the distribution function
----------------------------------------

The isotropic distribution functions are defined as

             / Aexp(-E)                                 , n=1
  f(E, n) = <
             \ A[exp(-E) - Sum_{m=0}^{n-2} (-E)^m/!m ]  , n>1
.. math::
      f_n(E) = \begin{cases}
      Aexp(-E)  , n=1 \\
      A[exp(-E) - Sum_{m=0}^{n-2} (-E)^m/!m ]  , n>1
      \end{cases}

where E = (v2/2 - phi + phi(r_t))/sig2, sig is a velocity scale and
      0<phi<W0/sig2

 *  n = 1 : `Woolley (1954) <http://adsabs.harvard.edu/abs/1954MNRAS.114..191W>`_
 *  n = 2 : `King (1966) <http://adsabs.harvard.edu/abs/1966AJ.....71...64K>`_
 *  n = 3 : `Wilson (1975) <http://adsabs.harvard.edu/abs/1975AJ.....80..175W>`_

The anisotropic models are

.. math::
  f(E, J^2, n) = \exp(-J^2)f(E, n),

where J^2 = (r*vt)^2/(2*ra2*sig2), here ra is the anisotropy radius

Multi-mass models are found by summing the DFs of individual mass
components and adopting for each component

.. math::
\sigma_j \propto \mu_j^{-delta},\\
r_{{\rm a},j} \propto \mu_j^{eta}

where :math:`\mu_j = m_j/m` and :math:`m` is the central density weighted mean mj

"""

