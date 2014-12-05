========
Usage
========

To use dflab in a project::

    import dflab


Description of the distribution function
----------------------------------------

**Description of the distribution function:**

The isotropic distribution functions are defined as

.. math::
   f_n(E) = \displaystyle \begin{cases}
   A\exp(-E), &n=1 \\
   \displaystyle A\left[\exp(-E) - \sum_{m=0}^{n-2} \frac{(-E)^m}{!m} \right], &n>1
   \end{cases}

where :math:`\displaystyle E = \frac{v^2/2 - \phi + \phi(r_{\rm t})}{\sigma^2}` and :math:`\sigma` is a velocity scale and :math:`0 < \phi-\phi(r_{\rm t}) <W_0/\sigma^2`

*  n = 1 : `Woolley (1954) <http://adsabs.harvard.edu/abs/1954MNRAS.114..191W>`_
*  n = 2 : `King (1966) <http://adsabs.harvard.edu/abs/1966AJ.....71...64K>`_
*  n = 3 : `Wilson (1975) <http://adsabs.harvard.edu/abs/1975AJ.....80..175W>`_

Radial anisotropy a la `Michie (1963)
<http://adsabs.harvard.edu/abs/1963MNRAS.125..127M>`_ can be
included

.. math::
   f_n(E, J^2) = \exp(-J^2)f_n(E),

where :math:`J^2 = (rv_t)^2/(2r_{\rm a}^2\sigma^2)`, here :math:`r_{\rm a}` is the anisotropy radius

Multi-mass models are found by summing the DFs of individual mass
components and adopting for each component following `Gunn &
Griffin (1979) <http://adsabs.harvard.edu/abs/1979AJ.....84..752G>`_

.. math::
   \sigma_j       &\propto  \mu_j^{-\delta}\\
   r_{{\rm a},j}  &\propto  \mu_j^{\eta}

where :math:`\mu_j = m_j/\bar{m}` and :math:`\bar{m}` is the central density weighted mean mass.

