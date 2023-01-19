Colorado Student Space Weather Experiment (CSSWE)
========================================================================
The routines in this module can be used to load data from the Colorado Student Space Weather Experiment (CSSWE) mission.


Relativistic Electron and Proton Telescope integrated little experiment (REPTile)
----------------------------------------------------------
.. autofunction:: pyspedas.csswe.reptile

Example
^^^^^^^^^

.. code-block:: python
   
   import pyspedas
   from pytplot import tplot
   reptile_vars = pyspedas.csswe.reptile(trange=['2013-11-5', '2013-11-6'])
   tplot(['E1flux', 'E2flux', 'E3flux', 'P1flux', 'P2flux', 'P3flux'])

.. image:: _static/csswe_reptile.png
   :align: center
   :class: imgborder



