Coordinate Systems
====================================

Transformations
------------------------------------
.. autofunction:: pyspedas.cotrans

Examples
^^^^^^^^^

.. code-block:: python
   
   import pyspedas
   pyspedas.themis.state(trange=['2015-10-16', '2015-10-17'], probe='c')

   from pyspedas import cotrans
   cotrans(name_in='thc_pos_gse', name_out='gsm_data', coord_in='gse', coord_out='gsm')
   cotrans(name_in='thc_pos_gse', name_out='sm_data', coord_in='gse', coord_out='sm')
   cotrans(name_in='thc_pos_gse', name_out='geo_data', coord_in='gse', coord_out='geo')

   from pytplot import tplot
   tplot(['gsm_data', 'sm_data', 'geo_data'])
   

LMN Coordinates
------------------------
.. autofunction:: pyspedas.cotrans.gsm2lmn.gsm2lmn

Getting/Setting the Coordinate System
----------------------------------------
.. autofunction:: pyspedas.cotrans_get_coord
.. autofunction:: pyspedas.cotrans_set_coord

Support Routines
------------------------
.. autofunction:: pyspedas.cotrans.cotrans_lib.get_time_parts
.. autofunction:: pyspedas.cotrans.cotrans_lib.csundir_vect
.. autofunction:: pyspedas.cotrans.cotrans_lib.cdipdir
.. autofunction:: pyspedas.cotrans.cotrans_lib.cdipdir_vect
.. autofunction:: pyspedas.cotrans.cotrans_lib.tgeigse_vect
.. autofunction:: pyspedas.cotrans.cotrans_lib.subgei2gse
.. autofunction:: pyspedas.cotrans.cotrans_lib.tgsegei_vect
.. autofunction:: pyspedas.cotrans.cotrans_lib.subgse2gei
.. autofunction:: pyspedas.cotrans.cotrans_lib.tgsegsm_vect
.. autofunction:: pyspedas.cotrans.cotrans_lib.subgse2gsm
.. autofunction:: pyspedas.cotrans.cotrans_lib.tgsmgse_vect
.. autofunction:: pyspedas.cotrans.cotrans_lib.subgsm2gse
.. autofunction:: pyspedas.cotrans.cotrans_lib.tgsmsm_vect
.. autofunction:: pyspedas.cotrans.cotrans_lib.subgsm2sm
.. autofunction:: pyspedas.cotrans.cotrans_lib.tsmgsm_vect
.. autofunction:: pyspedas.cotrans.cotrans_lib.subsm2gsm
.. autofunction:: pyspedas.cotrans.cotrans_lib.subgei2geo
.. autofunction:: pyspedas.cotrans.cotrans_lib.subgeo2gei
.. autofunction:: pyspedas.cotrans.cotrans_lib.subgeo2mag
.. autofunction:: pyspedas.cotrans.cotrans_lib.submag2geo
.. autofunction:: pyspedas.cotrans.cotrans_lib.ctv_mm_mult
.. autofunction:: pyspedas.cotrans.cotrans_lib.j2000_matrix_vec
.. autofunction:: pyspedas.cotrans.cotrans_lib.ctv_mx_vec_rot
.. autofunction:: pyspedas.cotrans.cotrans_lib.subgei2j2000
.. autofunction:: pyspedas.cotrans.cotrans_lib.subj20002gei
.. autofunction:: pyspedas.cotrans.cotrans_lib.subcotrans
