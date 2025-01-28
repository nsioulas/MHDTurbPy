
import os
import unittest
from pytplot import data_exists, del_data, timespan,tplot
from pyspedas.projects.erg import erg_lep_part_products

import pyspedas
import pytplot

display=False

class LoadTestCases(unittest.TestCase):

    def test_lepi_theta(self):
        del_data('*')
        # Load LEP-i Lv.2 3-D flux data
        timespan('2017-04-05 21:45:00', 2.25, keyword='hours')
        pyspedas.erg.lepi( trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'], datatype='3dflux' )
        # Calculate and plot energy spectrum
        vars = erg_lep_part_products( 'erg_lepi_l2_3dflux_FPDU', outputs='theta', trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'] )
        tplot( 'erg_lepi_l2_3dflux_FPDU_theta', display=display, save_png='erg_lepi_theta.png' )
        self.assertTrue(data_exists('erg_lepi_l2_3dflux_FPDU_theta'))
        self.assertTrue('erg_lepi_l2_3dflux_FPDU_theta' in vars)

    def test_lepi_theta_limit_phi(self):
        del_data('*')
        # Load LEP-i Lv.2 3-D flux data
        timespan('2017-04-05 21:45:00', 2.25, keyword='hours')
        pyspedas.erg.lepi( trange=[ '2017-04-05 21:45:00', '2017-04-05 22:45:00'], datatype='3dflux' )
        # Calculate and plot energy spectrum
        vars = erg_lep_part_products( 'erg_lepi_l2_3dflux_FPDU', outputs='theta', trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'], phi_in=[0., 180.0] )
        tplot( 'erg_lepi_l2_3dflux_FPDU_theta', display=display, save_png='erg_lepi_theta_limit_phi.png' )
        self.assertTrue(data_exists('erg_lepi_l2_3dflux_FPDU_theta'))
        self.assertTrue('erg_lepi_l2_3dflux_FPDU_theta' in vars)

    def test_lepi_theta_no_trange(self):
        del_data('*')
        # Load LEP-i Lv.2 3-D flux data
        timespan('2017-04-05 21:45:00', 2.25, keyword='hours')
        pyspedas.erg.lepi( trange=[ '2017-04-05 21:45:00', '2017-04-05 22:45:00'], datatype='3dflux' )
        # Calculate and plot energy spectrum
        vars = erg_lep_part_products( 'erg_lepi_l2_3dflux_FPDU', outputs='theta' )
        tplot( 'erg_lepi_l2_3dflux_FPDU_theta', display=display, save_png='erg_lepi_theta_no_trange.png' )
        self.assertTrue(data_exists('erg_lepi_l2_3dflux_FPDU_theta'))
        self.assertTrue('erg_lepi_l2_3dflux_FPDU_theta' in vars)

    def test_lepi_phi(self):
        del_data('*')
        # Load LEP-i Lv.2 3-D flux data
        timespan('2017-04-05 21:45:00', 2.25, keyword='hours')
        pyspedas.erg.lepi( trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'], datatype='3dflux' )
        # Calculate and plot energy spectrum
        vars = erg_lep_part_products( 'erg_lepi_l2_3dflux_FPDU', outputs='phi', trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'] )
        tplot( 'erg_lepi_l2_3dflux_FPDU_phi', display=display, save_png='erg_lepi_phi.png' )
        self.assertTrue(data_exists('erg_lepi_l2_3dflux_FPDU_phi'))
        self.assertTrue('erg_lepi_l2_3dflux_FPDU_phi' in vars)


    def test_lepi_pa(self):
        del_data('*')
        # Load LEP-i Lv.2 3-D flux data
        timespan('2017-04-05 21:45:00', 2.25, keyword='hours')
        pyspedas.erg.lepi( trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'], datatype='3dflux' )
        vars = pyspedas.erg.mgf(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary B-field data
        vars = pyspedas.erg.orb(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary orbit data
        mag_vn = 'erg_mgf_l2_mag_8sec_dsi'
        pos_vn = 'erg_orb_l2_pos_gse'
        # Calculate and plot energy spectrum
        vars = erg_lep_part_products( 'erg_lepi_l2_3dflux_FPDU', mag_name=mag_vn, pos_name=pos_vn, outputs='pa', trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'] )
        tplot( 'erg_lepi_l2_3dflux_FPDU_pa', display=display, save_png='erg_lepi_pa.png' )
        self.assertTrue(data_exists('erg_lepi_l2_3dflux_FPDU_pa'))
        self.assertTrue('erg_lepi_l2_3dflux_FPDU_pa' in vars)

    def test_lepi_gyro(self):
        del_data('*')
        # Load LEP-i Lv.2 3-D flux data
        timespan('2017-04-05 21:45:00', 2.25, keyword='hours')
        pyspedas.erg.lepi( trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'], datatype='3dflux' )
        vars = pyspedas.erg.mgf(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary B-field data
        vars = pyspedas.erg.orb(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary orbit data
        mag_vn = 'erg_mgf_l2_mag_8sec_dsi'
        pos_vn = 'erg_orb_l2_pos_gse'
        # Calculate and plot energy spectrum
        vars = erg_lep_part_products( 'erg_lepi_l2_3dflux_FPDU', mag_name=mag_vn, pos_name=pos_vn, outputs='gyro', trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'] )
        tplot( 'erg_lepi_l2_3dflux_FPDU_gyro', display=display, save_png='erg_lepi_gyro.png' )
        self.assertTrue(data_exists('erg_lepi_l2_3dflux_FPDU_gyro'))
        self.assertTrue('erg_lepi_l2_3dflux_FPDU_gyro' in vars)

    def test_lepi_moments(self):
        del_data('*')
        # Load LEP-i Lv.2 3-D flux data
        timespan('2017-04-05 21:45:00', 2.25, keyword='hours')
        pyspedas.erg.lepi( trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'], datatype='3dflux' )
        vars = pyspedas.erg.mgf(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary B-field data
        vars = pyspedas.erg.orb(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary orbit data
        mag_vn = 'erg_mgf_l2_mag_8sec_dsi'
        pos_vn = 'erg_orb_l2_pos_gse'
        # Calculate and plot energy spectrum
        vars = erg_lep_part_products( 'erg_lepi_l2_3dflux_FPDU', mag_name=mag_vn, pos_name=pos_vn, outputs='moments', trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'] )
        tplot(vars, display=display, save_png='erg_lepi_moments.png' )
        self.assertTrue(data_exists('erg_lepi_l2_3dflux_FPDU_density'))
        self.assertTrue('erg_lepi_l2_3dflux_FPDU_density' in vars)

    def test_lepi_fac_moments(self):
        del_data('*')
        # Load LEP-i Lv.2 3-D flux data
        timespan('2017-04-05 21:45:00', 2.25, keyword='hours')
        pyspedas.erg.lepi( trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'], datatype='3dflux' )
        vars = pyspedas.erg.mgf(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary B-field data
        vars = pyspedas.erg.orb(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary orbit data
        mag_vn = 'erg_mgf_l2_mag_8sec_dsi'
        pos_vn = 'erg_orb_l2_pos_gse'
        # Calculate and plot energy spectrum
        vars = erg_lep_part_products( 'erg_lepi_l2_3dflux_FPDU', mag_name=mag_vn, pos_name=pos_vn, outputs='fac_moments', trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'] )
        tplot(vars, display=display, save_png='erg_lepi_fac_moments.png' )
        self.assertTrue(data_exists('erg_lepi_l2_3dflux_FPDU_density_mag'))
        self.assertTrue('erg_lepi_l2_3dflux_FPDU_density_mag' in vars)

    def test_lepi_fac_energy(self):
        del_data('*')
        # Load LEP-i Lv.2 3-D flux data
        timespan('2017-04-05 21:45:00', 2.25, keyword='hours')
        pyspedas.erg.lepi( trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'], datatype='3dflux' )
        vars = pyspedas.erg.mgf(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary B-field data
        vars = pyspedas.erg.orb(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary orbit data
        mag_vn = 'erg_mgf_l2_mag_8sec_dsi'
        pos_vn = 'erg_orb_l2_pos_gse'
        # Calculate and plot energy spectrum
        vars = erg_lep_part_products( 'erg_lepi_l2_3dflux_FPDU', mag_name=mag_vn, pos_name=pos_vn, outputs='fac_energy', trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'] )
        tplot(vars, display=display, save_png='erg_lepi_fac_energy.png' )
        self.assertTrue(data_exists('erg_lepi_l2_3dflux_FPDU_energy_mag'))
        self.assertTrue('erg_lepi_l2_3dflux_FPDU_energy_mag' in vars)

    def test_lepi_energy(self):
        del_data('*')
        # Load LEP-i Lv.2 3-D flux data
        timespan('2017-04-05 21:45:00', 2.25, keyword='hours')
        pyspedas.erg.lepi( trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'], datatype='3dflux' )
        # Calculate and plot energy spectrum
        vars = erg_lep_part_products( 'erg_lepi_l2_3dflux_FPDU', outputs='energy', trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'] )
        tplot( 'erg_lepi_l2_3dflux_FPDU_energy', display=display, save_png='erg_lepi_en_spec.png' )
        self.assertTrue(data_exists('erg_lepi_l2_3dflux_FPDU_energy'))
        self.assertTrue('erg_lepi_l2_3dflux_FPDU_energy' in vars)

    def test_lepi_pad(self):
        del_data('*')
        # Load LEP-i Lv.2 3-D flux data
        timespan('2017-04-05 21:45:00', 2.25, keyword='hours')
        pyspedas.erg.lepi( trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'], datatype='3dflux' )
        vars = pyspedas.erg.mgf(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary B-field data
        vars = pyspedas.erg.orb(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary orbit data
        mag_vn = 'erg_mgf_l2_mag_8sec_dsi'
        pos_vn = 'erg_orb_l2_pos_gse'
        # Calculate the pitch angle distribution
        vars = erg_lep_part_products('erg_lepi_l2_3dflux_FPDU', outputs='pa', energy=[15000., 22000.], fac_type='xdsi',
                                     mag_name=mag_vn, pos_name=pos_vn,
                                     trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])
        tplot( 'erg_lepi_l2_3dflux_FPDU_pa', display=display, save_png='erg_lepi_pad.png' )
        self.assertTrue(data_exists('erg_lepi_l2_3dflux_FPDU_pa'))
        self.assertTrue('erg_lepi_l2_3dflux_FPDU_pa' in vars)

    def test_lepi_energy_limit_gyro(self):
        del_data('*')
        # Load LEP-i Lv.2 3-D flux data
        timespan('2017-04-05 21:45:00', 2.25, keyword='hours')
        pyspedas.erg.lepi(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'], datatype='3dflux')
        vars = pyspedas.erg.mgf(
            trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary B-field data
        vars = pyspedas.erg.orb(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary orbit data
        mag_vn = 'erg_mgf_l2_mag_8sec_dsi'
        pos_vn = 'erg_orb_l2_pos_gse'
        # Calculate the pitch angle distribution
        vars = erg_lep_part_products('erg_lepi_l2_3dflux_FPDU', outputs='energy', gyro=[0., 180.],
                                     fac_type='xdsi',
                                     mag_name=mag_vn, pos_name=pos_vn,
                                     trange=['2017-04-05 21:45:00', '2017-04-05 22:45:00'])
        tplot( 'erg_lepi_l2_3dflux_FPDU_energy_mag', display=display, save_png='erg_lepi_energy_limit_gyro.png' )
        self.assertTrue(data_exists('erg_lepi_l2_3dflux_FPDU_energy_mag'))
        self.assertTrue('erg_lepi_l2_3dflux_FPDU_energy_mag' in vars)

    def test_lepi_en_pad_limit(self):
        del_data('*')
        # Load LEP-i Lv.2 3-D flux data
        timespan('2017-04-05 21:45:00', 2.25, keyword='hours')
        pyspedas.erg.lepi( trange=[ '2017-04-05 21:45:00', '2017-04-05 23:59:59'], datatype='3dflux' )
        vars = pyspedas.erg.mgf(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary B-field data
        vars = pyspedas.erg.orb(trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'])  # Load necessary orbit data
        mag_vn = 'erg_mgf_l2_mag_8sec_dsi'
        pos_vn = 'erg_orb_l2_pos_gse'
        # Calculate energy-time spectra of electron flux for limited pitch-angle (PA) ranges
        ## Here we calculate energy-time spectra for PA = 0-10 deg and PA = 80-100 deg.
        vars1 = erg_lep_part_products('erg_lepi_l2_3dflux_FPDU', outputs='fac_energy', pitch=[80., 100.],
                                     fac_type='xdsi', mag_name=mag_vn, pos_name=pos_vn,
                                     trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'], suffix='_pa80-100')
        vars2 = erg_lep_part_products('erg_lepi_l2_3dflux_FPDU', outputs='fac_energy', pitch=[0., 10.], fac_type='xdsi',
                                     mag_name=mag_vn, pos_name=pos_vn,
                                     trange=['2017-04-05 21:45:00', '2017-04-05 23:59:59'], suffix='_pa0-10')
        ## Decorate the obtained spectrum variables
        pytplot.options('erg_lepi_l2_3dflux_FPDU_energy_mag_pa80-100', 'ytitle', 'LEP-i flux\nPA: 80-100\n\n[eV]')
        pytplot.options('erg_lepi_l2_3dflux_FPDU_energy_mag_pa0-10', 'ytitle', 'LEP-i flux\nPA: 0-10\n\n[eV]')
        tplot(['erg_lepi_l2_3dflux_FPDU_energy_mag_pa80-100', 'erg_lepi_l2_3dflux_FPDU_energy_mag_pa0-10'], display=display, save_png='erg_lepi_en_pa_limit.png')
        self.assertTrue(data_exists('erg_lepi_l2_3dflux_FPDU_energy_mag_pa0-10'))
        self.assertTrue('erg_lepi_l2_3dflux_FPDU_energy_mag_pa0-10' in vars2)
        self.assertTrue(data_exists('erg_lepi_l2_3dflux_FPDU_energy_mag_pa80-100'))
        self.assertTrue('erg_lepi_l2_3dflux_FPDU_energy_mag_pa80-100' in vars1)

if __name__ == '__main__':
    unittest.main()