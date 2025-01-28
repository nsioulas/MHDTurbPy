import os
import unittest
from pytplot import data_exists
import pyspedas


class LoadTestCases(unittest.TestCase):
    def test_load_mag_data(self):
        mag_vars = pyspedas.solo.mag(time_clip=True)
        self.assertTrue(data_exists('B_RTN'))
        mag_vars = pyspedas.solo.mag(datatype='rtn-normal-1-minute')
        self.assertTrue(data_exists('B_RTN'))
        mag_vars = pyspedas.solo.mag(notplot=True, datatype='rtn-burst')
        self.assertTrue('B_RTN' in mag_vars)

    def test_load_mag_data_prefix_none(self):
        mag_vars = pyspedas.solo.mag(time_clip=True, prefix=None)
        self.assertTrue(data_exists('B_RTN'))
        mag_vars = pyspedas.solo.mag(datatype='rtn-normal-1-minute', prefix=None)
        self.assertTrue(data_exists('B_RTN'))

    def test_load_mag_data_suffix_none(self):
        mag_vars = pyspedas.solo.mag(time_clip=True, suffix=None)
        self.assertTrue(data_exists('B_RTN'))
        mag_vars = pyspedas.solo.mag(datatype='rtn-normal-1-minute', suffix=None)
        self.assertTrue(data_exists('B_RTN'))

    def test_load_mag_data_prefix_suffix(self):
        mag_vars = pyspedas.solo.mag(time_clip=True, prefix='pre_', suffix='_suf')
        self.assertTrue(data_exists('pre_B_RTN_suf'))
        mag_vars = pyspedas.solo.mag(datatype='rtn-normal-1-minute', prefix='pre_', suffix='_suf')
        self.assertTrue(data_exists('pre_B_RTN_suf'))

    def test_load_mag_ll02_data(self):
        mag_vars = pyspedas.solo.mag(level='ll02', trange=['2020-08-04', '2020-08-06'])
        self.assertTrue(data_exists('B_RTN'))
        self.assertTrue(data_exists('B_SRF'))

    def test_load_epd_data(self):
        epd_vars = pyspedas.solo.epd()
        self.assertTrue(data_exists('Magnet_Rows_Flux'))
        self.assertTrue(data_exists('Integral_Rows_Flux'))
        self.assertTrue(data_exists('Magnet_Cols_Flux'))
        self.assertTrue(data_exists('Integral_Cols_Flux'))

    def test_load_epd_data_pefix_none(self):
        epd_vars = pyspedas.solo.epd(prefix=None)
        self.assertTrue(data_exists('Magnet_Rows_Flux'))

    def test_load_epd_data_suffix_none(self):
        epd_vars = pyspedas.solo.epd(suffix=None)
        self.assertTrue(data_exists('Magnet_Rows_Flux'))

    def test_load_epd_data_prefix_suffix(self):
        epd_vars = pyspedas.solo.epd(prefix='pre_', suffix='_suf')
        self.assertTrue(data_exists('pre_Magnet_Rows_Flux_suf'))

    def test_load_rpw_data(self):
        rpw_vars = pyspedas.solo.rpw()
        self.assertTrue(data_exists('AVERAGE_NR'))
        self.assertTrue(data_exists('TEMPERATURE'))
        # self.assertTrue(data_exists('FLUX_DENSITY1'))
        # self.assertTrue(data_exists('FLUX_DENSITY2'))
    
    def test_load_rpw_data_prefix_none(self):
        rpw_vars = pyspedas.solo.rpw(prefix=None)
        self.assertTrue(data_exists('AVERAGE_NR'))

    def test_load_rpw_data_suffix_none(self):
        rpw_vars = pyspedas.solo.rpw(suffix='None')
        self.assertTrue(data_exists('AVERAGE_NR'))

    def test_load_rpw_data_prefix_suffix(self):
        rpw_vars = pyspedas.solo.rpw(prefix='pre_', suffix='_suf')
        self.assertTrue(data_exists('pre_AVERAGE_NR_suf'))

    def test_load_swa_data(self):
        swa_vars = pyspedas.solo.swa()
        self.assertTrue(data_exists('eflux'))
        swa_vars = pyspedas.solo.swa(level='l2', datatype='eas1-nm3d-def')
        self.assertTrue(data_exists('SWA_EAS1_NM3D_DEF_Data'))
        swa_vars = pyspedas.solo.swa(notplot=True)
        self.assertTrue('eflux' in swa_vars)

    def test_load_swa_data_prefix_none(self):
        swa_vars = pyspedas.solo.swa(prefix=None)
        self.assertTrue(data_exists('eflux'))

    def test_load_swa_data_suffix_none(self):
        swa_vars = pyspedas.solo.swa(suffix=None)
        self.assertTrue(data_exists('eflux'))

    def test_load_swa_data_prefix_suffix(self):
        swa_vars = pyspedas.solo.swa(prefix='pre_', suffix='_suf')
        self.assertTrue(data_exists('pre_eflux_suf'))

    def test_load_swa_l1_data(self):
        swa_vars = pyspedas.solo.swa(level='l1', datatype='eas-padc')
        self.assertTrue(data_exists('SWA_EAS_BM_Data'))
        self.assertTrue(data_exists('SWA_EAS_MagDataUsed'))
        swa_vars = pyspedas.solo.swa(level='l1', datatype='his-pha', trange=['2020-06-03', '2020-06-04'])
        self.assertTrue(data_exists('HIS_PHA_EOQ_STEP'))

    def test_downloadonly(self):
        files = pyspedas.solo.mag(downloadonly=True)
        self.assertTrue(os.path.exists(files[0]))


if __name__ == '__main__':
    unittest.main()
