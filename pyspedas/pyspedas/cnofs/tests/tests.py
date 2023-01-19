import os
import unittest
from pyspedas.utilities.data_exists import data_exists
import pyspedas


class LoadTestCases(unittest.TestCase):
    def test_load_cindi_data(self):
        c_vars = pyspedas.cnofs.cindi(time_clip=True)
        self.assertTrue(data_exists('ionVelocityX'))
        self.assertTrue(data_exists('ionVelocityY'))
        self.assertTrue(data_exists('ionVelocityZ'))

    def test_load_plp_data(self):
        l_vars = pyspedas.cnofs.plp()
        self.assertTrue(data_exists('Ni'))

    def test_load_vefi_data(self):
        l_vars = pyspedas.cnofs.vefi()
        self.assertTrue(data_exists('E_meridional'))
        self.assertTrue(data_exists('E_zonal'))

    def test_load_notplot(self):
        c_vars = pyspedas.cnofs.cindi(notplot=True)
        self.assertTrue('ionVelocityX' in c_vars)

    def test_downloadonly(self):
        files = pyspedas.cnofs.cindi(downloadonly=True, trange=['2013-2-15', '2013-2-16'])
        self.assertTrue(os.path.exists(files[0]))


if __name__ == '__main__':
    unittest.main()
