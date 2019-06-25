import unittest
from AlgoritmiFunzionanti import LofExample
import numpy as np


class TestLOF(unittest.TestCase):

    def testDBFittizzio(self):
        '''array=np.array([1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1])'''
        array=[1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]
        self.assertListEqual(LofExample.DBFittizzio().tolist(), array)
