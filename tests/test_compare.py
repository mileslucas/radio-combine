import unittest
from scripts import compare

class TestCompare(unittest.Testcase):

	def test_test(self):
		self.assertTrue(True)
	
	def test_get_arr(self):
		data_path = 'data/orion.gbt.im'
		arr = compare.get_arr(data_path)
		self.assertEqual(arr.shape, (300,300))	


if __name__=='__main__':
	unittest.main()	

