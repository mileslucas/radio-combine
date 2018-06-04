import unittest

import test_compare

loader = unittest.TestLoader()
suite = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(test_compare))

runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)
