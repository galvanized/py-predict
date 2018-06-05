import unittest
from main import *
from models import *

class ReferenceModelReturnsInput(SingleOutputModel):
    def eval(self, x):
        return x

class ReferenceModelReturnsOne(SingleOutputModel):
    def eval(self, x):
        return 1

class ReferenceModel1dOnly(SingleOutputModel):
    def __init__(self):
        self.x_shape = (-1, )

    def eval(self, x):
        return 1

class ReferenceModelFixedShape(SingleOutputModel):
    def __init__(self):
        self.x_shape = (2,2)

    def eval(self, x):
        return 1

def error_to_2by5(params):
    '''
    takes a 2-tuple as argument
    '''
    err = abs(2-params[0]) + abs(5-params[1])
    return(err)

class TestReferenceModels(unittest.TestCase):
    def test_returns_input(self):
        ri = ReferenceModelReturnsInput()
        self.assertEqual(
            [ri.run(x) for x in range(-20,20)],
            list(range(-20,20))
        )

    def test_returns_1(self):
        r1 = ReferenceModelReturnsOne()
        self.assertEqual(
            [r1.run(x) for x in range(-20,20)],
            [1]*40
        )

    def test_1d_only(self):
        rm = ReferenceModel1dOnly()
        pass1 = True
        try:
            rm.run([1,2,3])
        except InvalidInputDimensions:
            pass1 = False
        self.assertTrue(pass1)

        pass2 = False
        try:
            rm.run([[3,4,2],[1,2,3]])
        except InvalidInputDimensions:
            pass2 = True
        self.assertTrue(pass2)

        pass3 = False
        try:
            rm.run(1)
        except InvalidInputDimensions:
            pass3 = True
        self.assertTrue(pass3)

    def test_fixed_shape(self):
        rm = ReferenceModelFixedShape()
        pass1 = True
        try:
            rm.run([[1,2],[3,4]])
        except InvalidInputDimensions:
            pass1 = False
        self.assertTrue(pass1)

        pass1 = False
        try:
            rm.run([[1,2],[3,4],[5,6]])
        except InvalidInputDimensions:
            pass1 = True
        self.assertTrue(pass1)

    def test_calculate_error(self):
        rm = ReferenceModelReturnsOne()
        o = rm.calculate_one_one_error([1,2,3],[0,1,1])
        self.assertEqual(o,
        {'c': 0.8660254037844385, 'cedb': -8.73, 'e': 0, 'er': 0.0,
         'erp': 0.0, 'fn': 1, 'fnr': 0.3333333333333333, 'fnrp': 33.333,
         'fp': 0, 'fpr': 0.0, 'fprp': 0.0, 'l': 3, 'lr': 1.0, 'lrp': 100.0,
         'q': 3, 'rms': 1.4142135623730951, 'rmsp': 141.421, 'tn': 0,
         'tnr': 0.0, 'tnrp': 0.0, 'tp': 2, 'tpr': 0.6666666666666666, 'tprp': 66.667})

class TestModels(unittest.TestCase):
    def test_sago(self):
        target = (2, 5)
        errf = error_to_2by5
        params = sago(errf, [0,0], param_mags=[10,10], pows=10, iters=100, reps=10)
        self.assertTrue(round(params[0],3)==round(target[0],3))
        self.assertTrue(round(params[1],3)==round(target[1],3))

class TestDatabase(unittest.TestCase):
    def test_goog(self):
        # create new db in memory
        # get from yahoo
        # convert to Stock
        # check certain index
        # ensure close matches
        pass


if __name__ == '__main__':
    unittest.main()
