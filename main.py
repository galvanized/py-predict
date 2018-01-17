import numpy as np
from matplotlib import use
use('tkagg') # select pyplot backend
import matplotlib.pyplot as plt
import math

class SingleOutputModel():
    def __init__(self):
        self.x_shape = None
        self.name = 'model'

    def check_x_shape(self, x):
        if self.x_shape != None:
            t = repr(type(x))
            if t!="<class 'numpy.ndarray'>" and t!="<class 'numpy.core.memmap.memmap'>":
                x = np.array(x)

            if len(x.shape) != len(self.x_shape):
                raise InvalidInputDimensions("Mismatch between {} input and {} expected".format(
                    x.shape, self.x_shape
                ))

            for ex, rx in zip(self.x_shape, x.shape):
                if ex == -1:
                    continue
                else:
                    if ex != rx:
                        raise InvalidInputDimensions("Mismatch between {} input and {} expected".format(
                            x.shape, self.x_shape
                        ))

    def eval(self, x):
        pass
        # this function is defined by the model instance

    def run(self, x):
        self.check_x_shape(x)
        return self.eval(x)

    def run_multi(self, xs):
        return [self.run(x) for x in xs]

    def analysis(self, xs, ys, graph_path=None, csv_path=None):
        pass

    def calculate_one_one_error(self, actual_list, predicted_list):
        '''
        Compare both to (1,1)
        Return dictionary with:
        rms, tp, tn, fp, fn, e, l, q,
        tpr, tnr, fpr, fnr, er, lr,
        tprp, tprp, fprp, fnrp, erp, lrp,
        (percentages rounded to 3 places)
        c, cedb


        with x:actual, y:predicted
                    |
          false     |  true
          positive  |  positive
        ------------+-------------
          true      |  false
          negative  |  negative
                    |


                            /
                excessive /
                        /
                      /
                    /
                  /   lacking
                /
        '''
        rms = 0 # root mean squared error

        tp = 0 # true positive
        tn = 0 # true negative
        fp = 0 # false positive
        fn = 0 # false negative

        e = 0 # excessive (overestimate)
        l = 0 # lacking (underestimate)

        q = 0 # quantity

        for z in zip(actual_list, predicted_list):
            if z[0] >= 1:
                # actual value is positive
                if z[1] >= 1:
                    # predicted is positive.
                    # true positive
                    tp += 1
                else:
                    # predicted is negative
                    # false negative
                    fn += 1
            else:
                # actual value is negative
                if z[1] >= 1:
                    # predicted is positive
                    # false positive
                    fp += 1
                else:
                    # predicted is negative
                    # true negative
                    tn += 1

            if z[1] > z[0]:
                # predicted is too high
                # excessive
                e += 1
            elif z[1] < z[0]:
                # predicted is too low
                # lacking
                l += 1

            q += 1

            rms += (z[0] - z[1])**2

        if q == 0:
            print("No values for error calculation!")
            q = 1e-6

        rms = (rms / q)**0.5
        # compute rates
        tpr = tp / q
        tnr = tn / q
        fpr = fp / q
        fnr = fn / q
        er = e / q
        lr = l / q

        # compute rate percentages
        tprp = round(tpr * 100, 3)
        tnrp = round(tnr * 100, 3)
        fprp = round(fpr * 100, 3)
        fnrp = round(fnr * 100, 3)
        erp = round(er * 100, 3)
        lrp = round(lr * 100, 3)
        rmsp = round(rms * 100, 3)

        c = np.corrcoef(actual_list, predicted_list)[0, 1]
        cedb = round(10 * math.log(1-c, 10), 3) # correlation error, decibels

        return {'rms':rms,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'e':e,'l':l,'q':q,
                'tpr':tpr,'tnr':tnr,'fpr':fpr,'fnr':fnr,'er':er,'lr':lr,
                'tprp':tprp,'tnrp':tnrp,'fprp':fprp,'fnrp':fnrp,'erp':erp,'lrp':lrp, 'rmsp':rmsp,
                'c':c,'cedb':cedb}

    def graph_one_one_error(self, xs, ys, out_path, title=None):
        '''
        x is real values
        y is predicted values
        '''
        x_vals = xs
        y_vals = self.run_multi(xs)

        alpha = max(0.05, min(1, 100/(len(x_vals)+1)))

        plt.close()
        plt.figure(num=1, figsize=(5, 5), dpi=300)

        plt.scatter(x_vals, y_vals, s=5, c='r', linewidths=0, alpha=alpha)
        plt.plot([.5,2],[.5,2], 'b', alpha=0.5)
        plt.plot([1,1],[0.95,1.05],'b')
        plt.plot([0.95,1.05],[1,1],'b')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True)
        plt.axis([0.1,10,0.1,10])
        plt.xlabel('Real')
        plt.ylabel('Predicted')


        e = self.calculate_one_one_error(x_vals, y_vals)

        lh = 3.1622777 # logarithmic half, 10^(1/2)

        textalpha = 0.5

        plt.text(lh, lh, 'True Positive:\n{}%'.format(e['tprp']), ha='center', alpha=textalpha)
        plt.text(1/lh, lh, 'False Positive:\n{}%'.format(e['fprp']), ha='center', alpha=textalpha)
        plt.text(lh, 1/lh, 'False Negative:\n{}%'.format(e['fnrp']), ha='center', alpha=textalpha)
        plt.text(1/lh, 1/lh, 'True Negative:\n{}%'.format(e['tnrp']), ha='center', alpha=textalpha)

        plt.text(1, 7, 'Overestimate: {}%'.format(e['erp']), ha='center', alpha=textalpha)
        plt.text(1, 0.13, 'Underestimate: {}%'.format(e['lrp']), ha='center', alpha=textalpha)

        if title == None:
            title = self.name

        if title:
            title += '\n'

        title += 'RMS={}% CatErr={}% cedB={}'.format(e['rmsp'],round(e['fprp']+e['fnrp'],3),e['cedb'])

        plt.title(title)
        plt.savefig(out_path)

class InvalidInputDimensions(Exception):
    pass
