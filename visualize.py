from matplotlib import use
use('tkagg') #use AGG png render
import matplotlib.pyplot as plt
from stocks import Database
from analysis import *
import numpy as np
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import dataset_gen

def rgba(hexc, a=1):
    '''
    Takes a 3 channel hex color, outputs a normalized 4-tuple with full alpha.
    '''
    out = [0,0,0,255]
    out = []
    for i in range(3):
        v = hexc % 0x100
        hexc -= v
        hexc = hexc//0x100
        out.append(v/0xFF)

    return tuple(out[::-1] + [a])

def graph_stock(symbol, title=None, fname=None, dtd=0):
    '''
    Save the graph of a stock and its derivative to a file.
    title: figure title
    fname: filename of output (full path)
    dtd: days to date
    '''
    db = Database('stockdata.sqlite')
    dat, dates = db.close_and_dates(symbol.upper())
    '''if dtd and len(dat) > dtd:
        dat = dat[-dtd:]
        dates = dates[-dtd:]'''
    if dtd > len(dat)-1:
        print("Not enough data. ({} !> {})".format(len(dat),dtd))
        return
    if not dtd:
        dtd = len(dat)-1 # one less due to deltas

    plt.close()
    plt.figure(1, figsize=(10,7))
    plt.subplot(211)
    if not title:
        plt.title(symbol)
    else:
        plt.title(title)
    plt.yscale('log')
    plt.grid(True)
    plt.grid(b=True, which='minor', color=(0.9,0.9,0.9,1), linestyle='--')
    plt.plot(range(dtd), sma(dat,15)[-dtd:], color=rgba(0xefbf00,0.8), label="15d SMA")
    plt.plot(range(dtd), sma(dat,50)[-dtd:], color=rgba(0x94df00,0.8), label="50d SMA")
    plt.plot(range(dtd), sma(dat,100)[-dtd:], color=rgba(0x047594), label="100d SMA")
    moving = ema(dat, 0.01)
    plt.plot(range(dtd), moving[-dtd:], color=rgba(0x3d0ea3), label="1% EMA")
    plt.plot(range(dtd), dat[-dtd:], 'k', label="Close price")
    plt.ylabel('Share Price')
    plt.legend(loc='best')


    plt.subplot(212)
    d = multipliers(dat)
    ds = wma(d, 90)
    dmoving = multipliers(moving)
    movingdmoving = ema(dmoving, 0.01)
    plt.yscale('log')
    plt.grid(True)
    axes = plt.gca()
    axes.set_ylim([0.99, 1.01])
    axes.yaxis.set_major_formatter(ScalarFormatter()) #ticklabel_format(useOffset=False, style='plain')
    plt.yticks([1+x/1000 for x in range(-10,11,2)])
    plt.ylabel('Daily Multiplicand')

    plt.plot(range(dtd), wma(d, 7)[-dtd:], color=(0.8,0.0,0.5,0.2), label="d/dt 7d WMA")
    plt.plot(range(dtd), ds[-dtd:], color=(0.0,0.0,0.8,0.4), label="d/dt 90d WMA")
    plt.plot(range(dtd), dmoving[-dtd:], 'r', label="EMA d/dt")
    plt.plot(range(dtd), movingdmoving[-dtd:], color=(0.0,0.8,0.0,0.9), label="EMA d/dt EMA")
    plt.plot(range(dtd), [1]*(dtd), color=(0,0,0,0.9))

    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if not fname:
        plt.savefig('graphs/{}.png'.format(symbol), dpi=200)
    else:
        plt.savefig(fname, dpi=200)
    print("Saved {}".format(symbol))


def graph_all(outdir='graphs/full'):
    db = Database('stockdata.sqlite')
    syms = db.list_symbols()
    del db

    for sym in syms:
        try:
            graph_stock(sym, fname='{}/{}-full.png'.format(outdir,sym))
        except:
            print("ERROR GRAPHING {}".format(sym))

def graph_all_ytd(outdir='graphs/ytd'):
    db = Database('stockdata.sqlite')
    syms = db.list_symbols()
    del db

    for sym in syms:
        #try:
        graph_stock(sym, title='{} YTD'.format(sym), fname='{}/{}-ytd.png'.format(outdir,sym), dtd=260)
        #except:
        #    print("ERROR GRAPHING {}".format(sym))

def boxhist_to_file(outfile, datalist, var_mult=2, bin_count=20):
    """
    Create a logarithmic, 1-centered combination boxplot-histogram and plots to file.

    Arguments
    outfile: file path to write file. directory structure must exist
    datalist: a list of values to be plotted. expeced to be centered near 1.
    var_mult: range multiplier. minimum for the plot = 1/var_mult, max = var_mult
    bin_count: number of bins for the histogram. logarithmically spaced.
    percentiles: list of percentiles to plot
    """

    hist, bins  = loghist(datalist, bin_count, var_mult)

    # workaround until Y scaling gets fixed
    hist_max = max(hist)
    hist = [x*1.4/hist_max for x in hist]

    plt.close()
    fig = plt.figure(1, figsize=(10,2))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xscale('log')
    #plt.xticks(rotation=90)

    plt.axis([1/var_mult, var_mult, 0, max(hist)*1.1])


    #ax.set_ylim([0,max(hist)*1.1])
    #ax.set_xlim([1/var_mult, var_mult])
    ax.xaxis.set_minor_formatter(FormatStrFormatter("")) #"%.2f"
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))


    ax.yaxis.set_major_formatter(FormatStrFormatter(""))
    ax.yaxis.set_minor_formatter(FormatStrFormatter(""))
    ax.yaxis.set_tick_params(size=0)
    ax.yaxis.tick_left()

    for i in range(len(hist)):
        b = plt.bar(bins[i],hist[i], bins[i+1]-bins[i], align='edge', color=(0,0,0,0.2), linewidth=0)

    ax.set_xticks([1,1/var_mult, var_mult])
    ax.set_xticks(bins, minor=True)
    #ax.set_xticks([-0.499,1,1.999])
    ax.grid(which='x')

    bp = ax.boxplot(datalist, vert=False, widths=[0.75], whis='range', notch=True, bootstrap=2000)
    for element in bp['boxes'] + bp['caps'] + bp['medians']:
            element.set(linewidth=3)

    #fig.autofmt_xdate()

    #plt.grid(True)

    #plt.grid(b=True, which='minor', color=(0.9,0.9,0.9,1), linestyle='--')
    #plt.xticks(rotation=90)
    #plt.setp( ax.xaxis.get_majorticklabels(), rotation=70 )
    #fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    #ax.tick_params(axis='x', labelrotation=90)

    plt.savefig(outfile)

def html_deviance_report(outfile='report.html', ranges=[7,30,90,365], input_length=600, syms=None):
    htmlout = '<table><tr><th>Symbol</th>'
    htmlout += ''.join(['<th>'+str(x)+'d'+'</th>' for x in ranges])
    htmlout += '</tr>'

    db = Database('stockdata.sqlite')
    if not syms:
        syms = db.list_symbols()
        syms.sort()

    for sym in syms:
        print(sym)
        htmlout += '<tr><th>{}</th>'.format(sym)
        for r in ranges:
            filename = 'graphs/report/{}{}.png'.format(sym, r)
            datalist = db.n_day_changes(sym, r, input_length)
            if datalist:
                try:
                    boxhist_to_file(filename, datalist, bin_count=50)
                    htmlout += '<th><img src={}></th>'.format(filename)
                except:
                    htmlout += '<th>ERR</th>'
            else:
                htmlout += '<th>NED</th>'
        htmlout += '</tr>'

    with open(outfile, 'w') as of:
        of.write(htmlout)

def plot_error(sym,incl_len=None, step=1, in_len = 100, f_len = 10):
    print('Getting data for {}.'.format(sym))
    d = dataset_gen.return_single(sym, incl_len=incl_len, step=step,
                                  in_len = in_len, f_len = f_len)

    print('Reshaping.')
    xs = []
    ys = []
    for pt in d:
        xs.append(pt[0][0])
        ys.append(pt[1][0])

    print('Reconstructing.')
    # build the true graph from each normalized section
    reconstructed = [1]
    for x in xs:
        coeff = reconstructed[-1]/x[0]
        addtl = x[1:1+step]
        for a in addtl:
            if a is not 0:
                reconstructed.append(a*coeff)
            else:
                reconstructed.append(reconstructed[-1])

    plt.close()
    plt.figure(1, figsize=(10,7))
    plt.subplot(211)
    plt.title(sym)
    plt.yscale('linear')
    plt.grid(True)
    plt.grid(b=True, which='minor', color=(0.9,0.9,0.9,1), linestyle='--')

    for xi in range(len(xs)):
        if xi%5==0:
            #plt.plot([1]*step*xi+xs[xi])
            print(xs[xi])

    '''for yi in range(len(ys)):
        index = step*(yi+in_len)
        pts = ys[yi]
        newpts = []
        for p in pts:
            newpts.append(p-5)
        plt.plot([1]*index+newpts)'''

    plt.plot(reconstructed)

    plt.show()


    print(reconstructed)








if __name__ == '__main__':
    '''
    own=['MSFT','NKE','BAC','ABBV','RF','BLUE','BGNE','FOLD','EC','CBOE','ATVI']
    watch=['FB','BABA','BAC','VKTX','NRZ','PSB','GAIN','WRD','NEWT','RGA','AET',
           'SHOP','TRU','ALGN','BLKB','CCMP','FIZZ','GDI','GMED','INGN','IPGP',
           'PYPL','BBD','FLN','FHK','COMT','MTBC','SQ','ANET','ZNH','BA']
    combo = own+watch
    combo.sort()
    html_deviance_report(outfile='watch.htm',ranges=[30,90],syms=combo)
    '''
    plot_error('GOOG',300,10,10,5)
