import numpy as np
import os
import re
import pdb
import bokeh
import bokeh.palettes as bp
from bokeh.plotting import figure, output_file, show

from scipy.optimize import curve_fit

#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#normalize an array to a float between 0 and 1. Easier for plotting.
#Input: 
def normalize(x):
    return (x-np.amin(x))/(np.amax(x)-np.amin(x))

def trim_tr(data, xmin = 1900, xmax = 2150):
    #wavenr_count = data.shape[0]

    low_index = np.amin(np.where(data[:,0]> xmin))  #np.where() gives the indices of those satifying the criterion. np.amin() gives the smallest index.
    hi_index = np.amax(np.where(data[:,0] < xmax))
 
    #pdb.set_trace()
    return data[low_index:hi_index, :]

def gaussian4(x, *params):
    
    #input :  x, an np.array holding the wavenumbers.
    #input : *params, an np.array of 4 * 3 gaussian params, like  [ctr_1, amp_1, wid_1,...]

    #output : y, an np.array holding the gaussian fitted values.

    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

def synthesize_gaussian4(x, params):

    #Why is it so similar to the previous function? I cannot recall

    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y



def plot_all_static(legendTrig = True):

    # A quick and dirty way to look at all the non-time-resolved data in the folder.

    #pdb.set_trace()
    #plt.figure()
    spectra_count = 0
    filenamelist = os.listdir()
    #handle_arr = []
    #filename_arr = []

    filename_trunk = os.getcwd().replace('\\','_')
    output_file(filename_trunk + '.html')

    TOOLTIPS = [("(x,y)", "($x, $y)"), "name: ", "$name"]

    p = figure(title = filename_trunk.replace('_', ', '), x_axis_label = 'Raman Shift', y_axis_label = 'Rel. Counts', tooltips = TOOLTIPS, plot_width = 1200, plot_height = 800)
    p.xaxis.axis_label_text_font_size = '16pt'
    p.xaxis.major_label_text_font_size = "16pt"
    p.yaxis.axis_label_text_font_size = '16pt'
    p.yaxis.major_label_text_font_size = "16pt"

    for filename in filenamelist:
        if re.search(".txt", filename) and (not re.search("Series", filename)) and (not re.search(".html", filename)):
            spectra_count += 1
            d = np.loadtxt(filename)
            #a = plt.plot(d[:,0],d[:,1])
            #handle_arr.append(a)
            #filename_arr.append(filename.replace('_',','))  
            X = d[:,0]
            Y = d[:,1]
            p.line(X, Y, color = bp.plasma(256)[(spectra_count*30)%145], muted_alpha=0.2, legend = filename, name = 'test')          

            #print(spectra_count)
    #plt.legend(filename_arr)

    p.legend.click_policy = "mute"
    p.legend.visible = legendTrig
    show(p)
    return

    
def plot_by_kw(keywords, leg = 1):

        # plot all data with all keywords in the keywords list.
        # keywords should be a list of strings instead of a single one

        #plt.figure()
        spectra_count = 0
        filenamelist = os.listdir()
        #handle_arr = []
        filename_arr = []

        filename_trunk = ''
        for j in keywords:
            filename_trunk = filename_trunk + j + ','
        
        output_file(filename_trunk + '.html')

        TOOLTIPS = [("(x,y)", "($x, $y)"),]

        p = figure(title = filename_trunk.replace('_', ', '), x_axis_label = 'Raman Shift', y_axis_label = 'Rel. Counts', tooltips = TOOLTIPS, plot_width = 1200, plot_height = 800)
        p.xaxis.axis_label_text_font_size = '16pt'
        p.xaxis.major_label_text_font_size = "16pt"
        p.yaxis.axis_label_text_font_size = '16pt'
        p.yaxis.major_label_text_font_size = "16pt"
        offset = 0.7


        for filename in filenamelist:
            search_res = True
            for kw in keywords:
                
                if re.match("-", kw):
                    search_res = search_res and (re.search(kw.replace('-',''), filename) == None) and (re.search(".html", filename) == None) and (re.search("Series", filename) == None)
                else:
                    search_res = search_res and (re.search(kw, filename) != None) and (re.search(".html", filename) == None) and (re.search("Series", filename) == None)

            #pdb.set_trace()
                
            if((re.search(".txt", filename) != None ) and search_res):
                spectra_count += 1
                d = np.loadtxt(filename)
                X = d[:,0]
                Y = normalize(d[:,1])

                p.line(X, Y+spectra_count*offset, color = bp.plasma(256)[(spectra_count*30)%145], legend = filename)
                #a = plt.plot(d[:,0],d[:,1])
                #handle_arr.append(a)
                #filename_arr.append(filename.replace('_',','))            

            #print(spectra_count)
        #if leg == 1:
            #plt.legend(filename_arr)
        p.legend.click_policy = "mute"
        p.legend.visible = True
        show(p)
        return


def plot_by_fn(filenames, leg = 0, norm = 1, offset = 0.7):

    
    spectra_count = 0

    filename_trunk = filenames[0]+str(np.random.random_integers(32767))
    output_file(filename_trunk + '.html')

    TOOLTIPS = [("(x,y)", "($x, $y)"),]

    p = figure(title = filename_trunk.replace('_', ', '), x_axis_label = 'Raman Shift', y_axis_label = 'Rel. Counts', tooltips = TOOLTIPS, plot_width = 1200, plot_height = 800)
    p.xaxis.axis_label_text_font_size = '16pt'
    p.xaxis.major_label_text_font_size = "16pt"
    p.yaxis.axis_label_text_font_size = '16pt'
    p.yaxis.major_label_text_font_size = "16pt"

    for filename in filenames:
        d = np.loadtxt(filename)
        if norm == 0:
            offset = np.amax(d[:,1])-np.amin(d[:,1])
            p.line(d[:,0], d[:,1]+spectra_count*offset, color = bp.plasma(256)[(spectra_count*30)%145], legend = filename)
        else:
            p.line(d[:,0], normalize(d[:,1])+spectra_count*offset, color = bp.plasma(256)[(spectra_count*30)%145], legend = filename)
        spectra_count += 1

    if leg == 1:
        p.legend.visible = True
    else:
        p.legend.visible = False


    show(p)
    return


def sum_by_kw(keywords):

    filenamelist = os.listdir()
    ret = np.zeros((3779,2))

    for filename in filenamelist:
        search_res = True
        for kw in keywords:
            
            if re.match("-", kw):
                search_res = search_res and (re.search(kw.replace('-',''), filename) == None)
            else:
                search_res = search_res and (re.search(kw, filename) != None)

        #pdb.set_trace()
            
        if((re.search(".txt", filename) != None ) and search_res):
            d = np.loadtxt(filename)
            ret[:,0] = d[:,0]
            ret[:,1] = ret[:,1] + d[:,1]

    return ret

def lint_comma(filename):
    with open(filename+'.txt', 'r') as incoming:
        with open(filename+'dot.txt','w') as outcoming:
                for line in incoming.readlines():
                    #pdb.set_trace()
                    outcoming.writelines(line.replace(',','.'))

    return

#def plot_Charlotte(filename):

def lint_tr(filename):
    d = np.loadtxt(filename)
    dshape = d.shape

    timeindex = []
    for i in range(dshape[0]):
        if d[i,0] not in timeindex:
            timeindex.append(d[i,0])

    spectracnt = len(timeindex)
    spectralength = int(dshape[0]/spectracnt)
    #pdb.set_trace()
    ret = np.zeros((spectralength,spectracnt+1))
    #pdb.set_trace()
    ret[:,0] = d[0:spectralength,1]

    for k in range(spectracnt):
        ret[:,k+1] = d[k*spectralength:(k+1)*spectralength,2]

    np.savetxt(('Reshape_'+filename),ret)
    return ret

def batch_lint_tr():
    filenamelist = os.listdir()

    for filename in filenamelist:
        if re.search('.txt', filename) and not re.search("Mapping", filename) and not re.search("Reshape", filename) and re.search('Series', filename):
            lint_tr(filename)

def plot_tr(filename, norm = False, legend = True, interval = 1):
    filename_trunk = filename.replace('.txt','')
    d = np.loadtxt(filename)
    spectra_count = d.shape[1] - 2
    #spectra_width = d.shape[0]
    X = d[:,0]

    output_file(filename_trunk + '.html')

    TOOLTIPS = [
    ("(x,y)", "($x, $y)"),("Nr: ", "$name")]

    p = figure(title = filename_trunk.replace('_', ', '), x_axis_label = 'Raman Shift', y_axis_label = 'Rel. Counts', tooltips = TOOLTIPS, plot_width = 600, plot_height = 900)
    p.xaxis.axis_label_text_font_size = '16pt'
    p.xaxis.major_label_text_font_size = "16pt"
    p.yaxis.axis_label_text_font_size = '16pt'
    p.yaxis.major_label_text_font_size = "16pt"
    offset = 0

    for i in range(spectra_count):
        if norm == True:
            Y = normalize(d[:,i+1])
            p.line(X, Y+i-1, color = bp.plasma(256)[(i*3)%145],  muted_alpha=0.2, legend_label = str(interval*i) + 'sec', name = 'test')
        else:
            Y = d[:,i+1] + offset
            offset += 1.5*(np.amax(d[:,i+1])-np.amin(d[:,i+1]))
            p.line(X, Y, color = bp.plasma(256)[(i*3)%145],  muted_alpha=0.2, legend_label = str(interval*i) + 'sec', name = str(i+1))



    p.legend.click_policy = "mute"
    p.legend.visible = legend

    show(p)

def plot_CV_tr(filename, norm = False, legend = False,  positive_scan = True):
    filename_trunk = filename.replace('.txt','')
    d = np.loadtxt(filename)
    spectra_count = d.shape[1] - 2
    #spectra_width = d.shape[0]
    X = d[:,0]

    output_file(filename_trunk + '.html')

    TOOLTIPS = [
    ("(x,y)", "($x, $y)"),("Nr and bias: ", "$name")]

    p = figure(title = filename_trunk.replace('_', ', '), x_axis_label = 'Raman Shift', y_axis_label = 'Rel. Counts', tooltips = TOOLTIPS, plot_width = 600, plot_height = 900)
    p.xaxis.axis_label_text_font_size = '16pt'
    p.xaxis.major_label_text_font_size = "16pt"
    p.yaxis.axis_label_text_font_size = '16pt'
    p.yaxis.major_label_text_font_size = "16pt"
    offset = 0

    for i in range(spectra_count):
        if norm == True:
            Y = normalize(d[:,i+1])
            p.line(X, Y+i-1, color = bp.plasma(150)[(i)%50],  muted_alpha=0.2, name = ''.join((str(i+1), ', ', "{:.3f}".format(conv_time_to_bias(i, positive_scan=positive_scan)), 'V')))
        else:
            Y = d[:,i+1] + offset
            offset += 1.5*(np.amax(d[:,i+1])-np.amin(d[:,i+1]))
            p.line(X, Y, color = bp.plasma(150)[(i)%50],  muted_alpha=0.2, name = ''.join((str(i+1), ', ', "{:.3f}".format(conv_time_to_bias(i, positive_scan=positive_scan)), 'V')))



    #p.legend.click_policy = "mute"
    #p.legend.visible = legend

    show(p)

def batch_plot_tr(legend = False):

    filenamelist = os.listdir()
    for filename in filenamelist:
        kw_list = filename.replace('.txt', '').split('_')
        #pdb.set_trace()
        if ((re.search('.txt', filename)) and ('Reshape' in kw_list)):
            for kw in kw_list:
                if re.search('sec', kw) and not re.search('V', kw):
                    interval = float(kw.replace('sec',''))
                    break
                elif re.search('Sec', kw):
                    interval = float(kw.replace('Sec',''))
                    break
                else:
                    interval = 5
            plot_tr(filename, legend=legend, interval=interval)


def plot_Autolab_CV(filename, offset = 0):

    filename_trunk = filename.replace('.txt','')
    d = np.loadtxt(filename)
    output_file(filename_trunk + '.html')

    TOOLTIPS = [
    ("(x,y)", "($x, $y)"),]

    X = d[:,0] + offset
    Y = d[:,1] * 1000

    p = figure(title = filename_trunk.replace('_', ', '), x_axis_label = 'E vs. RHE', y_axis_label = 'Current / mA', tooltips = TOOLTIPS, plot_width = 900, plot_height = 600)
    p.xaxis.axis_label_text_font_size = '16pt'
    p.xaxis.major_label_text_font_size = "16pt"
    p.yaxis.axis_label_text_font_size = '16pt'
    p.yaxis.major_label_text_font_size = "16pt"

    p.line(X, Y)

    show(p)


def remove_ladders(filename, delimiter = ',', write_new = False):
    d = np.loadtxt(filename, delimiter = delimiter)
    spectra_count = d.shape[1] - 1
    #spectra_width = d.shape[0]
    X = d[:,0]

    output_filename = filename.replace('.csv', '') + '_Flat.csv'

    ret_flat = np.zeros(d.shape)
    ret_flat[:,0] = X

    for i in range(spectra_count):
        p0 = np.mean( d[0:4, i+1] )
        p1 = np.mean( d[-5:-1, i+1] )
        incline = p1 - p0
        X0 = X[0]
        X_Max = X[-1] 
        
        baseline = (X - X0) * (incline / (X_Max - X0))
        Y = d[:,i+1] - baseline - p0
        ret_flat[:,i+1] = Y

    if write_new == True:
        np.savetxt(output_filename, ret_flat, delimiter = ',')

    return ret_flat, output_filename

def heatmap_tr( filename, time_interval = 0.717, xmin=1900, xmax=2150, startOffset = 0, VCompress = 2, Center = 2050):
    #flat_array = remove_ladders(filename)
    flat_array = np.loadtxt(filename, delimiter=',')
    Xbar = time_interval * np.arange(flat_array.shape[1] - 1) + startOffset
    
    #X, Y = np.meshgrid(Xbar, Ybar)

    if Center == 1050:
        xmin = 900
        xmax = 1200

    if Center == 500:
        xmin = 400
        xmax = 700

    data = trim_tr( flat_array, xmin, xmax )
    Ybar = data[:,0]
    data = data[:,1:]

    #noise removal, brutal
    noise  = 0.8*np.amax(data[0:7,:])
    data  = data - noise
    data = np.where(data<0, 0, data)
    
    '''
    legacy bokeh plot
    output_filename = filename.replace('.csv', '') + '.html'
    output_file(output_filename, title="image.py example")

    p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
    p.x_range.range_padding = p.y_range.range_padding = 0

    p.image(image=[data], x=0, y=0, dw=10, dh=10, palette="Spectral11", level="image")
    p.grid.grid_line_width = 0.5

    show(p)
    '''

    '''
    fig = plt.figure()
    
    ax = fig.gca(projection='3d')
    ax.view_init(elev = 90, azim = 0)
    #pdb.set_trace()
    

    surf = ax.plot_surface(X, Y, data, rcount = 100, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlim(np.amin(Xbar), np.amax(Xbar))
    plt.ylim(1900, 2150)

    

    plt.show()
    '''
    #pdb.set_trace()
    
    fig = plt.figure()
    ax1 = fig.gca()
    im = ax1.imshow(data, interpolation = 'bilinear', extent = [np.amin(Xbar), np.amax(Xbar), np.amax(Ybar), np.amin(Ybar)], aspect=0.03 * VCompress)
    #
    ax1.set_xticks([0,3,6,9,12,15,18])

    fig.colorbar(im,ax=ax1)
    plt.title(filename)

def fit_flat_tr(filetag, param_fromFile = False, p0 = [2090,650,50,2050,200,50,2015,100,50,2010,20,50], bounds=([2084,20,5, 2000,30,5, 2010,10,5, 1950, 10,5],[2100, 10000, 100, 2076, 10000, 150, 2028, 400, 70, 2080, 40, 70])):
    filename = filetag + '_Flat.csv'
    flat_array = np.loadtxt(filename, delimiter=',')
    X = flat_array[:,0]
    spectra_cnt = flat_array.shape[1] - 1
    #pdb.set_trace()
    params_arr = np.zeros( (12, spectra_cnt) )
    #pdb.set_trace()

    if param_fromFile is True:
        init_params = np.loadtxt(filetag + '_init.csv', delimiter = ',')
        p0 = init_params[0,:]
        bounds = (init_params[1,:], init_params[2,:])

    for i in range(0, spectra_cnt): #The first spectrum is not fitted! (perhaps because it's usually blank)
        #plt.figure(filename + ', ' + str(1.2 * i) + 'sec' )
        Y = flat_array[:,i+1]
        

        popt = fit_4gaussian_single(X, Y, title = filetag + '_' + str(0.717 * i) + 'sec', filetag = filetag, p0 = p0, bounds = bounds)

        params_arr[:, i-1] = popt

    plot_peak_position(params_arr)   
    plot_peak_intensity(params_arr)
    np.savetxt(filetag + '_params.csv', params_arr, delimiter = ',')

    return params_arr

def generate_4gaussian_values(filetag):
    #   Input: a filetag. Raw data and params are inferred from the tag.
    #   Output: filetag + '_GaussianValues.csv', 1st column is the X, 2nd column is unfitted 1st spectrum, then data-> HFB -> LFB -> Bridge.
    tr_data  = np.loadtxt(filetag + '_Flat.csv', delimiter=',')
    data_row_cnt = tr_data.shape[0]
    data_col_cnt = tr_data.shape[1] - 1 #   Fist spectrum is not fitted! and -1 for X
    params = np.loadtxt(filetag + '_params.csv', delimiter=',')

    ret_mat = np.zeros( (data_row_cnt, data_col_cnt*4 + 1) )   #   +1 for X, +1 for the unfitted first spectrum
    X = tr_data[:,0]
    ret_mat[:,0] = X
    #ret_mat[:,1] = tr_data[:,1]

    for i in range(data_col_cnt):
        ret_mat[:,i*4+1] = tr_data[:,i+1]
        for j in range(3):
            ctr = params[j*3,i]
            amp = params[j*3+1, i]
            wid = params[j*3+2, i]
            ret_mat[:,i*4+j+2] = amp * np.exp( -((X - ctr)/wid)**2)

    np.savetxt(filetag + '_GaussianValues.csv', ret_mat, delimiter=',')
    return ret_mat


    


def fit_4gaussian_single(X, Y, title, filetag, p0, bounds, autosave = True, legend_list = ('HFB', 'LFB', 'Bridge', 'Bracer')):
    
    path = './' + filetag + '_fit/'
    plt.figure(title )
    plt.plot(X,Y)

    popt, pcov = curve_fit(gaussian4, X, Y, p0, bounds = bounds )
    fit = synthesize_gaussian4( X, popt)
    plt.plot(X, fit)

    for i in range(4):
        ctr = popt[i*3]
        amp = popt[i*3+1]
        wid = popt[i*3+2]
        ith_fit = amp * np.exp( -((X - ctr)/wid)**2)
        plt.plot(X, ith_fit, label = legend_list[i])

    plt.legend()
    if autosave == True:
        plt.savefig(path + title +'.png')
    plt.close()
    return popt

def fit_single_GenerateValues(filetag):
    single_data  = np.loadtxt(filetag + '_Flat.csv', delimiter=',')
    init_params = np.loadtxt(filetag + '_init.csv', delimiter = ',')
    p0 = init_params[0,:]
    bounds = (init_params[1,:], init_params[2,:])

    ret_mat = np.zeros( (single_data.shape[0], 6) )

    X = single_data[:,0]
    ret_mat[:,0] = X
    Y = single_data[:,1]
    ret_mat[:,1] = Y
    title = filetag + ', stabilized'

    params = fit_4gaussian_single (X=X, Y=Y, title=title, filetag=filetag, p0=p0, bounds=bounds, autosave= False, legend_list=('HFB_1', 'HFB_2', 'LFB', 'Bridge'))

    for j in range(4):
            ctr = params[j*3]
            amp = params[j*3+1]
            wid = params[j*3+2]
            ret_mat[:,j+2] = amp * np.exp( -((X - ctr)/wid)**2)

    np.savetxt(filetag + '_GaussianValues.csv', ret_mat, delimiter=',')
    np.savetxt(filetag+'_Params.csv', params, delimiter = ',')

    return params




def plot_peak_position(params_arr):

    plt.figure('peak positions')
    legend_list = ('HFB', 'LFB', 'Bridge', 'Bracer')
    for i in range(0,4):
        plt.plot(params_arr[i*3,:], label = legend_list[i])
        plt.legend()    

    return

def plot_peak_intensity(params_arr):
    plt.figure('peak intensities')
    legend_list = ('HFB', 'LFB', 'Bridge', 'Bracer')
    for i in range(0,4):
        plt.plot(np.array(params_arr[i*3+1,:]), label = legend_list[i])
        plt.legend()    

    return    

def trim_tr_byLine(filename, startline, stopline):
    d = np.loadtxt(filename)
    X = d[:,0]
    d_trimmed = d[:, startline:stopline+1]
    d_trimmed[:,0] = X

    ret_filename = filename.replace('.txt', '') + '_' + str(startline) + '_to_' + str(stopline) + '.csv'
    np.savetxt(ret_filename, d_trimmed,delimiter=',')

    return d_trimmed, ret_filename

def heatmap_tr_byLine(filename, startline, stopline, time_interval = 0.717, VCompress = 2, Center = 2050, startOffset = 0):
    M1, F1 = trim_tr_byLine(filename, startline, stopline)
    M2, F2 = remove_ladders(F1, write_new=True)
    heatmap_tr(F2, time_interval = time_interval, startOffset = startOffset, VCompress = VCompress, Center=Center)


def conv_time_to_bias(spectra_nr, init_bias = -0.07, spd = 0.05, low_bias = -1.6, hi_bias = 1.0, interval = 1.227, offset=15, positive_scan = True):
    
    net_spectra_nr = spectra_nr - offset
    
    if net_spectra_nr <= 0:
        return init_bias

    cycle_spec_cnt = (hi_bias - low_bias)*2 / (spd * interval)
    spectra_nr_residue = net_spectra_nr % cycle_spec_cnt

    tick_1 = (hi_bias*(positive_scan) + init_bias * 2 * (0.5 - positive_scan) - low_bias*(not positive_scan)) / (spd*interval)
    tick_2 = tick_1 + (hi_bias - low_bias) / (spd * interval) 

    if spectra_nr_residue < tick_1:
        bias = init_bias + (spectra_nr_residue) * spd * interval * 2 * (positive_scan-0.5)
    elif spectra_nr_residue < tick_2:
        bias = low_bias * (not positive_scan) + hi_bias*(positive_scan) + (spectra_nr_residue - tick_1) * spd * interval * 2 * (0.5-positive_scan)
    else:
        bias = hi_bias * (not positive_scan) + low_bias * (positive_scan) + (spectra_nr_residue - tick_2) * spd * interval * 2 * (positive_scan - 0.5)

    return bias





    
















