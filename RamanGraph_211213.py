import numpy as np
import pandas as pd
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

#import RamanBatchPlot as RBP

def remove_straight_baseline( input_X, input_Y ):

    p0 = np.mean( input_Y[0:4] )
    p1 = np.mean( input_Y[-5:-1] )
    incline = p1 - p0
    X0 = input_X[0]
    X_Max = input_X[-1] 
        
    baseline = (input_X - X0) * (incline / (X_Max - X0))
    ret_Y = input_Y - baseline - p0

    return ret_Y




def txt_to_df( input_filename, baselinecorr = False ):

    df = pd.read_csv(input_filename, sep = '\t', names = ['X', 'Y', 'Raman Shift', 'Counts'])
    
    X_set = set(df['X'])
    X_cnt = len(X_set)
    X_list = list(X_set)
    X_interval = X_list[1] - X_list[0]

    Y_set = set(df['Y'])
    Y_cnt = len(Y_set)
    Y_list = list(Y_set)
    Y_interval = Y_list[1] - Y_list[0]

    #Shift_set = set(df['Raman Shift'])
    Shift_cnt = len(df[(df['X'] == X_list[0]) & (df['Y']==Y_list[0])])
    Shift = df['Raman Shift'][0:Shift_cnt]

    ret_array = np.zeros((Shift_cnt, (X_cnt * Y_cnt + 1)))

    ret_array[:,0] = np.array(Shift)

    for i in range(X_cnt):
        for j in range(Y_cnt):
            Y_current = df.iloc[(i*Y_cnt + j)*Shift_cnt:(i*Y_cnt + j+1)*Shift_cnt,3]

            if baselinecorr == True:
                ret_array[:, (i*Y_cnt + j + 1)] = remove_straight_baseline(np.array(Shift), Y_current)
            else:
                ret_array[:, (i*Y_cnt + j + 1)] = Y_current

    ret_columns = ['Wavenumber']

    for i in range(X_cnt):
        for j in range(Y_cnt):
            ret_columns.append(str(X_list[i] * X_interval) + ',' + str(Y_list[j] * Y_interval))

    ret_df = pd.DataFrame(ret_array, columns=ret_columns)

    return ret_df, X_list, Y_list

def df_to_waterfall_Bokeh(input_df, X_list, Y_list, input_title):

    output_file(input_title + '.html')
    X = input_df['Wavenumber']

    TOOLTIPS = [("(wavenumber, height)", "($x, $y)"),("coord", "$name")]

    p1 = figure(title = input_title, x_axis_label = 'Raman Shift', y_axis_label = 'Rel. Counts', tooltips = TOOLTIPS, plot_width = 600, plot_height = 900)


    p1.xaxis.axis_label_text_font_size = '16pt'
    p1.xaxis.major_label_text_font_size = "16pt"
    p1.yaxis.axis_label_text_font_size = '16pt'
    p1.yaxis.major_label_text_font_size = "16pt"
    offset = 0

    for i in range(len(X_list)):
        for j in range(len(Y_list)):
            Y = input_df.iloc[:, i * len(Y_list) + j + 1] + offset
            offset += 1.5*(np.amax(Y)-np.amin(Y))
            
            p1.line(X, Y, color = bp.plasma(256)[(i*3)%145], name = 'Coordinate:' + str(X_list[i]) + ',' + str(Y_list[j]) )


    #p1.legend.visible = False

    show(p1)

def df_to_single_intensity_image( input_df, X_list, Y_list, wavenumber_of_interest, title = '_'):

    # White-light image use top-left corner as origin. Matplotlib use the bottom-left corner.
    # Needs to be fixed.
    im_data = np.zeros((len(Y_list), len(X_list))) # Not a mistake here. Fill Y first, then X.
    X = input_df['Wavenumber']
    #wavenumber_cnt = len(X)

    point_index = np.amin(np.where(X > wavenumber_of_interest))

    for i in range(len(X_list)):
        for j in range(len(Y_list)):
            im_data[j,i] = input_df.iloc[point_index, (i * len(Y_list) + j + 1)]

    fig = plt.figure()
    ax1 = fig.gca()
    im = ax1.imshow(im_data, interpolation = 'bilinear', extent = [np.amin(X_list), np.amax(X_list), np.amax(Y_list), np.amin(Y_list)], aspect=1)
    #
    #ax1.set_xticks([0,3,6,9,12,15,18])

    fig.colorbar(im,ax=ax1)
    plt.title('Filename: ' + title + ', Heatmap created using Raman shift at ' + str(wavenumber_of_interest) + ' cm-1')

    return im_data

def df_to_two_intensity_ratio( input_df, X_list, Y_list, wavenumber_of_interest_1, wavenumber_of_interest_2, ratio_ceiling = 10, title = '_' , ):

    im_data = np.zeros((len(Y_list), len(X_list))) # Not a mistake here. Fill Y first, then X.
    X = input_df['Wavenumber']
    #wavenumber_cnt = len(X)

    point_index_1 = np.amin(np.where(X > wavenumber_of_interest_1))    
    point_index_2 = np.amin(np.where(X > wavenumber_of_interest_2))

    for i in range(len(X_list)):
        for j in range(len(Y_list)):
            im_data[j,i] = ( input_df.iloc[point_index_1, (i * len(Y_list) + j + 1)] / input_df.iloc[point_index_2, (i * len(Y_list) + j + 1)])

    fig = plt.figure()
    ax1 = fig.gca()
    im = plt.imshow(im_data, interpolation = 'bilinear', extent = [np.amin(X_list), np.amax(X_list), np.amax(Y_list), np.amin(Y_list)], aspect=1 )
    #
    #ax1.set_xticks([0,3,6,9,12,15,18])

    plt.clim(0,ratio_ceiling)

    fig.colorbar(im,ax=ax1 )
    plt.title('Filename: ' + title + ', Heatmap created using Raman shift at ' + str(wavenumber_of_interest_1) + 'over' + str(wavenumber_of_interest_2) + ' cm-1')

    return im_data

def df_to_LHRatio( input_df, X_list, Y_list, splitpoint = 2075, ratio_floor = 0, ratio_ceiling = 1, title = '_'):

    im_data = np.zeros((len(Y_list), len(X_list))) # Not a mistake here. Fill Y first, then X.
    X = input_df['Wavenumber']
    #wavenumber_cnt = len(X)

    point_index_2000 = np.amin(np.where(X > 2000))
    point_index_2100 = np.amin(np.where(X > 2100))
    X = np.array(X[point_index_2000:point_index_2100])
    point_index_split = np.amin(np.where(X > splitpoint))

    for i in range(len(X_list)):
        for j in range(len(Y_list)):

            Y = np.array(input_df.iloc[point_index_2000:point_index_2100, (i * len(Y_list) + j + 1)])
            Y_Flat = remove_straight_baseline(X, Y)

            Low_sum = Y_Flat[:point_index_split].sum()
            Hi_sum = Y_Flat[point_index_split:].sum()



            im_data[j,i] = ( Low_sum/(Low_sum+Hi_sum))

    fig = plt.figure()
    ax1 = fig.gca()
    im = plt.imshow(im_data, interpolation = 'bilinear', extent = [np.amin(X_list), np.amax(X_list), np.amax(Y_list), np.amin(Y_list)], aspect=1 )
    #
    #ax1.set_xticks([0,3,6,9,12,15,18])

    plt.clim(ratio_floor,ratio_ceiling)

    fig.colorbar(im,ax=ax1 )
    plt.title('Filename: ' + title + ', LFB vs. HFB splitting at ' + str(splitpoint) + ' cm-1')

    return im_data    



def df_to_peakpos(input_df, X_list, Y_list, input_title, clim = False):

    im_data = np.zeros((len(Y_list), len(X_list)))
    X = input_df['Wavenumber']

    for i in range(len(X_list)):
        for j in range(len(Y_list)):
            peak_index = np.argmax(input_df.iloc[:, (i * len(Y_list) + j + 1)])
            im_data[j,i] = X[peak_index]

    fig = plt.figure()
    ax1 = fig.gca()
    im = ax1.imshow(im_data, interpolation = 'bilinear', extent = [np.amin(X_list), np.amax(X_list), np.amax(Y_list), np.amin(Y_list)], aspect=1)
    #
    #ax1.set_xticks([0,3,6,9,12,15,18])

    if clim == True:
        im.set_clim(2060,2100)

    fig.colorbar(im,ax=ax1)
    plt.title('Filename: ' + input_title + ', Heatmap created using Peak Position')

    return im_data




def txt_to_waterfall( input_filename , write_df = False):

    df, X_list, Y_list = txt_to_df(input_filename)

    df_to_waterfall_Bokeh(df, X_list, Y_list, input_title = input_filename)

    if write_df == True:
        df.to_csv(input_filename.replace('.txt', '') + '_dataframe.csv', index = False)

    return df

def dir_to_waterfall():

    filenamelist = os.listdir()

    for filename in filenamelist:
        if re.search(".txt", filename) and (not re.search(".html", filename)):
            txt_to_waterfall(filename, write_df=True)

def txt_to_single_intensity_image(input_filename, wavenumber_of_interest):

    df, X_list, Y_list = txt_to_df(input_filename)

    intensity_arr = df_to_single_intensity_image(df, X_list, Y_list, wavenumber_of_interest = wavenumber_of_interest, title = input_filename)

    return intensity_arr

def txt_to_two_intensity_ratio(input_filename, wavenumber_of_interest_1, wavenumber_of_interest_2, ratio_ceiling):

    df, X_list, Y_list = txt_to_df(input_filename)

    ratio_arr = df_to_two_intensity_ratio(df, X_list, Y_list, wavenumber_of_interest_1 = wavenumber_of_interest_1, wavenumber_of_interest_2=wavenumber_of_interest_2, title = input_filename, ratio_ceiling = ratio_ceiling)

    return ratio_arr

def txt_to_peak_pos(input_filename, clim = False):

    df, X_list, Y_list = txt_to_df(input_filename)

    peak_pos_arr = df_to_peakpos(df, X_list, Y_list, input_title = input_filename, clim = clim)

    return peak_pos_arr

def plotall_separate(input_filename):

    input_df = pd.read_csv(input_filename, index_col=0)

    spectra_cnt = input_df.shape[1] - 1

    max_val = 0

    hi_max_val = 0

    folder_name = input_filename.replace('.', '_')
    os.mkdir(folder_name)
    os.chdir('./' + folder_name)

    for i in range(spectra_cnt):
        Y = input_df.iloc[:,i+1]
        this_max = np.amax(Y) - np.amin(Y)
        if this_max > max_val:
            max_val  = this_max

    for j in range(spectra_cnt):
        plt.figure(figsize = (10,8), dpi=150)
        Y = input_df.iloc[:,j+1]
        X = input_df.iloc[:,0]
        plt.plot(X, Y)
        plt.ylim([0,max_val+1000])
        title_str = input_filename + input_df.columns[j+1].replace('.','_').replace(',', '_')+'um'
        plt.title(title_str)
        plt.xticks(np.arange(min(X), max(X)+1, 200))
        plt.savefig(title_str.replace('.', '_'))
        plt.close()

    input_df_low = input_df[input_df['Wavenumber'] < 700]

    folder_name = input_filename.replace('.', '_') + '_low'
    os.chdir('./..')
    os.mkdir(folder_name)
    os.chdir('./' + folder_name)

    for j in range(spectra_cnt):
        plt.figure(figsize = (10,8), dpi=150)
        Y = input_df_low.iloc[:,j+1]
        X = input_df_low.iloc[:,0]
        plt.plot(X, Y)
        plt.ylim([0,max_val+1000])
        title_str = input_filename + input_df_low.columns[j+1].replace('.','_').replace(',', '_')+'um_100-700 cm-1'
        plt.title(title_str)
        plt.xticks(np.arange(min(X), max(X)+1, 100))
        plt.savefig(title_str.replace('.', '_'))
        plt.close()

    input_df_hi = input_df[(input_df['Wavenumber'] > 1900) & (input_df['Wavenumber'] < 2200)]

    folder_name = input_filename.replace('.', '_') + '_hi'
    os.chdir('./..')
    os.mkdir(folder_name)
    os.chdir('./' + folder_name)

    for i in range(spectra_cnt):
        Y = input_df_hi.iloc[:,i+1]
        this_max = np.amax(Y) - np.amin(Y)
        if this_max > hi_max_val:
            hi_max_val  = this_max    

    for j in range(spectra_cnt):
        plt.figure(figsize = (10,8), dpi=150)
        Y = input_df_hi.iloc[:,j+1]        
        X = input_df_hi.iloc[:,0]
        Y = remove_straight_baseline(np.array(X), Y)+200
        plt.plot(X, Y)
        plt.ylim([0,2400])
        title_str = input_filename + input_df_hi.columns[j+1].replace('.','_').replace(',', '_')+'um_1900-2100 cm-1'
        plt.title(title_str)
        plt.xticks(np.arange(min(X), max(X)+1, 50))
        plt.savefig(title_str.replace('.', '_'))
        plt.close()

    os.chdir('./..')


    return






