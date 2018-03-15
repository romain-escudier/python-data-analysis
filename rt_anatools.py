import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot     as plt


def loess_smooth_handmade(data, fc, step=1, t=np.nan, t_final=np.nan):
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Loess filtering of a time serie
    % 
    % data_smooth = loess_smooth_handmade(data, fc)
    %
    % IN:
    %       - data      : time serie to filter
    %       - fc        : cut frequency
    % OPTIONS:
    %       - step      : step between two samples  (if regular)
    %       - t         : coordinates of input data (if not regular)
    %       - t_final   : coordinates of output data (default is t)
    %
    % OUT:
    %       - data_smooth : filtered data
    %
    % Written by R. Escudier (2018)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
  
    if np.all(np.isnan(t)):
        t = np.arange(0, len(data)*step, step) 
    
    
    if np.all(np.isnan(t_final)):
        t_final = t
  
    # Remove NaNs
    id_nonan = np.where(~np.isnan(data))
    t = t[id_nonan]
    data = data[id_nonan]
  
    # Period of filter
    tau = 1/fc
    
    # Initialize output vector
    data_smooth = np.ones(t_final.shape)*np.nan
 
    # Only compute for the points where t_final is in the range of t
    sx = np.where(np.logical_and(t_final >= t.min(), t_final <= t.max()))

    # Loop on final coordinates
    for i in sx[0]:
        # Compute distance between current point and the rest
        dn_tot = np.abs(t-t_final[i])/tau
        # Select neightboring points
        idx_weights = np.where(dn_tot<1)
        n_pts = len(idx_weights[0])
    
        # Only try to adjust the polynomial if there are at least 4 neighbors
        if n_pts > 3:
            dn = dn_tot[idx_weights]
            w = 1-dn*dn*dn
            weights = w**3
            # adjust a polynomial to these data points
            X = np.stack((np.ones((n_pts,)),t[idx_weights],t[idx_weights]**2)).T
            W = np.diag(weights)
            B = np.linalg.lstsq(np.dot(W,X),np.dot(W,data[idx_weights]))
            coeff = B[0]
            # Smoothed value is the polynomial value at this location
            data_smooth[i] = coeff[0]+coeff[1]*t_final[i]+coeff[2]*t_final[i]**2
        
    return data_smooth



def detrend_timeserie(data,x=np.ones(2)*np.nan):
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Detrend a time serie
    % 
    % data_smooth = loess_smooth_handmade(data, fc)
    %
    % IN:
    %       - data      : time serie to detrend
    % OPTIONS:
    %       - x         : coordinates (default is 1:N)
    %
    % OUT:
    %       - data_detrend : detrended data
    %
    % Written by R. Escudier (2018)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    
    # Create x vector if not in input
    if np.any(np.isnan(x)):
        x = np.arange(len(data))
        
    # Ignore NaNs for the regression
    id_nonan = np.where(~np.isnan(data))
    
    # Linear regression on a line
    X = np.stack((np.ones((len(x),)),x)).T
    B = np.linalg.lstsq(X[id_nonan,:].squeeze(),data[id_nonan])
    
    # Output are the residuals
    data_detrend = data-(B[0][0]+B[0][1]*x)

    return data_detrend


def compute_spectrum(y, Fs, taper=None, dof=2):
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the discrete Fourier Transform of the vector y using fft
    % h1 = rt_draw_fft_new(y,Fs,...)
    %
    % IN:
    %       - y      : time serie to transform
    %       - Fs     : sampling frequency
    %
    % OPTIONS:
    %       - taper  : type of taper to use (default is None)
    %       - dof    : number of degrees of freedom (default is 2, no band averaging)
    %
    % OUT:
    %       - PSD    : power spectrum
    %       - freq   : associated frequencies
    %
    % Written by R. Escudier (2018)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    
    y_tmp = y
    Dt = 1./Fs
    L = len(y)
    T = L*Dt

    # Remove NaNs (set to mean)
    y_tmp[np.isnan(y)] = np.nanmean(y_tmp)
    
    # Tappering
    if (not taper == None): 
        if (taper == 'Bartlett'):
            if L%2 == 0:
                w = np.concatenate((np.linspace(0,1,np.int(L/2)),np.linspace(1,0,np.int(L/2))))
            else:
                w_up = np.linspace(0,1,np.int(L/2)+1)
                w    = np.concatenate((w_up,np.linspace(w_up[-2],0,np.int(L/2))))
        elif (taper == 'Hann'):
            w = 0.5*(1-np.cos(2*np.pi*np.arange(0,L)/(L-1)))
            
        else:
            print('Error! Not a taper name')
            return
        y_tmp = y_tmp*w

    # Compute spectrum with fft function
    Y = np.fft.rfft(y_tmp)/L
    freq_tmp    = np.arange(0,np.int(L/2)+1)/T
    PSD_tmp     = 2*T*np.abs(Y**2)
    PSD_tmp[1]  = PSD_tmp[1]/2
    PSD_tmp[-1] = PSD_tmp[-1]/2

    # Smoothing
    if dof > 2:
        if dof%2 == 0:
            N_band_avg = np.int(dof/2)
            PSD_bands  = np.reshape(PSD_tmp[0:-(len(PSD_tmp)%N_band_avg)],(N_band_avg,-1))
            PSD        = np.mean(PSD_bands,axis=0);
            freq       = freq_tmp[np.int(np.ceil(N_band_avg/2)):len(freq_tmp):N_band_avg]
        else:
            print('Error! Number of DOF must be a multiple of 2')
            return
    else:
        PSD = PSD_tmp
        freq = freq_tmp
    
    return PSD, freq


def confidence_fft(dof, alpha=0.95):
    '''
    Return confidence interval factors for a spectrum with dof degrees of freedom
    '''
    perc = 1-alpha
    
    coeff = np.zeros(2)
    coeff[0] = dof/chi2.ppf(1-perc/2.,dof)
    coeff[1] = dof/chi2.ppf(perc/2.,dof)
    
    return coeff


def plot_spectrum(PSD, freq, xlim=[np.nan,np.nan],ylim=[np.nan,np.nan]):
    '''
    Plot a spectrum 
    '''
    if np.any(np.isnan(ylim)):
        vmin = np.floor(np.log10(PSD.min()))
        vmax = np.ceil(np.log10(PSD.max()))
        ylim = [10**vmin,10**vmax]
    
    plt.loglog(freq,PSD)
    plt.grid()

def compute_spectrum_slope(PSD,freq,freq_lim):
    '''
    Return the slope of a spectrum *PSD* within the *freq_lim* band of *freq*
    '''
    
    # Get the band where to compute the slope
    id_slope = np.where(np.logical_and(freq >= freq_lim[0],freq < freq_lim[1]))
    freq_slope = np.log10(freq[id_slope])
    PSD_slope = np.log10(PSD[id_slope])
    
    # Linear regression on a line
    X = np.stack((np.ones((len(freq_slope),)),freq_slope)).T
    B = np.linalg.lstsq(X,PSD_slope)
    
    return B[0][1]

def plot_spectrum_slope(x1,x2,y1,slope,color='k'):
    '''
    Plot the slope of a spectrum 
    '''
    
    # compute y-axis of second point
    y2 = 10**(slope*np.log10(x2)+np.log10(y1)-slope*np.log10(x1))
    # compute coordinates of text
    xt = 10**(np.log10(x1)+(np.log10(x2)-np.log10(x1))/2)
    yt = 10**(slope*np.log10(xt)+np.log10(y1)-slope*np.log10(x1))
    # Plot the line
    plt.loglog([x1,x2],[y1,y2],color=color)
    # Write the value
    plt.text(xt,yt,'%1.3f'%slope,color=color)

