import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.signal as scs

def read_data():
    df = pd.read_csv(r"C:/Users/macdo/OneDrive/Pulpit\studia/TSAF/project/CPITimeSeries.csv", low_memory=False)

    greece_cpi = df[(df['Country Name'] == 'Greece') & 
                    (df['Indicator Name'] == 'Consumer Price Index, All items')]

    date_columns = [col for col in greece_cpi.columns if 'M' in col and col[:4].isdigit()]
    date_columns_1990_onwards = [col for col in date_columns if int(col[:4]) >= 1990]

    values = np.array(pd.to_numeric(greece_cpi[date_columns_1990_onwards].iloc[0], errors='coerce'))
    start_date = '1990-01-01'
    periods = len(date_columns_1990_onwards)
    index = pd.date_range(start=start_date, periods=periods, freq='MS') 
    return values, index

def decompose_trend(values):
    omega = np.ones(13)
    M = omega.shape[0] 
    lag = int(np.floor((M-1)/2))
    
    values_aug = np.concatenate((values, np.flip(values[-lag:])))
    values_aug = np.concatenate((np.flip(values[0:lag]), values_aug))

    nf = range(lag, values_aug.size - lag)
    trend = np.zeros(values.size).astype('float64') 
    
    for n in nf:
        trend[n-lag] = (1 / float(np.sum(omega))) * np.sum(np.multiply(values_aug[n-lag : n+lag+1], omega))

    x_numeric = np.arange(len(values)) 
    [a,b] = np.polyfit(x_numeric, trend, 1) 
    trendline = a * x_numeric + b

    return trend, trendline

def decompose_seasonality(values_detrended):
    sos = scs.butter(N=5, fs=12, Wn=2.5, btype='lowpass', output='sos')
    return scs.sosfiltfilt(sos, values_detrended)

def autocov(values, T=0):
    N = len(values)
    mean = np.mean(values)
    cov = 0.0
    for i in range(N-T):
        cov += (values[i] - mean) * (values[i+T] - mean)
    return cov / N

def autocoef(values, T=0):
    return autocov(values,T)/autocov(values)

def correlogram(values):
    N=len(values)
    maxT=100
    corrl=np.zeros(maxT+1)
    for i in range(maxT+1):
        corrl[i]=autocoef(values,i)
    return corrl

def differencing(values):
    values_detrended1 = np.zeros(values.shape)
    erratic1 = np.zeros(values.shape)
    values_deseasoned1 = np.zeros(values.shape)
    for i in range(1, len(values)):
        values_detrended1[i] = values[i] - values[i-1]
    for i in range(1, len(values_detrended1)):
        erratic1[i] = values_detrended1[i] - values_detrended1[i-12]
    for i in range(1, len(values)):
        if i <= 12:
            values_deseasoned1[i]= values[i] - values[0]
        else:
            values_deseasoned1[i]= values[i] - values[i-12]


    return erratic1, values_detrended1, values_deseasoned1

def plot_cpi(values, index, trend, values_detrended, seasonal, values_deseasoned, erratic, trendline, values_detrended1, values_deseasoned1, erratic1):
    cpiTS = pd.Series(data=values, index=index, name="Greece CPI - All items")

    fig1 = plt.figure(constrained_layout=True, figsize=(10, 8))

    gs = GridSpec(4, 2, figure=fig1)
    ax1 = fig1.add_subplot(gs[0, 0])
    ax2 = fig1.add_subplot(gs[1, 0])
    ax3 = fig1.add_subplot(gs[1, 1])
    ax4 = fig1.add_subplot(gs[2, 0])
    ax5 = fig1.add_subplot(gs[2, 1])
    ax6 = fig1.add_subplot(gs[0, 1])
    ax7 = fig1.add_subplot(gs[3, :])


    cpiTS.plot(ax=ax1, xlabel='Time', ylabel='Value', legend=True)
    ax1.set_title('Consumer Price Index (CPI) - Greece')
    
    ax2.plot(index, trend, color='red', linewidth=2)
    ax2.plot(index, trendline, color='blue', linewidth=1, alpha=0.5)
    ax2.set_title('Trend (Moving Average)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')

    ax3.plot(index, values_detrended, color='green', linewidth=2)
    ax3.set_title('Detrended CPI (MA)')  
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')

    ax4.plot(index, seasonal, color='orange', linewidth=2)
    ax4.set_title('Seasonality (Low-pass Filtered)')    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Value')

    ax5.plot(index, values_deseasoned, color='purple', linewidth=2)
    ax5.set_title('Deseasoned CPI (Low-Pass)') 
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Value')

    ax6.plot(index, erratic, color='brown', linewidth=2)
    ax6.set_title('Erratic Component')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Value')

    ax7.plot(correlogram(erratic), marker='o', linestyle='-')
    ax7.set_title('Correlogram')
    ax7.set_xlabel('Lag')
    ax7.set_ylabel('Autocorrelation')
    ax7.grid()

    fig2 = plt.figure(constrained_layout=True, figsize=(10, 8))

    gs = GridSpec(4, 2, figure=fig2)
    ax8 = fig2.add_subplot(gs[0, 0])
    ax9 = fig2.add_subplot(gs[0, 1])
    ax10 = fig2.add_subplot(gs[1, 0])
    ax11 = fig2.add_subplot(gs[1, 1])
    ax12 = fig2.add_subplot(gs[2, 0])
    ax13 = fig2.add_subplot(gs[2, 1])
    ax14 = fig2.add_subplot(gs[3, :])

    ax8.plot(index, values, color='blue', linewidth=2)
    ax8.set_title('Original CPI Time Series')
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Value')
    
    ax9.plot(index, values_detrended1, color='green', linewidth=2)
    ax9.set_title('Detrended CPI (Differencing)')
    ax9.set_xlabel('Time')
    ax9.set_ylabel('Value')

    ax10.plot(index, values-values_detrended1-erratic1, color='orange', linewidth=2)
    ax10.set_title('Seasonality (Original-detrended)')
    ax10.set_xlabel('Time')
    ax10.set_ylabel('Value')

    ax11.plot(index, values_deseasoned1, color='purple', linewidth=2)
    ax11.set_title('Deseasoned CPI (Differencing)')
    ax11.set_xlabel('Time')
    ax11.set_ylabel('Value')

    ax12.plot(index, erratic1, color='brown', linewidth=2)
    ax12.set_title('Erratic Component (Differencing)')
    ax12.set_xlabel('Time')
    ax12.set_ylabel('Value')

    ax13.plot(index, values - values_deseasoned1-erratic1, color='cyan', linewidth=2)
    ax13.set_title('Trend (Original - Deseasoned)')
    ax13.set_xlabel('Time')
    ax13.set_ylabel('Value')

    ax14.plot(correlogram(erratic1), marker='o', linestyle='-')
    ax14.set_title('Correlogram (Differencing)')
    ax14.set_xlabel('Lag')
    ax14.set_ylabel('Autocorrelation')
    ax14.grid()

    plt.show()

def split_data(values, index, test_size=24):

    train_values = values[:-test_size]    
    train_index = index[:-test_size]    
    
    test_values = values[-test_size:]
    test_index = index[-test_size:]
    
    return train_values, test_values, train_index, test_index

if __name__ == "__main__":
    values, index = read_data()
    train_values, test_values, train_index, test_index = split_data(values, index)
    trend, trendline = decompose_trend(train_values) 
    values_detrended = train_values-trend
    seasonal = decompose_seasonality(values_detrended)
    values_deseasoned = train_values - seasonal
    erratic= values_detrended - seasonal
    erratic1, values_detrended1, values_deseasoned1 = differencing(train_values)
    show_plots=True # show/hide plots
    if(show_plots):
        plot_cpi(train_values, train_index, trend, values_detrended, seasonal, values_deseasoned, erratic, trendline,values_detrended1, values_deseasoned1, erratic1)
