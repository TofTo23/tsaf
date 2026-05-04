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

def split_data(values, index, test_size=48):
    train_values = values[:-test_size]    
    train_index = index[:-test_size]    
    
    test_values = values[-test_size:]
    test_index = index[-test_size:]
    
    return train_values, test_values, train_index, test_index

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
    N = len(values)
    maxT = 100
    corrl = np.zeros(maxT+1)
    for i in range(maxT+1):
        corrl[i] = autocoef(values,i)
    return corrl

def fit_ar(data, p):
    N = len(data)
    if p == 0:
        mean = np.mean(data)
        return mean, np.array([]), data - mean

    X = np.zeros((N - p, p + 1))
    Y = data[p:]
    
    X[:, 0] = 1.0  
    for i in range(1, p + 1):
        X[:, i] = data[p - i : N - i]

    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    
    intercept = beta[0]
    ar_coefs = beta[1:]

    fitted = X @ beta
    residuals = np.zeros(N)
    residuals[p:] = Y - fitted
    
    return intercept, ar_coefs, residuals

def fit_arma(data, p, q, m=15):
    if q == 0:
        intercept, ar_coefs, res = fit_ar(data, p)
        return intercept, ar_coefs, np.array([]), res

    m = max(m, p + q + 1)
    _, _, init_residuals = fit_ar(data, m)

    N = len(data)
    start = max(p, q, m) 
    num_rows = N - start

    X = np.zeros((num_rows, 1 + p + q))
    Y = data[start:]
    X[:, 0] = 1.0  
    
    for i in range(1, p + 1):
        X[:, i] = data[start - i : N - i]
        
    for j in range(1, q + 1):
        X[:, p + j] = init_residuals[start - j : N - j]

    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    
    intercept = beta[0]
    ar_coefs = beta[1 : p + 1]
    ma_coefs = beta[p + 1 :]

    fitted = X @ beta
    final_residuals = np.zeros(N)
    final_residuals[start:] = Y - fitted

    return intercept, ar_coefs, ma_coefs, final_residuals

def forecast_arma(data, intercept, ar_coefs, ma_coefs, residuals, steps):
    p, q = len(ar_coefs), len(ma_coefs)
    history, past_res = list(data), list(residuals)
    forecasts = []
    
    for h in range(steps):
        pred = intercept
        for i in range(p):
            pred += ar_coefs[i] * history[-1 - i]
        for j in range(q):
            if h - j <= 0: 
                pred += ma_coefs[j] * past_res[-1 - j + h]

        forecasts.append(pred)
        history.append(pred) 
        past_res.append(0.0) 
        
    return np.array(forecasts)

def reconstruct_forecast(original_values, differenced_values, erratic_forecast, steps):
    hist_D = list(differenced_values)
    hist_X = list(original_values)
    final_forecast = []
    
    for h in range(steps):
        pred_erratic = erratic_forecast[h]
        pred_D = pred_erratic + hist_D[-12]
        hist_D.append(pred_D) 
        
        pred_X = pred_D + hist_X[-1]
        hist_X.append(pred_X) 
        
        final_forecast.append(pred_X)
        
    return np.array(final_forecast)

def holt_winters(data, slen=12, alpha=0.2, beta=0.05, gamma=0.3, steps_ahead=24):
    data = list(data) 
    
    L = [sum(data[0:slen]) / float(slen)]
    trend_init = sum([ (data[slen+i] - data[i]) / slen for i in range(slen) ]) / slen
    B = [trend_init]
    S = [data[i] - L[0] for i in range(slen)]
    
    for t in range(len(data)):
        if t == 0: continue
            
        s_t_minus_s = S[t] if t < slen else S[-slen]
        
        l_t = alpha * (data[t] - s_t_minus_s) + (1 - alpha) * (L[-1] + B[-1])
        b_t = beta * (l_t - L[-1]) + (1 - beta) * B[-1]
        s_t = gamma * (data[t] - l_t) + (1 - gamma) * s_t_minus_s
        
        L.append(l_t)
        B.append(b_t)
        S.append(s_t)
        
    forecast = []
    for m in range(1, steps_ahead + 1):
        s_index = -slen + ((m - 1) % slen) 
        pred = L[-1] + (m * B[-1]) + S[s_index]
        forecast.append(pred)
        
    return np.array(forecast)

def plot_cpi(values, index, trend, values_detrended, seasonal, values_deseasoned, erratic, trendline, values_detrended1, values_deseasoned1, erratic1):
    cpiTS = pd.Series(data=values, index=index, name="Greece CPI - All items")

    fig1 = plt.figure(constrained_layout=True, figsize=(10, 8))
    gs = GridSpec(4, 2, figure=fig1)
    
    ax1, ax2 = fig1.add_subplot(gs[0, 0]), fig1.add_subplot(gs[1, 0])
    ax3, ax4 = fig1.add_subplot(gs[1, 1]), fig1.add_subplot(gs[2, 0])
    ax5, ax6 = fig1.add_subplot(gs[2, 1]), fig1.add_subplot(gs[0, 1])
    ax7 = fig1.add_subplot(gs[3, :])

    cpiTS.plot(ax=ax1, xlabel='Time', ylabel='Value', legend=True)
    ax1.set_title('Consumer Price Index (CPI) - Greece')
    
    ax2.plot(index, trend, color='red', linewidth=2)
    ax2.plot(index, trendline, color='blue', linewidth=1, alpha=0.5)
    ax2.set_title('Trend (Moving Average)')

    ax3.plot(index, values_detrended, color='green', linewidth=2)
    ax3.set_title('Detrended CPI (MA)')  

    ax4.plot(index, seasonal, color='orange', linewidth=2)
    ax4.set_title('Seasonality (Low-pass Filtered)')    

    ax5.plot(index, values_deseasoned, color='purple', linewidth=2)
    ax5.set_title('Deseasoned CPI (Low-Pass)') 

    ax6.plot(index, erratic, color='brown', linewidth=2)
    ax6.set_title('Erratic Component')

    ax7.plot(correlogram(erratic), marker='o', linestyle='-')
    ax7.set_title('Correlogram')
    ax7.grid()

    fig2 = plt.figure(constrained_layout=True, figsize=(10, 8))
    gs2 = GridSpec(4, 2, figure=fig2)
    
    ax8, ax9 = fig2.add_subplot(gs2[0, 0]), fig2.add_subplot(gs2[0, 1])
    ax10, ax11 = fig2.add_subplot(gs2[1, 0]), fig2.add_subplot(gs2[1, 1])
    ax12, ax13 = fig2.add_subplot(gs2[2, 0]), fig2.add_subplot(gs2[2, 1])
    ax14 = fig2.add_subplot(gs2[3, :])

    ax8.plot(index, values, color='blue', linewidth=2)
    ax8.set_title('Original CPI')
    
    ax9.plot(index, values_detrended1, color='green', linewidth=2)
    ax9.set_title('Detrended CPI (Differencing)')

    ax10.plot(index, values_detrended1-erratic1, color='orange', linewidth=2)
    ax10.set_title('Seasonality (Detrended-erratic)')

    ax11.plot(index, values_deseasoned1, color='purple', linewidth=2)
    ax11.set_title('Deseasoned CPI (Differencing)')

    ax12.plot(index, erratic1, color='brown', linewidth=2)
    ax12.set_title('Erratic Component (Differencing)')

    ax13.plot(index, values-values_detrended1, color='cyan', linewidth=2)
    ax13.set_title('Trend (Original - Detrended)')

    ax14.plot(correlogram(erratic1), marker='o', linestyle='-')
    ax14.set_title('Correlogram (Differencing)')
    ax14.grid()

    plt.show()

def plot_forecasts(train_index, train_values, test_index, test_values, forecasts_dict, title="Prognoza CPI", lookback=60):

    plt.figure(figsize=(12, 6))

    plot_start = max(0, len(train_values) - lookback)
    
    plt.plot(train_index[plot_start:], train_values[plot_start:], color='blue', label='Historia CPI (Trening)')
    plt.plot(test_index, test_values, color='black', label='Rzeczywiste CPI (Test)')

    colors = ['red', 'green', 'orange', 'purple']
    for idx, (name, forecast) in enumerate(forecasts_dict.items()):
        color = colors[idx % len(colors)]
        plt.plot(test_index, forecast, color=color, linestyle='dashed', linewidth=2, label=name)

    plt.title(title)
    plt.xlabel("Czas")
    plt.ylabel("Wartość CPI")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    SHOW_PLOTS = True 
    SHOW_FORECAST = True      
    
    values, index = read_data()
    train_values, test_values, train_index, test_index = split_data(values, index, test_size=48)
    
    trend, trendline = decompose_trend(train_values) 
    values_detrended = train_values - trend
    seasonal = decompose_seasonality(values_detrended)
    values_deseasoned = train_values - seasonal
    erratic = values_detrended - seasonal
    erratic1, values_detrended1, values_deseasoned1 = differencing(train_values)
    
    if SHOW_PLOTS:
        plot_cpi(train_values, train_index, trend, values_detrended, seasonal, 
                 values_deseasoned, erratic, trendline, values_detrended1, 
                 values_deseasoned1, erratic1)
    
    clean_erratic_train = erratic1[13:]
    steps_ahead = len(test_values)
    
    p_order, q_order = 2, 2
    intercept, ar_coefs, ma_coefs, residuals = fit_arma(
        clean_erratic_train, p=p_order, q=q_order
    )
    forecast_erratic = forecast_arma(
        clean_erratic_train, intercept, ar_coefs, ma_coefs, residuals, steps=steps_ahead
    )
    reconstructed_arma_forecast = reconstruct_forecast(
        original_values=train_values, 
        differenced_values=values_detrended1, 
        erratic_forecast=forecast_erratic, 
        steps=steps_ahead
    )
    
    hw_forecast = holt_winters(
        train_values, slen=12, alpha=0.3, beta=0.02, gamma=0.4, steps_ahead=steps_ahead
    )

    if SHOW_FORECAST:
        forecasts_to_plot = {
            'SARIMA ': reconstructed_arma_forecast,
            'Holt-Winters': hw_forecast
        }
        
        plot_forecasts(
            train_index, train_values, test_index, test_values, 
            forecasts_dict=forecasts_to_plot, 
            title="Forecast Models Comparison",
            lookback=48 
        )