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

    ax10.plot(index, values_detrended1-erratic1, color='orange', linewidth=2)
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

    ax13.plot(index, values-values_detrended1, color='cyan', linewidth=2)
    ax13.set_title('Trend (Original - Deseasoned)')
    ax13.set_xlabel('Time')
    ax13.set_ylabel('Value')

    ax14.plot(correlogram(erratic1), marker='o', linestyle='-')
    ax14.set_title('Correlogram (Differencing)')
    ax14.set_xlabel('Lag')
    ax14.set_ylabel('Autocorrelation')
    ax14.grid()

    plt.show()

def split_data(values, index, test_size=48):

    train_values = values[:-test_size]    
    train_index = index[:-test_size]    
    
    test_values = values[-test_size:]
    test_index = index[-test_size:]
    
    return train_values, test_values, train_index, test_index

def manual_pacf(data, max_lag=20):
    """
    Ręcznie oblicza Częściową Funkcję Autokorelacji (PACF) 
    poprzez sekwencyjne dopasowywanie modeli AR(p) metodą OLS.
    """
    N = len(data)
    pacf_values = np.zeros(max_lag + 1)
    pacf_values[0] = 1.0  # PACF dla opóźnienia 0 to zawsze 1.0 (korelacja z samym sobą)

    print("Obliczanie PACF...")
    for p in range(1, max_lag + 1):
        # 1. Przygotowujemy macierz opóźnień X i wektor celów Y dla modelu AR(p)
        X = np.zeros((N - p, p))
        Y = np.zeros(N - p)

        for i in range(p, N):
            X[i-p] = data[i-p : i][::-1]  # Opóźnienia od 1 do p
            Y[i-p] = data[i]

        # Dodajemy kolumnę jedynek (wyraz wolny) do macierzy X
        X = np.column_stack((np.ones(len(X)), X))

        # 2. Metoda Najmniejszych Kwadratów (OLS): beta = (X^T * X)^-1 * X^T * Y
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y

        # 3. PACF dla danego opóźnienia to OSTATNI współczynnik z wyliczonego wektora beta
        # beta[0] to wyraz wolny, beta[1] to lag 1, ..., beta[-1] to lag p.
        pacf_values[p] = beta[-1]

    # Obliczanie przedziału ufności dla wykresu (5% poziom istotności)
    conf_interval = 1.96 / np.sqrt(N)
    
    return pacf_values, conf_interval

def plot_manual_pacf(pacf_values, conf_interval):
    """
    Rysuje wykres Częściowej Funkcji Autokorelacji (PACF) z granicami ufności.
    """
    lags = np.arange(len(pacf_values))
    
    plt.figure(figsize=(10, 4))
    
    # Rysowanie "lizaków" (słupków)
    plt.stem(lags, pacf_values, basefmt="k-")    
    
    # Rysowanie niebieskiego paska przedziału ufności
    plt.axhspan(-conf_interval, conf_interval, alpha=0.2, color='blue', label='Przedział ufności (95%)')
    
    # Linie pomocnicze i opisy
    plt.axhline(0, color='black', linewidth=1)
    plt.title("Ręczny wykres PACF (Partial Autocorrelation Function)")
    plt.xlabel("Opóźnienie (Lag)")
    plt.ylabel("Częściowa Autokorelacja")
    plt.xticks(lags) # Pokazuje wszystkie numery lagów na osi X
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def fit_ar_ols(data, p):
    """
    Dopasowuje model AR(p) za pomocą Metody Najmniejszych Kwadratów (OLS).
    """
    N = len(data)
    if p == 0:
        mean = np.mean(data)
        return mean, np.array([]), data - mean

    # Macierz X (predyktory) i wektor Y (cel)
    X = np.zeros((N - p, p + 1))
    Y = data[p:]
    
    X[:, 0] = 1.0  # Wyraz wolny (intercept)
    for i in range(1, p + 1):
        X[:, i] = data[p - i : N - i]

    # Rozwiązanie OLS: beta = (X^T * X)^-1 * X^T * Y
    # Używamy lstsq dla lepszej stabilności numerycznej
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    
    intercept = beta[0]
    ar_coefs = beta[1:]

    # Obliczanie reszt (residuals)
    fitted = X @ beta
    residuals = np.zeros(N)
    residuals[p:] = Y - fitted
    
    return intercept, ar_coefs, residuals


def fit_arma_hannan_rissanen(data, p, q, m=15):
    """
    Dopasowuje model ARMA(p,q) wykorzystując algorytm Hannana-Rissanena.
    m to rząd pomocniczego modelu AR używanego do oszacowania reszt.
    """
    if q == 0:
        # Jeśli q=0, to po prostu zwykły model AR
        intercept, ar_coefs, res = fit_ar_ols(data, p)
        return intercept, ar_coefs, np.array([]), res

    # KROK 1: Dopasowanie AR(m), aby uzyskać pierwsze oszacowanie reszt
    # m powinno być większe niż max(p, q).
    m = max(m, p + q + 1)
    _, _, init_residuals = fit_ar_ols(data, m)

    # KROK 2: Właściwe dopasowanie OLS z wykorzystaniem X i oszacowanych reszt
    N = len(data)
    start = max(p, q, m) # Startujemy tam, gdzie mamy pełne dane dla X i reszt
    num_rows = N - start

    X = np.zeros((num_rows, 1 + p + q))
    Y = data[start:]

    X[:, 0] = 1.0  # Wyraz wolny
    
    # Dodawanie opóźnień AR (zmienna zależna)
    for i in range(1, p + 1):
        X[:, i] = data[start - i : N - i]
        
    # Dodawanie opóźnień MA (oszacowane reszty z Kroku 1)
    for j in range(1, q + 1):
        X[:, p + j] = init_residuals[start - j : N - j]

    # Rozwiązanie OLS
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    
    intercept = beta[0]
    ar_coefs = beta[1 : p + 1]
    ma_coefs = beta[p + 1 :]

    # Ostateczne wyliczenie reszt dla naszego modelu ARMA
    fitted = X @ beta
    final_residuals = np.zeros(N)
    final_residuals[start:] = Y - fitted

    return intercept, ar_coefs, ma_coefs, final_residuals


def forecast_arma(data, intercept, ar_coefs, ma_coefs, residuals, steps):
    """
    Generuje predykcję na 'steps' kroków w przód na bazie dopasowanego modelu ARMA.
    """
    p = len(ar_coefs)
    q = len(ma_coefs)
    
    # Kopiujemy dane i reszty do list, aby móc dynamicznie "doklejać" predykcje
    history = list(data)
    past_res = list(residuals)

    forecasts = []
    for h in range(steps):
        pred = intercept
        
        # Komponent AR: opiera się na rzeczywistej historii lub poprzednich predykcjach
        for i in range(p):
            pred += ar_coefs[i] * history[-1 - i]
            
        # Komponent MA: opiera się TYLKO na historycznych, znanych błędach.
        # Wartość oczekiwana błędu w przyszłości to 0, więc ich nie dodajemy.
        for j in range(q):
            # Używamy historycznych reszt tylko jeśli wciąż sięgamy w przeszłość
            if h - j <= 0: 
                pred += ma_coefs[j] * past_res[-1 - j + h]

        forecasts.append(pred)
        history.append(pred) # Dodajemy predykcję jako "fakt" dla kolejnego kroku AR
        past_res.append(0.0) # Przyszłe błędy zawsze wynoszą 0
        
    return np.array(forecasts)

def reconstruct_forecast(original_values, differenced_values, erratic_forecast, steps):
    """
    Odtwarza oryginalny szereg czasowy z prognozy komponentu resztkowego (erratic),
    odwracając proces pierwszego różniczkowania i różniczkowania sezonowego (lag=12).
    """
    # Tworzymy listy z historycznymi danymi, aby móc do nich "doklejać" 
    # nasze predykcje, z których będziemy korzystać w kolejnych krokach iteracji.
    hist_D = list(differenced_values)
    hist_X = list(original_values)
    
    final_forecast = []
    
    for h in range(steps):
        # Aktualna prognoza samego szumu
        pred_erratic = erratic_forecast[h]
        
        # 1. Odwrócenie sezonowości (lag=12)
        # D(t) = Erratic(t) + D(t-12)
        pred_D = pred_erratic + hist_D[-12]
        hist_D.append(pred_D) # Zapisujemy, bo przyda się za 12 kroków
        
        # 2. Odwrócenie trendu (lag=1)
        # X(t) = D(t) + X(t-1)
        pred_X = pred_D + hist_X[-1]
        hist_X.append(pred_X) # Zapisujemy, bo przyda się w następnym kroku
        
        final_forecast.append(pred_X)
        
    return np.array(final_forecast)

def manual_holt_winters(data, slen=12, alpha=0.2, beta=0.05, gamma=0.3, steps_ahead=24):
    """
    Ręczna implementacja modelu Holta-Wintersa (Triple Exponential Smoothing)
    dla wariantu addytywnego.
    
    alpha, beta, gamma to wagi dla (odpowiednio) poziomu, trendu i sezonowości.
    """
    data = list(data) # Upewniamy się, że operujemy na liście
    
    # ---------------------------------------------------------
    # 1. INICJALIZACJA (dla pierwszego roku, tj. slen=12)
    # ---------------------------------------------------------
    # Początkowy poziom: średnia z pierwszego roku
    L = [sum(data[0:slen]) / float(slen)]
    
    # Początkowy trend: średnia różnica między 2. a 1. rokiem
    # b0 = ( (Y_13 - Y_1) + (Y_14 - Y_2) + ... ) / 12^2
    trend_init = sum([ (data[slen+i] - data[i]) / slen for i in range(slen) ]) / slen
    B = [trend_init]
    
    # Początkowa sezonowość: różnica między wartością z 1. roku a początkowym poziomem
    S = [data[i] - L[0] for i in range(slen)]
    
    # ---------------------------------------------------------
    # 2. DOPASOWANIE DO DANYCH (Filtrowanie)
    # ---------------------------------------------------------
    for t in range(len(data)):
        if t == 0:
            continue # Zerowy element już zainicjowaliśmy
            
        # Odczytujemy historyczną sezonowość (sprzed roku)
        # Jeśli t < slen, bierzemy sezonowość inicjalną. Jeśli t >= slen, bierzemy z historii wyliczeń.
        s_t_minus_s = S[t] if t < slen else S[-slen]
        
        # Równania Holta-Wintersa
        l_t = alpha * (data[t] - s_t_minus_s) + (1 - alpha) * (L[-1] + B[-1])
        b_t = beta * (l_t - L[-1]) + (1 - beta) * B[-1]
        s_t = gamma * (data[t] - l_t) + (1 - gamma) * s_t_minus_s
        
        # Zapisujemy nowe wartości
        L.append(l_t)
        B.append(b_t)
        S.append(s_t)
        
    # ---------------------------------------------------------
    # 3. PROGNOZOWANIE (Forecasting)
    # ---------------------------------------------------------
    forecast = []
    for m in range(1, steps_ahead + 1):
        # Pobieramy najświeższą sezonowość z odpowiedniego miesiąca z przeszłości
        # s_index wylicza, którą z ostatnich 12 wartości sezonowych musimy pobrać
        s_index = -slen + ((m - 1) % slen) 
        
        # Wzór na prognozę
        pred = L[-1] + (m * B[-1]) + S[s_index]
        forecast.append(pred)
        
    return np.array(forecast)

if __name__ == "__main__":
    values, index = read_data()
    train_values, test_values, train_index, test_index = split_data(values, index)
    
    trend, trendline = decompose_trend(train_values) 
    values_detrended = train_values - trend
    seasonal = decompose_seasonality(values_detrended)
    values_deseasoned = train_values - seasonal
    erratic = values_detrended - seasonal
    
    erratic1, values_detrended1, values_deseasoned1 = differencing(train_values)
    
    show_plots = False # show/hide plots
    if(show_plots):
        plot_cpi(train_values, train_index, trend, values_detrended, seasonal, values_deseasoned, erratic, trendline, values_detrended1, values_deseasoned1, erratic1)
    
    # 1. Przygotowanie stacjonarnego szeregu po różniczkowaniu
    clean_erratic_train = erratic1[13:]    
    
    # 2. Ręczne wyliczenie i wykres PACF
    my_pacf, my_conf_int = manual_pacf(clean_erratic_train, max_lag=20)    
    plot_manual_pacf(my_pacf, my_conf_int)

    # ------------------ NOWY KOD - DODANIE MODELU ARMA ------------------ #
    
    # Wybieramy rzędy p i q (przykładowo p=2, q=2)
    p_order = 2
    q_order = 2
    steps_ahead = len(test_values)
    
    # Dopasowanie modelu algorytmem Hannana-Rissanena
    print(f"Trenowanie ręcznego modelu ARMA({p_order}, {q_order})...")
    intercept, ar_coefs, ma_coefs, residuals = fit_arma_hannan_rissanen(
        clean_erratic_train, p=p_order, q=q_order
    )
    
    # Generowanie prognozy samej reszty (erratic)
    forecast_erratic = forecast_arma(
        clean_erratic_train, intercept, ar_coefs, ma_coefs, residuals, steps=steps_ahead
    )
    
    # -------------------------------------------------------------------- #

    # Rekonstruujemy prognozę CPI (odwracamy różniczkowanie)
    reconstructed_cpi_forecast = reconstruct_forecast(
        original_values=train_values, 
        differenced_values=values_detrended1, 
        erratic_forecast=forecast_erratic, 
        steps=steps_ahead
    )

    # Rysowanie końcowego wykresu
    plt.figure(figsize=(12, 6))

    # Rysujemy końcówkę danych treningowych (np. ostatnie 5 lat = 60 miesięcy)
    plot_start = 300
    plt.plot(train_index[plot_start:], train_values[plot_start:], color='blue', label='Historia CPI (Trening)')

    # Rysujemy rzeczywiste dane testowe CPI
    plt.plot(test_index, test_values, color='black', label='Rzeczywiste CPI (Test)')

    # Rysujemy naszą zrekonstruowaną prognozę CPI
    plt.plot(test_index, reconstructed_cpi_forecast, color='red', linestyle='dashed', linewidth=2, label='Prognoza z modelu (ARMA + Trend + Sezon)')

    plt.title("Ostateczna Prognoza CPI")
    plt.xlabel("Czas")
    plt.ylabel("Wartość CPI")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    steps_to_forecast = len(test_values)
    
    # Dobór parametrów alpha, beta, gamma to tzw. hiperparametryzacja.
    # W gotowych bibliotekach (jak statsmodels) działa pod spodem optymalizator (np. L-BFGS), 
    # który szuka takich wag, aby zminimalizować błąd. My ustawiamy je na sztywno, 
    # ale możesz się nimi pobawić! (np. większa gamma = większa uwaga dla nowszych zmian sezonowych)
    hw_manual_forecast = manual_holt_winters(
        train_values, 
        slen=12, 
        alpha=0.3,  
        beta=0.02, 
        gamma=0.4, 
        steps_ahead=steps_to_forecast
    )

    # Rysowanie wykresu
    plt.figure(figsize=(12, 6))

    # Aby było lepiej widać, pokazujemy tylko np. ostatnie 60 miesięcy z treningu
    plot_start = 300
    plt.plot(train_index[plot_start:], train_values[plot_start:], color='blue', label='Historia CPI (Trening)')

    # Prawdziwe CPI ze zbioru testowego
    plt.plot(test_index, test_values, color='black', label='Rzeczywiste CPI (Test)')

    # Prognoza z ręcznego Holta-Wintersa
    plt.plot(test_index, hw_manual_forecast, color='green', linestyle='dashed', linewidth=2, label='Ręczna Prognoza Holt-Winters')

    plt.title("Ręczna implementacja Holta-Wintersa")
    plt.xlabel("Czas")
    plt.ylabel("Wartość CPI")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()