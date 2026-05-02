import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

def read_data():
    df = pd.read_csv("C:/Users/macdo/OneDrive/Pulpit\studia/TSAF/project/CPITimeSeries.csv", low_memory=False)

    greece_cpi = df[(df['Country Name'] == 'Greece') & 
                    (df['Indicator Name'] == 'Consumer Price Index, All items')]

    date_columns = [col for col in greece_cpi.columns if 'M' in col and col[:4].isdigit()]
    date_columns_1990_onwards = [col for col in date_columns if int(col[:4]) >= 1990]

    values = np.array(pd.to_numeric(greece_cpi[date_columns_1990_onwards].iloc[0], errors='coerce'))
    start_date = '1990-01-01'
    periods = len(date_columns_1990_onwards)
    index = pd.date_range(start=start_date, periods=periods, freq='MS') 
    
    # Funkcja seasonal_decompose najlepiej współpracuje bezpośrednio z obiektami pd.Series,
    # które mają zdefiniowany indeks czasowy.
    return pd.Series(data=values, index=index, name="Greece CPI - All items")

if __name__ == "__main__":
    # Wczytanie danych
    cpi_series = read_data()

    # Wykonanie klasycznej dekompozycji za pomocą statsmodels
    # model='additive' oznacza: Obserwacja = Trend + Sezonowość + Reszty (Erratic)
    # period=12 wymusza szukanie cykli rocznych na danych miesięcznych
    decomposition = seasonal_decompose(cpi_series, model='additive', period=12)

    # Funkcja .plot() obiektu decomposition automatycznie generuje panel z 4 wykresami:
    # 1. Oryginalne dane (Observed)
    # 2. Trend (wyliczony jako symetryczna średnia ruchoma)
    # 3. Sezonowość (Seasonal)
    # 4. Część losowa / błąd (Resid) - to Twój "Erratic component"
    fig = decomposition.plot()
    
    # Poprawa wyglądu i rozmiaru figury
    fig.set_size_inches(12, 8)
    fig.axes[0].set_title('Dekompozycja CPI dla Grecji (Statsmodels)')
    plt.tight_layout()
    plt.show()

    # (Opcjonalnie) Jak dobrać się do pojedynczych komponentów w kodzie?
    # Możesz do nich sięgnąć np. do dalszych testów stacjonarności:
    # trend_data = decomposition.trend
    # seasonal_data = decomposition.seasonal
    # erratic_data = decomposition.resid

    # Usunięcie NaN (resid ma NaN na brzegach)
    residuals = decomposition.resid.dropna()

    # Wykres korelogramu (ACF) z granicami ufności
    plt.figure(figsize=(10, 5))
    plot_acf(residuals, lags=40)  # możesz zmienić liczbę lagów
    plt.title('Korelogram (ACF) dla składnika losowego (residuals)')
    plt.tight_layout()
    plt.show()