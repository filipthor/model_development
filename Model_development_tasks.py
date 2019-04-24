import numpy as np
import scipy.stats as sp
import pandas as pd

'''
Written by Filip Thor between 2019-04-17 to 2019-04-24
'''


def black_scholes_european_nodividend(S, K, T, r, sigma, option="call"):
    '''

    :param S: spot price
    :param K: strike price
    :param T: time to maturity
    :param r: rate of interest
    :param sigma: volatility
    :param option: type of option: "call" or "put"
    :return: price of option
    '''
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))

    if option == "call": return S * sp.norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * sp.norm.cdf(d2, 0, 1)
    if option == "put": K * np.exp(-r * (T)) * sp.norm.cdf(-d2, 0, 1) - S * sp.norm.cdf(-d1, 0, 1)


def correlation(df):
    '''
    
    :param df: DataFrame of historical equity prices to which correlation is to be calculated
    :return: DataFrame, correlation matrix 
    '''
    return pd.DataFrame(data=df).corr()  # returns correlation matrix


def basket_option(S, rho, a, T, K, r, dividend, volatility, option="call"):
    '''
    
    :param S: np.array of prices
    :param rho: DataFrame of correlation matrix
    :param a: np.array of weights
    :param T: time to maturity
    :param K: strike price
    :param r: rate of interest
    :param dividend: np.array of individual dividends for assets 
    :param volatility: np.array of individual volatilities
    :param option: type of option: "call" or "put"
    :return: price of basket
    '''

    if len(S) != len(a) or len(S) != len(dividend) or len(S) != len(volatility):
        if len(S) != len(a): print("Error: There has to be the same amount of weights as there are equities in the basket. Presented: %d, required: %d." % (len(a), len(S)))
        if len(S) != len(dividend): print("Error: There has to be the same amount of individual dividends as there are equities in the basket. Presented: %d, required: %d." % (len(dividend), len(S)))
        if len(S) != len(volatility): print("Error: There has to be the same amount of individual volatilities as there are equities in the basket. Presented: %d, required: %d." % (len(volatility), len(S)))
        return -1

    q = -1 / T * np.log(np.inner(a * S, np.exp(T * dividend)) / np.inner(a, S))  # calculates dividend yield according to Brigo et al

    sigma_squared = 1 / T * np.log(np.inner(a * S * np.exp(T * dividend), np.exp(T * dividend) * np.exp(T * np.outer(volatility, volatility) * rho.values).dot(a * S)) / np.inner(a * S, np.exp(T * dividend)) ** 2)  # calculates sigma**2 according to Brigo et al.

    # calculates price of basket according to Black-Scholes using calculated dividend yield, volatility and weighted spot prices
    d1 = (np.log(np.inner(a, S) / K) + (r - q + sigma_squared / 2) * T) / (np.sqrt(sigma_squared) * np.sqrt(T))
    d2 = (np.log(np.inner(a, S) / K) + (r - q - sigma_squared / 2) * T) / (np.sqrt(sigma_squared) * np.sqrt(T))

    if option == 'call': return (np.inner(a, S) * np.exp(-q * T) * sp.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * sp.norm.cdf(d2, 0.0, 1.0))
    if option == 'put': return (K * np.exp(-r * T) * sp.norm.cdf(-d2, 0.0, 1.0) - np.inner(a, S) * np.exp(-q * T) * sp.norm.cdf(-d1, 0.0, 1.0))



def extract_historical_data(filenames, index="Seneste", date_column_name="Dato", data_merge_method="outer"):
    '''
    
    :param filenames: list of excel filenames as strings to read, from the same folder
    :param index: name of column of which type of data to read; opening, high, low, latest price.
    :param date_column_name: name of column where date information is stored
    :param data_merge_method: if there are dates missing, this determines the way to merge the data, by reducing the missing dates or increasing, placing NaNs where data is missing
    :return: DataFrame with columns: Date, followed by the specified price for each equity 
    '''
    files = []
    for f in filenames:
        files.append(pd.read_excel(f))  # reads equity data from files

    df = pd.Series.to_frame(files[0][date_column_name])  # Extracting first column from data to which merging equity price columns

    for f in files:
        df = pd.merge(df, f[[date_column_name, index]], on=date_column_name, how=data_merge_method)  # adds column of data based on indicated price, merging based on merge type

    df.sort_values(by=[date_column_name], inplace=True, ascending=False)  # sorts data in descending order in case data got jumbled when merging

    for i in range(len(filenames)):
        df.rename(columns={df.columns[i + 1]: filenames[i]}, inplace=True)  # setting names of columns based on file name

    return df


def extract_prices(filenames):
    '''
    
    :param filenames: list of excel filenames as strings to read, from the same folder
    :return: the last known price for each equity listed in filenames
    '''
    files = []
    for f in filenames:
        files.append(pd.read_excel(f))  # extracts all equitiy data from list of files to read

    S = np.zeros(len(filenames))
    for i in range(len(files)):
        S[i] = files[i].iloc[[-1], 4].values  # extracts latest price (column 4) from each equity

    return S


#===== test data =====#

filenames = ["Fingerprint.xls", "Sandvik.xls", "SwedishMatch.xls"]

T = 1
volatility = np.array([0.2, 0.5, 0.6])
dividend = np.array([0.04, 0.02, 0.03])
a = np.array([0.7, 0.2, 0.1])
S = np.array([10, 12, 8])
S = extract_prices(filenames)
rho=correlation(extract_historical_data(filenames))
K = 150
r = 0.03

print("black-scholes test:", black_scholes_european_nodividend(S=50, K=75, T=1, r=0.05, sigma=0.60), "\n")
print("correlation test: ", rho, "\n")
print("basket price test: ", basket_option(S,rho, a, T, K, r, dividend, volatility))

