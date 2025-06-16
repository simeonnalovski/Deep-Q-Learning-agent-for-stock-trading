import yfinance as yf
import pandas as pd
import datetime
def get_sp500_symbols():
    #kreiranje na lista od akcii od s&p500 indeks
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_df = tables[0]  # Citanje na prvata kolona
    return sp500_df['Symbol'].tolist()
def download_stocks_data(tickers_list):
    #prezemanje na podatoci za s&p500 kompaniite
   return yf.download(tickers_list, start="2019-01-01", end=datetime.date.today(),
                      group_by="ticker", interval="1d")
def preproces_and_save(data):
    #predprocesiranje i zacuvuvanje na podatocite
    data = (data.stack(level=0, future_stack=True)
            .rename_axis(['Date', 'Ticker']).reset_index())
    data.columns = ['Date', 'Ticker', 'Open', 'High', 'Low',
                    'Close', 'Adj Close', 'Volume']
    data = data.dropna()
    data= data.drop_duplicates()
    data = data.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    data.to_csv("s&p500.csv",index=False)
tickers = get_sp500_symbols()
dataset = download_stocks_data(tickers)
preproces_and_save(dataset)