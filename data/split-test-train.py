import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('s&p500.csv')
unique_tickers = df['Ticker'].unique()

for ticker in unique_tickers:
    # iteracija spored ticker (kratenka za akcijata)
    ticker_data = df[df['Ticker'] == ticker]
    ticker_data = ticker_data.drop('Ticker', axis=1)
    train_data, test_data = train_test_split(ticker_data, test_size=0.2,
                                             shuffle=False)
    train_data.to_csv(f'{ticker}_train.csv', index=False)
    test_data.to_csv(f'{ticker}_test.csv', index=False)

