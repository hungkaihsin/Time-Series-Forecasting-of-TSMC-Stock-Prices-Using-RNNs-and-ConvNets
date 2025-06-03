import yfinance as yf

# for download the data set
tsmc = yf.download('TSM', start='2018-01-01', end='2024-12-31')
tsmc.to_csv('tsmc_data.csv')