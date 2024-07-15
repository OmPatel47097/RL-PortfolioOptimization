import yfinance as yf

def download_save_stock_data(tickers, start_date, end_date, interval = '1d'):
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval)
    data.to_csv('../data/stockdata.csv')
    print('Data downloaded and saved in data folder')

if __name__ == '__main__':
    tickers = ["AAPL", "MSFT", "GOOG", "NFLX", "NVDA"]
    start_date = "2010-01-01"
    end_date = "2024-01-01"
    download_save_stock_data(tickers, start_date, end_date)