import yfinance as yf

# use yahoo finance dataset
def download(company, start, end):
    dataset = yf.download(company, start = start, end=end)
    dataset.to_csv(f'..\..\dataset/{company}_data.csv')

    return f"{company}_data.csv"


# file path
def get_filepath():
    return "src/dataset/TSM_data.csv"
    
