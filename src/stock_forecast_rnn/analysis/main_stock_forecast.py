import warnings

import yfinance as yf

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # import six-year Google Stock data
    goog = yf.Ticker("GOOGL")
    goog_6y = goog.history(period="6y")

    # Data that I want to focus on are 'Open','High','Low','Closing', and Volume
    goog = goog_6y.iloc[:, 0:5]

    # Diagnose whether there are null data
    goog.isnull().sum()
