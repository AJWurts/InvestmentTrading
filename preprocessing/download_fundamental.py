import requests
def getURL(ticker):
    return "https://stockrow.com/api/companies/" + ticker + "/financials.xlsx?dimension=MRQ&section=Income%20Statement&sort=desc"

with open("stocks.txt", 'r') as stocks:
    split = stocks.read().split('\n')
    for s in split:
        result = requests.get(getURL(s))
        print(result)
        with open(s + '.xlsx', 'wb') as out:
            out.write(result.content)
