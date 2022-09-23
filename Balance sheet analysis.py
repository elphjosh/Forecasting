# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 18:59:43 2021

@author: joshe
"""
import requests
import matplotlib.pyplot as plt

api_key = "378dccb401f0fbea5da1add85b69b1c0"

company = "NVDA"
years = 5 

balance_sheet = requests.get(f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{company}?limit={years}&apikey={api_key}')
balance_sheet = balance_sheet.json()


total_current_assets = balance_sheet[0]['totalCurrentAssets']
print(f"Total Current Assets of {company}: {total_current_assets:,}")

total_current_liabilities = balance_sheet[0]['totalCurrentLiabilities']
print(f"Total Current Liabilities of {company}: {total_current_liabilities:,}")

total_debt = balance_sheet[0]['totalDebt']
cash_and_equivalents = balance_sheet[0]['cashAndCashEquivalents']
cash_debt_difference = cash_and_equivalents - total_debt
print(f"Cash Debt Difference: {cash_debt_difference:,}")  #:, (formats numbers)


goodwill_and_intangibles = balance_sheet[0]['goodwillAndIntangibleAssets']
total_assets = balance_sheet[0]['totalAssets']
pct_intangible = goodwill_and_intangibles / total_assets

print(f"Pct: {pct_intangible * 100:.2f}")

##########Quarterly Data######

qbalance_sheet = requests.get(f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{company}?period=quarterly&limit={years}&apikey={api_key}')
qbalance_sheet = qbalance_sheet.json()
print(qbalance_sheet)

assets_q1 = qbalance_sheet[4]['totalAssets']    #most recent quarter is 0
assets_q2 = qbalance_sheet[3]['totalAssets']
assets_q3 = qbalance_sheet[2]['totalAssets']
assets_q4 = qbalance_sheet[1]['totalAssets']

assets_data = [assets_q1, assets_q2, assets_q3, assets_q4]
assets_data = [x / 1000000000 for x in assets_data]

plt.bar([1,2,3,4], assets_data)
plt.title(f"Quarterly Assets of {company}")
plt.xlabel("Quarters")
plt.ylabel("Total Assets (USD) (in Billions)")
plt.xticks([1,2,3,4],['Q1', 'Q2', 'Q3', 'Q4'])
plt.show()








######Get daily stock data




from fmp_python.fmp import FMP

fmp = FMP(api_key='378dccb401f0fbea5da1add85b69b1c0')
quote = fmp.get_quote('NVDA')
print(quote)

timeseries = fmp.get_quote_short('NVDA')
print(timeseries)

hist_chart = fmp.get_historical_chart('4hour','NVDA')
#print(hist_chart)



dates = []
close = []
count = 0
for dct in hist_chart:
    count += 1
    if count%6 == 0:
        dates.append(dct['date'])
        close.append(dct['close'])
    else:
        pass

#x = datetime. strptime(hist_chart['date'], '%y/%m/%d %H:%M:%S')
#y = hist_chart['close']
print(dates[::-1])
plt.plot(dates[::-1],close[::-1])
plt.show()


ax = plt.subplot()
ax.grid(True)
ax.set_axisbelow(True)
ax.set_title('NVDA Share Price', color = 'white')
ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')
ax.tick_params(axis='x', colors = 'white')
ax.tick_params(axis='y', colors = 'white')
ax.xaxis_date()
ax.plot(dates[::-1],close[::-1])

plt.show()
#fmp.get_index_quote('GSPC')
#fmp.get_historical_chart_index('5min', GSPC')
#fmp.get_historical_price('GSPC')3


