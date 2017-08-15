import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statistics

initial_investment = 1000
csv_extension = '.csv'
portfolio_stock_names = ['AABA', 'FB', 'VEEV', 'GLD']
# Number of days of data
num_days = 0
# (1) Calculating the daily portfolio closing value for a portfolio consisting of Veeva, Gold, Facebook, Altaba


#INPUT: Name of stock data file to read from
#OUTPUT: A pandas dataframe containing the relevant data
def read_csv(name):
	return pd.read_csv(name)

#INPUT: Array of stock data file names
#FUNCTION: Plots all of the stock prices in a relatively elegant looking graph
#OUTPUT: None
def plot_price_data(name_array):
	index = 0
	graph_legend_data = []
	for stock_name in name_array:
		stock_data = pd.read_csv(stock_name+csv_extension)
		stock_data.set_index('Date', inplace=True)
		stock_data['Adj Close'].plot()
		num_days = stock_data['Adj Close'].values.size
		current_patch = mpatches.Patch(color= 'C'+str(index), label=stock_name)
		graph_legend_data.append(current_patch)
		index+=1
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.legend(handles=graph_legend_data)
	plt.show()

#INPUT: List of names of stocks in the given portfolio
#OUTPUT: Numpy array of the prices of all stocks in the portfolio
def get_price_data(name_array):
	stock_prices = 0
	for stock_name in name_array:
		stock_data = pd.read_csv(stock_name+csv_extension)
		stock_data = stock_data['Adj Close']
		current_price_column = []
		for price in stock_data.values:
			current_price_column.append([price])

		if type(stock_prices) == int:
			stock_prices = np.array(current_price_column)
		else:
			stock_prices = np.concatenate((stock_prices, current_price_column), 1)

	return stock_prices

#INPUT: List of the stock names in the portfolio
#OUTPUT: List of opening prices of the portfolio stocks on the first day of trading
def first_day_opening_prices(name_array):
	opening_costs = []
	for stock_name in name_array:
		stock_data = pd.read_csv(stock_name + csv_extension)
		stock_data_opening = stock_data['Open'][0]
		opening_costs.append(stock_data_opening)
	return opening_costs


#INPUT: Amount of money user is investing, number of stocks to invest in
#OUTPUT: An array of the percentages of each stock assuming we are trying to reach near-equal allocations
def get_almost_equal_allocations(cash_amount, number_stocks, opening_day_costs):
	percentage_allocations = []
	stock_amounts = []
	total_cash_spent = 0
	cash_allocation_per_stock = cash_amount/number_stocks
	for opening_day_cost in opening_day_costs:
		current_number_stocks = int(cash_allocation_per_stock/opening_day_cost)
		stock_amounts.append(current_number_stocks)
		current_stock_value = current_number_stocks*opening_day_cost
		percentage_allocations.append(current_stock_value)
		total_cash_spent+= current_stock_value

	print("Total Cash Actually Spent: " + str(total_cash_spent))
	print("Cash Not Spent: " + str(initial_investment-total_cash_spent))
	print("Total Stock Amounts: " + str(stock_amounts))
	print()

	for index in range(len(percentage_allocations)):
		percentage_allocations[index] = percentage_allocations[index]/total_cash_spent
	return (percentage_allocations, total_cash_spent)

#INPUT:
#OUTPUT:
def get_normed_vector(price_matrix):
	normed_matrix = []
	for index in range(len(price_matrix)):
		price_vector = price_matrix[index]
		normed_vector = np.divide(price_vector, opening_day_prices)
		print(normed_vector)
		normed_matrix.append(normed_vector)
	print()
	return np.array(normed_matrix)

#INPUT:
#OUTPUT:
def display_portfolio_value_over_time(portfolio_values):
	portfolio_value_vector = convert_row_to_column(portfolio_values)
	print(portfolio_value_vector)
	print()
	date_vector = get_dates_column(portfolio_stock_names[0])
	portfolio_value_vector = np.array(portfolio_value_vector)
	portfolio_value_matrix = np.append(date_vector, portfolio_value_vector, axis=1)
	portfolio_value_matrix = pd.DataFrame(portfolio_value_matrix, columns=('Date', 'Price'))
	portfolio_value_matrix.set_index('Date', inplace=True)
	portfolio_value_matrix['Price'] = portfolio_value_matrix['Price'].astype(float)
	print(portfolio_value_matrix)
	plt.title("Portfolio Value Over Time")
	plt.xlabel("Date")
	plt.ylabel("Price")
	portfolio_value_matrix['Price'].plot()
	plt.show()

#INPUT:
#OUTPUT:
def convert_row_to_column(row_list):
	column_list = []
	for row_element in row_list:
		column_list.append([row_element])
	return column_list

#INPUT:
#OUTPUT: 
def get_dates_column(name):
	stock_data = pd.read_csv(name+csv_extension)
	stock_date_data = stock_data['Date'].values
	date_column = convert_row_to_column(stock_date_data)
	return date_column

#INPUT:
#OUTPUT:
def get_daily_returns(portfolio_values):
	daily_returns = []
	for index in range(len(portfolio_values)-1):
		current_daily_return = (portfolio_values[index+1]-portfolio_values[index])/(portfolio_values[index])
		daily_returns.append(current_daily_return)
	return daily_returns

#INPUT:
#OUTPUT:
def get_cumulative_return(portfolio_values):
	return (portfolio_values[-1]/portfolio_values[0])-1

#INPUT:
#OUTPUT:
def get_average_daily_return(daily_returns):
	average_return = 0
	for daily_return in daily_returns:
		average_return+=daily_return
	return average_return/len(daily_returns)

#INPUT:
#OUTPUT:
def get_standard_daily_return(daily_returns):
	return statistics.stdev(daily_returns)

#INPUT: A list of the daily returns.
#OUTPUT: Computes the Sharpe ratio assuming a 0% daily risk free rate (since eoconomy is currently doing well)
def get_sharpe_ratio(daily_returns):
	return (get_average_daily_return(daily_returns))/(get_standard_daily_return(daily_returns))

plot_price_data(portfolio_stock_names)
print()
prices = get_price_data(portfolio_stock_names)
print("Total Stock Price Data: " + str(prices))
print()
opening_day_prices = first_day_opening_prices(portfolio_stock_names)
allocations = get_almost_equal_allocations(initial_investment, len(portfolio_stock_names), opening_day_prices)
normed_matrix = get_normed_vector(prices)
allocated_matrix = np.multiply(normed_matrix, allocations[0])
position_values = np.multiply(allocated_matrix, allocations[1])
portfolio_values = position_values.sum(axis=1)

daily_returns = get_daily_returns(portfolio_values)
cumulative_return = get_cumulative_return(portfolio_values)
average_daily_return = get_average_daily_return(daily_returns)
standard_daily_return = get_standard_daily_return(daily_returns)
sharpe_ratio = get_sharpe_ratio(daily_returns)

print("Normed Matrix: " + str(normed_matrix))
print()
print("Allocated Matrix: " + str(allocated_matrix))
print()
print("Position Values Matrix: " + str(position_values))
print()
print("Portfolio Values Matrix: " + str(portfolio_values))
print()
print("Daily Returns: " + str(daily_returns))
print()
print("Cumulative Return: " + str(cumulative_return))
print()
print("Average Daily Return: " + str(average_daily_return))
print()
print("Standard Daily Return: " + str(standard_daily_return))
print()
print("Sharpe Ratio: " + str(sharpe_ratio))

display_portfolio_value_over_time(portfolio_values)
