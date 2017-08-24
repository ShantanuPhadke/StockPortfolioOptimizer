import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statistics
import math
import scipy.optimize as spo


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

#INPUT: A matrix of stock prices over time for multiple stocks
#OUTPUT: A matrix of stock prices that has been "normalized" with respect to the opening price
#of each of the stocks.
def get_normed_vector(price_matrix):
	normed_matrix = []
	for index in range(len(price_matrix)):
		price_vector = price_matrix[index]
		normed_vector = np.divide(price_vector, opening_day_prices)
		print(normed_vector)
		normed_matrix.append(normed_vector)
	print()
	return np.array(normed_matrix)

#INPUT: A list containing a series of portfolio values (over the course of time)
#OUTPUT: Graphs out the portfolio valuation over time using matplotlib.pyplot
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

#INPUT: A list that has a single row
#OUTPUT: A list that has a single column
def convert_row_to_column(row_list):
	column_list = []
	for row_element in row_list:
		column_list.append([row_element])
	return column_list

#INPUT: A String which is the name of a given stock.
#OUTPUT: Returns the dates column of the stock with the specified name.
def get_dates_column(name):
	stock_data = pd.read_csv(name+csv_extension)
	stock_date_data = stock_data['Date'].values
	date_column = convert_row_to_column(stock_date_data)
	return date_column

#INPUT: A list of portfolio values over time.
#OUTPUT: A list of daily returns (in terms of %) values over time.
def get_daily_returns(portfolio_values):
	daily_returns = []
	for index in range(len(portfolio_values)-1):
		current_daily_return = (portfolio_values[index+1]-portfolio_values[index])/(portfolio_values[index])
		daily_returns.append(current_daily_return)
	return daily_returns

#INPUT: A list of portfolio values of a given portfolio over time.
#OUTPUT: Cummulative return (that is the start-to-end return) for the given portfolio.
def get_cumulative_return(portfolio_values):
	return (portfolio_values[-1]/portfolio_values[0])-1

#INPUT: A list containing the daily return values for a given portfolio.
#OUTPUT: An average over the daily return values found within the inputted list.
def get_average_daily_return(daily_returns):
	average_return = 0
	for daily_return in daily_returns:
		average_return+=daily_return
	return average_return/len(daily_returns)

#INPUT: A list containing the daily return values for a given portfolio.
#OUTPUT: The standard deviation for the given portfolio values.
def get_standard_daily_return(daily_returns):
	return statistics.stdev(daily_returns)

#INPUT: A list of the daily returns.
#OUTPUT: Computes the Sharpe ratio assuming a 0% daily risk free rate (since eoconomy is currently doing well)
def get_sharpe_ratio(daily_returns):
	return (get_average_daily_return(daily_returns))/(get_standard_daily_return(daily_returns))\

#INPUT:  (1) Number of samples (252 for daily, 52 for weekly, 12 for monthly), (2) sharpe ratio
#OUTPUT: The annualized Sharpe ratio.
def get_annualized_sharpe_ratio(number_samples, sharpe_ratio):
	return math.sqrt(number_samples)*sharpe_ratio

#INPUT: Takes in an array of allocations for the stocks in the given portfolio.
#OUTPUT: Returns the sharpe ratio of the portfolio in question.
def negative_sharpe_ratio_from_allocations(allocations):
	prices = get_price_data(portfolio_stock_names)
	opening_day_prices = first_day_opening_prices(portfolio_stock_names)
	normed_matrix = get_normed_vector(prices)
	allocated_matrix = np.multiply(normed_matrix, allocations[0])
	position_values = np.multiply(allocated_matrix, 1000)
	portfolio_values = position_values.sum(axis=1)

	daily_returns = get_daily_returns(portfolio_values)
	cumulative_return = get_cumulative_return(portfolio_values)
	average_daily_return = get_average_daily_return(daily_returns)
	standard_daily_return = get_standard_daily_return(daily_returns)
	sharpe_ratio = get_sharpe_ratio(daily_returns)

	return -1*sharpe_ratio

#INPUT: A list of the current allocations that have been settled on.
#OUTPUT: Returns the constraint x[0]+x[1]+...x[n]-1=0
def allocations_constraint(allocations):
	total = 0;
	for allocation in allocations:
		total+=allocation
	total-=1
	return total

#INPUT: 
#OUTPUT: 
def optimize_allocations(opening_day_prices):
	allocations_guess = get_almost_equal_allocations(initial_investment, len(portfolio_stock_names), opening_day_prices)[0]
	allocations_guess = np.asarray(allocations_guess)
	allocation_bounds = ((0,1), (0,1), (0,1), (0,1))
	allocation_constraint = [{'type':'eq', 'fun': allocations_constraint}]
	optimal_sharpe_ratio_allocations = spo.minimize(negative_sharpe_ratio_from_allocations, allocations_guess, method='Nelder-Mead', bounds=allocation_bounds, constraints=allocations_constraint ,options={'disp': True})
	#print("Optimal Allocations for maximizing the sharpe ratio: " + str(optimal_sharpe_ratio_allocations.x))
	return optimal_sharpe_ratio_allocations.x

#INPUT:
#OUTPUT:
def display_optimized_portfolio_values():
	prices = get_price_data(portfolio_stock_names)
	opening_day_prices = first_day_opening_prices(portfolio_stock_names)
	normed_matrix = get_normed_vector(prices)
	optimized_allocations = optimize_allocations(opening_day_prices)
	allocated_matrix = np.multiply(normed_matrix, optimized_allocations)
	position_values = np.multiply(allocated_matrix, 1000)
	portfolio_values = position_values.sum(axis=1)
	plt.title("Optimized Portfolio Values Over Time")
	display_portfolio_value_over_time(portfolio_values)


plot_price_data(portfolio_stock_names)
print()
prices = get_price_data(portfolio_stock_names)
print("Total Stock Price Data: " + str(prices))
print()

opening_day_prices = first_day_opening_prices(portfolio_stock_names)
allocations = get_almost_equal_allocations(initial_investment, len(portfolio_stock_names), opening_day_prices)
normed_matrix = get_normed_vector(prices)
allocated_matrix = np.multiply(normed_matrix, allocations[0])
position_values = np.multiply(allocated_matrix, 1000)
portfolio_values = position_values.sum(axis=1)

daily_returns = get_daily_returns(portfolio_values)
cumulative_return = get_cumulative_return(portfolio_values)
average_daily_return = get_average_daily_return(daily_returns)
standard_daily_return = get_standard_daily_return(daily_returns)
sharpe_ratio = get_sharpe_ratio(daily_returns)
annualized_sharpe_ratio = get_annualized_sharpe_ratio(252, sharpe_ratio)

print("Normed Matrix: " + str(get_normed_vector(prices)))
print()
print("Allocated Matrix: " + str(np.multiply(normed_matrix, allocations[0])))
print()
print("Position Values Matrix: " + str(np.multiply(allocated_matrix, 1000)))
print()
print("Portfolio Values Matrix: " + str(position_values.sum(axis=1)))
print()
print("Daily Returns: " + str(get_daily_returns(portfolio_values)))
print()
print("Cumulative Return: " + str(get_cumulative_return(portfolio_values)))
print()
print("Average Daily Return: " + str(get_average_daily_return(daily_returns)))
print()
print("Standard Daily Return: " + str(get_standard_daily_return(daily_returns)))
print()
print("Sharpe Ratio: " + str(get_sharpe_ratio(daily_returns)))
print()
print("Annualized Sharpe Ratio: " + str(get_annualized_sharpe_ratio(252, sharpe_ratio)))
print()

display_portfolio_value_over_time(portfolio_values)
optimize_allocations(first_day_opening_prices(portfolio_stock_names))

display_optimized_portfolio_values()


