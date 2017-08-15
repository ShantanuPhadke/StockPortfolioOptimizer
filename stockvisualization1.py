import pandas as pd
import matplotlib.pyplot as plt

def print_entire_csv(dataframe):
	print(dataframe)

def print_first_five_rows(dataframe):
	print(dataframe.head(5))

def get_mean_volume(dataframe):
	return dataframe['Volume'].mean()

def plot_stock_high_data(dataframe):
	dataframe['High'].plot()
	plt.show()

veeva_dataframe = pd.read_csv("VEEV.csv")

print_entire_csv(veeva_dataframe)
print()
print_first_five_rows(veeva_dataframe)
print()
print( "Average Volume of Veeva Stock: " + str(get_mean_volume(veeva_dataframe)) )

plot_stock_high_data(veeva_dataframe)