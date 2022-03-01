
import pandas as pd
import nasdaqdatalink
import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, List, NamedTuple
from collections import defaultdict
import math
from math import sqrt
from collections import Counter

def return_nasdaq_data():
  stock_symbol = input("Please enter NASDAQ stock symbol: ")
  raw_data = get_nasdaq_data(stock_symbol)
  stock_df_from_csv = open_stock_csv(raw_data)
  return stock_df_from_csv

# Grab Microsoft Data
def get_nasdaq_data(stock_symbol):
    # WIKI/TICKER.8 = "WIKI/MSFT - Adj. Open"
    # WIKI/TICKER.9 = "WIKI/MSFT - Adj. High"
    # WIKI/TICKER.10 = "WIKI/MSFT - Adj. Low"
    # WIKI/TICKER.11 = "WIKI/MSFT - Adj. Close"
    # WIKI/TICKER.12 = "WIKI/MSFT - Adj. Volume"
    tick_8 = "WIKI/" + stock_symbol + ".8"
    tick_9 = "WIKI/" + stock_symbol + ".9"
    tick_10 = "WIKI/" + stock_symbol + ".10"
    tick_11 = "WIKI/" + stock_symbol + ".11"
    tick_12 = "WIKI/" + stock_symbol + ".12"
    # WIKI/TICKER.8 = "WIKI/MSFT - Adj. Open"
    # WIKI/TICKER.9 = "WIKI/MSFT - Adj. High"
    # WIKI/TICKER.10 = "WIKI/MSFT - Adj. Low"
    # WIKI/TICKER.11 = "WIKI/MSFT - Adj. Close"
    # WIKI/TICKER.12 = "WIKI/MSFT - Adj. Volume"
    column_8 = "WIKI/" + stock_symbol + " - Adj. Open"
    column_9 = "WIKI/" + stock_symbol + " - Adj. High"
    column_10 = "WIKI/" + stock_symbol + " - Adj. Low"
    column_11 = "WIKI/" + stock_symbol + " - Adj. Close"
    column_12 = "WIKI/" + stock_symbol + " - Adj. Volume"
    nasdaq_columns = [column_8,column_9,column_10,column_11,column_12]
    # get_nasdaq_data():
    # original call
    # data = nasdaqdatalink.get(['WIKI/MSFT.8','WIKI/MSFT.9','WIKI/MSFT.10','WIKI/MSFT.11','WIKI/MSFT.12'], start_date="2017-01-01", end_date="2017-12-31", api_key="j_GhevzmLxJxYMy8n8gr")
    nasdaq_data = nasdaqdatalink.get([tick_8,tick_9,tick_10,tick_11,tick_12], start_date="2017-01-01", end_date="2017-12-31", api_key="j_GhevzmLxJxYMy8n8gr")
    stock_df = pd.DataFrame(nasdaq_data, columns= [column_8,column_9,column_10,column_11,column_12])
    print(stock_df)
    csv_name = "./" + stock_symbol + "_datafame.csv"
    # stock_df.to_csv (r'./msft_dataframe.csv', index = True, header=True)
    stock_df.to_csv(csv_name, index = True, header=True)
    return csv_name, nasdaq_columns

def open_stock_csv(csv_name):
  # ms_df = pd.read_csv(r'./msft_dataframe.csv')
  nasdaq_stock_df = pd.read_csv(csv_name)
  return nasdaq_stock_df

def open_msft_csv():
    nasdaq_columns = ["WIKI/MSFT - Adj. Open", "WIKI/MSFT - Adj. High",
                      "WIKI/MSFT - Adj. Low", "WIKI/MSFT - Adj. Close",
                      "WIKI/MSFT - Adj. Volume"]
    ms_df = pd.read_csv(r'./msft_dataframe.csv')
    return ms_df, nasdaq_columns

def correct_df_labels(stock_data_df, nasdaq_columns):
  stock_data_df.rename(columns={nasdaq_columns[0]: 'open',
             nasdaq_columns[1]: 'high',
             nasdaq_columns[2]: 'low',
             nasdaq_columns[3]: 'close',
             nasdaq_columns[4]: 'volume'}, inplace=True)
  return stock_data_df

def add_diffs(df_labeled):
  """ Creates the following columns:
  'rise-drop' =  Label for day (open to close)
  'hl_diff' = ( High - Low )
  'oc_diff' = ( Open - Close )
  'ol_diff' = ( Open - Low )
  'oh_diff' = ( Open - High )
  'ch_diff' = ( Close - High )
  'cl_diff' = ( Close - Low )
  'no_diff = 1 in every row for Printing Baseline Comparisons
  """
  # Compute High - Low Comparison; create 'rise-drop' column(s)
  high_low_list = list()
  for x in range(250):
    if (df_labeled['open'][x] >= df_labeled['close'][x]):
      today_rise_drop = 'drop'
    else:
      today_rise_drop = 'rise'
    high_low_list.append(today_rise_drop)
  df_labeled['rise-drop'] = high_low_list

  # Compute High - Low Comparison; create 'hl_diff' column
  high_low_diff = list()
  for x in range(250):
    day_hl_diff = df_labeled['high'][x] - df_labeled['low'][x]
    high_low_diff.append(day_hl_diff)
  df_labeled['hl_diff'] = high_low_diff

  # Compute Open - Close Comparison; create 'oc_diff' column
  open_close_diff = list()
  for x in range(250):
    day_oc_diff = df_labeled['open'][x] - df_labeled['close'][x]
    open_close_diff.append(day_oc_diff)
  df_labeled['oc_diff'] = open_close_diff

  # Compute Open - Low Comparison; create 'ol_diff' column
  open_low_diff = list()
  for x in range(250):
    open_ol_diff = df_labeled['open'][x] - df_labeled['low'][x]
    open_low_diff.append(open_ol_diff)
  df_labeled['ol_diff'] = open_low_diff

  # Compute Open - High Comparison; create 'oh_diff' column
  open_high_diff = list()
  for x in range(250):
    open_oh_diff = df_labeled['open'][x] - df_labeled['high'][x]
    open_high_diff.append(open_oh_diff)
  df_labeled['oh_diff'] = open_high_diff

  # Compute Close - High Comparison; create 'ch_diff' column
  close_high_diff = list()
  for x in range(250):
    day_ch_diff = df_labeled['close'][x] - df_labeled['high'][x]
    close_high_diff.append(day_ch_diff)
  df_labeled['ch_diff'] = close_high_diff

  # Compute Close - Low Comparison; create 'cl_diff' column
  close_low_diff = list()
  for x in range(250):
    day_cl_diff = df_labeled['close'][x] - df_labeled['low'][x]
    close_low_diff.append(day_cl_diff)
  df_labeled['cl_diff'] = close_low_diff

  # All Ones - For Printing Baseline Comparisons 'no_diff' column
  # create all ones in every row of the column
  all_one = 1.0
  all_ones = list()
  for x in range(250):
    all_one = all_one + 1
    all_ones.append(all_one)
  df_labeled['no_diff'] = all_ones

  return df_labeled

def predict_nn_label(neighbors_list, neighbor_num):
  print("\n\tComputing Label for Neighbors")
  # receives sorted list
  #neighbors = get_neighbor_mod(train_data, test_row[0], num_neighbors)
  nn_list = list()
  for x in range(neighbor_num):
    if x == 0:
      pass
    nn_d = neighbors_list[0][x]
    nn_list.append(nn_d)
  # print("The ", neighbor_num, "nearest neighbors are", nn_list)
  # get labels from nn_list
  n_rise = 0 # rise
  n_drop = 0 # drop
  for each in range(1,neighbor_num):
    if nn_list[each][1][1] == 'rise':
      n_rise += 1
    else:
      n_drop += 1
  if n_rise > n_drop:
    prediction = 'rise'
  else:
    prediction = 'drop'
  print("The predicion is ", prediction)
  return prediction

# neighbors = get_neighbor_mod(test_data_ltd, test_day_row, num_neighbors, run_day)
def get_neighbor_mod(data_history, single_day, num_neighbors, runday):
  """ data_history needs to have the day_number, and the metric being tested
      single_day should be the row (from data_history)
      eventually, we will need to know the day_numbers which generated the label
      maybe we return a tuple of numbers like we did nasdaq labels
  """
  print(single_day[0], type(single_day))


  distances = list()
  # First we need to understand that we cannot compare future data. We can
  # only compare historical (up to the day which is being compared)
  # for each element in data_history (full dataset):
  for each in range(runday):
    # line below replaces call to: knn(each):
    dist = sqrt((float(single_day[0]) - float(data_history[each][0])) ** 2)
    # print("\nConfirm Computation of Euclidean Distances")
    # print("E_Distance from {:,} to {:,} is: {:,}".format(int(test_row), int(each), int(dist)))
    distances.append([dist,data_history[each]])
  print("\nDistances list and label before sorting")
  for each in distances:
    print(each)
  distances.sort(key=lambda tup: tup[0])
  print("\nDistances List and Label after sorting")
  for each in distances:
    print(each)
  neighbors = list()
  for i in range(num_neighbors):
    neighbors.append(distances)
  return neighbors


def run_data_diffs(stock_data_df):
  """ incoming dataframe will have the following columns
  'open', 'close', 'high', 'low', volume', 'rise-drop', 'hl_diff',
  'oc_diff', 'ol_diff', 'oh_diff', 'ch_diff', 'cl_diff', 'no_diff
  """
  print("\n\t\tFirst, please select a day to analyze: ")
  run_day = int(input("\t\t... select between 1 - 247: "))
  runday = 250
  if run_day > 1 and run_day < 247:
    runday = run_day + 2
  print("\n\t\tFor this day, the stock results are as follows:")
  print("\t\t\tOpen Price: {:.2f}".format(stock_data_df['open'][runday]))
  print("\t\t\tClose Price: {:.2f}".format(stock_data_df['close'][runday]))
  print("\t\t\tDaily High: {:.2f}".format(stock_data_df['high'][runday]))
  print("\t\t\tDaily Low: {:.2f}".format(stock_data_df['low'][runday]))
  print("\t\t\tDaily Volume: {:,}".format(stock_data_df['volume'][runday]))
  print("\t\t\tDay Performance: {}".format(stock_data_df['rise-drop'][runday]))
  # Next exclude columns from the dataset which are not being analyzed.
  # If we return the day_# (as a tuple) we can send in a two-column set
  print("\n\t\tNext, select a metric to compute: ")
  print("\t\t1: high_low, 2: open-close, 3: open-low 4: open-high")
  print("\t\t5: close-low, 6: close-high, 7: daily-volume")
  knn_metric = int(input("\t\t>>> "))
  knn_metrics = ['high_low','open-close','open-low','open-high','close-low','close-high','volume']
  test_data_ltd = pd.DataFrame(stock_data_df, columns=[knn_metrics[knn_metric - 1],'rise-drop'])
  test_day_row = test_data_ltd.loc[run_day]
  # Then convert to list and limit the data sent to knn function, as historical data.
  # this means test_dataset will be the main list, up to but not including runday
  test_dataset = list()
  for x in range(runday):
    test_dataset.append(test_data_ltd.loc[x])
  print(test_dataset)
  print("\n\t\tFinally, select how many neighbors to measure: ")
  num_neighbors = int(input("\t\t 1, 2, 3... 5 etc. >>> "))
  neighbors = get_neighbor_mod(test_dataset, test_day_row, num_neighbors, run_day)
  prediction = predict_nn_label(neighbors, num_neighbors)
  print("KNN Predicted: ", prediction, " Actual next day value = ", test_data_ltd.loc[runday]['rise-drop'])

  # select_data = pd.DataFrame(data_diffs, columns=['oc_diff','hl_diff'])
  # #print(select_data.info)
  # run_data = select_data[:].values
  # v0_data = pd.DataFrame(data_diffs, columns=['volume'])
  # run_v0 = select_data[:].values
  # run_volume = select_data[:].values
  # #print("\n\tLooking at Row Data")
  # #print(run_data)
  # #neighbors = get_neighbors()
  # #print("\n\tLooking at Distance Between Rows")
  # #knn(run_data)
  #
  # # make a selection of the data (columns volume and next-label) to analyze
  # volume_data = pd.DataFrame(data_diffs, columns=['volume','next'])
  # # strip out the column names using .values
  # run_vol = volume_data[:].values
  # # chunk out a selection of the rows for training data set
  # train_data = run_vol[1:20]
  # # identify the 'test-against' row
  # test_row = train_data[0]
  # # set the number of neighbors to analyze against
  # num_neighbors = 5
  # # call the nearest neighbors function
  # neighbors = get_neighbor_mod(train_data, test_row[0], num_neighbors)


def main():
  print("\n\tWelcome to Rise-Fall NASDAQ App")
  print("\tThis app uses K-Nearset Neighbor Algorithm to predict the next day stock performance.")
  print("\n\tOptions:")
  print("\t1) Use MSFT - Microsoft 2017 Sample Data")
  print("\t2) Enter other NASDAQ Symbol")
  select = int(input("\t>> "))
  if select == 1:
    stock_data_df, nasdaq_columns = open_msft_csv()
  else:
    stock_data_df, nasdaq_columns = return_nasdaq_data()
  # update labels from nasdaq strings for difference computations
  df_labeled = correct_df_labels(stock_data_df, nasdaq_columns)
  # add diff comparisons to dataset
  df_diffs = add_diffs(df_labeled)
  # run different kinds of analysis
  run_data_diffs(df_diffs)
  return

main()