#!/usr/bin/env python
# coding: utf-8

# # Question 1: Car Matrix Generation

# In[32]:


import pandas as pd
import numpy as np

# Reading the csv file and storing the data in a pandas dataframe named dataset
dataset = pd.read_csv("C:/Users/Ankita.KM/Downloads/dataset-1.csv")

def generate_car_matrix(dataset):

    # Pivot the DataFrame to create a car matrix
    car_matrix = dataset.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0 using np.fill_diagonal
    np.fill_diagonal(car_matrix.values, 0)

    return car_matrix

# Call the function with the file path
result_matrix = generate_car_matrix(dataset)

# Print the result
print(result_matrix)


# # Question 2: Car Type Count Calculation

# In[33]:


df = pd.read_csv(r'C:\Users\Ankita.KM\Downloads\dataset-1.csv')

def get_type_count(df):
    # Add a new categorical column 'car_type' based on 'car' values
    df['car_type'] = pd.cut(df['car'], bins=[float('-inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Calculate the count of occurrences for each 'car_type' category
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_count = dict(sorted(type_count.items()))

    return sorted_type_count

# Example usage
# Assuming df is your DataFrame obtained from dataset-1.csv
# df = pd.read_csv('path/to/dataset-1.csv')

# Call the function with the DataFrame
result = get_type_count(df)
print(result)


# # Question 3: Bus Count Index Retrieval

# In[34]:


def get_bus_indexes(df):
    # Calculate the mean value of the 'bus' column
    mean_bus_value = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

# Example usage
# Assuming df is your DataFrame obtained from dataset-1.csv
# df = pd.read_csv('path/to/dataset-1.csv')

# Call the function with the DataFrame
result = get_bus_indexes(df)
print(result)


# # Question 4: Route Filtering

# In[12]:


def filter_routes(df):
    # Calculate the average value of the 'truck' column for each route
    average_truck_values = df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' values is greater than 7
    selected_routes = average_truck_values[average_truck_values > 7].index.tolist()

    # Sort the list of selected routes
    selected_routes.sort()

    return selected_routes

# Example usage
# Assuming df is your DataFrame obtained from dataset-1.csv
# df = pd.read_csv('path/to/dataset-1.csv')

# Call the function with the DataFrame
result = filter_routes(df)
print(result)


# # Question 5: Matrix Value Modification

# In[35]:


def multiply_matrix(input_matrix):
    # Copy the input matrix to avoid modifying the original DataFrame
    modified_matrix = input_matrix.copy()

    # Apply the specified logic to modify values
    modified_matrix = modified_matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

# Example usage
# Assuming result_matrix is the DataFrame obtained from Question 1
# result_matrix = generate_car_matrix(df)

# Call the function with the resulting DataFrame
modified_result = multiply_matrix(result_matrix)
print(modified_result)


# # Question 6: Time Check

# In[36]:


from itertools import product

def check_timestamp_completeness(df):
    # Combine 'startDay' and 'startTime' columns to create a 'start_timestamp' column
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')

    # Combine 'endDay' and 'endTime' columns to create an 'end_timestamp' column
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')

    # Remove rows with invalid timestamps
    df = df.dropna(subset=['start_timestamp', 'end_timestamp'])

    # Create a MultiIndex for (id, id_2) pairs
    multi_index = pd.MultiIndex.from_frame(df[['id', 'id_2']])

    # Initialize a boolean series with True for all pairs
    completeness_series = pd.Series(True, index=multi_index)

    # Iterate over all pairs of (id, id_2) and days of the week
    for (id_val, id_2_val), day_of_week in product(df[['id', 'id_2']].drop_duplicates().values, range(7)):
        # Filter the DataFrame for the current pair and day of the week
        subset_df = df[(df['id'] == id_val) & (df['id_2'] == id_2_val) & (df['start_timestamp'].dt.dayofweek == day_of_week)]

        # Check if there is any overlap between timestamps and the 24-hour period
        has_overlap = any(
            ((subset_df['start_timestamp'] <= pd.Timestamp(f'2023-01-0{day_of_week + 1} 23:59:59')) &
             (subset_df['end_timestamp'] >= pd.Timestamp(f'2023-01-0{day_of_week} 00:00:00')))
        )

        # Update the completeness_series accordingly
        completeness_series.loc[(id_val, id_2_val)] = completeness_series.loc[(id_val, id_2_val)] and has_overlap

    return completeness_series

# Example usage
# Assuming df is your DataFrame obtained from dataset-2.csv
df = pd.read_csv(r'C:\Users\Ankita.KM\Downloads\dataset-2.csv')

# Call the function with the DataFrame
result_series = check_timestamp_completeness(df)
print(result_series)

