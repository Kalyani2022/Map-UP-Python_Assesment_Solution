#!/usr/bin/env python
# coding: utf-8

# # Question 1: Distance Matrix Calculation

# In[28]:


import pandas as pd
import numpy as np

# Reading the csv file and storing the data in a pandas dataframe named dataset
dataset = pd.read_csv("C:/Users/Ankita.KM/Downloads/dataset-3.csv")

def calculate_distance_matrix(dataset):
    # Group by 'id_start' and 'id_end' and sum the distances
    grouped_distances = dataset.groupby(['id_start', 'id_end'])['distance'].sum().reset_index()

    # Pivot the table to create the distance matrix
    distance_matrix = grouped_distances.pivot(index='id_start', columns='id_end', values='distance').fillna(0)

    # Make the matrix symmetric by filling missing values
    distance_matrix = distance_matrix + distance_matrix.T.fillna(0)

    # Set diagonal values to 0
    np.fill_diagonal(distance_matrix.values, 0)

    # Replace NaN values with 0
    distance_matrix = distance_matrix.fillna(0)

    return distance_matrix

result_matrix = calculate_distance_matrix(dataset)
print(result_matrix)


# # Question 2: Unroll Distance Matrix

# In[30]:


import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Reset the index to convert the index to a column
    distance_matrix = distance_matrix.reset_index()

    # Melt the DataFrame to reshape it for the desired output
    unrolled_matrix = pd.melt(distance_matrix, id_vars='index', var_name='id_end', value_name='distance')

    # Rename the 'index' column to 'id_start'
    unrolled_matrix = unrolled_matrix.rename(columns={'index': 'id_start'})

    # Filter out rows where 'id_start' is equal to 'id_end'
    unrolled_matrix = unrolled_matrix[unrolled_matrix['id_start'] != unrolled_matrix['id_end']]

    # Reset the index for the final DataFrame
    unrolled_matrix = unrolled_matrix.reset_index(drop=True)

    return unrolled_matrix

# Example usage
# Assuming 'result_matrix' is the DataFrame obtained from the previous question
result_unrolled = unroll_distance_matrix(result_matrix)
print(result_unrolled)


# # Question 3: Finding IDs within Percentage Threshold

# In[31]:


def find_ids_within_ten_percentage_threshold(distance_matrix, reference_value):
    # Calculate the average distance for the reference value
    reference_avg_distance = distance_matrix[distance_matrix['id_start'] == reference_value]['distance'].mean()

    # Calculate the threshold values (10% above and below the average)
    lower_threshold = reference_avg_distance - 0.1 * reference_avg_distance
    upper_threshold = reference_avg_distance + 0.1 * reference_avg_distance

    # Filter the DataFrame based on the threshold values
    filtered_ids = distance_matrix[(distance_matrix['distance'] >= lower_threshold) & (distance_matrix['distance'] <= upper_threshold)]

    # Get unique values from the 'id_start' column and sort them
    result_ids = sorted(filtered_ids['id_start'].unique())

    return result_ids

# Example usage
# Assuming 'result_unrolled' is the DataFrame obtained from the previous question and reference_value is some integer
reference_value = 1001400 # Replace this with the actual reference value
result_within_threshold = find_ids_within_ten_percentage_threshold(result_unrolled, reference_value)

print(result_within_threshold)


# # Question 3: Finding IDs within Percentage Threshold

# In[32]:


def find_ids_within_ten_percentage_threshold(distance_matrix, reference_value):
    # Filter DataFrame based on the reference_value
    reference_df = distance_matrix[distance_matrix['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    reference_avg_distance = reference_df['distance'].mean()

    # Calculate the threshold for 10% (including ceiling and floor)
    threshold_low = reference_avg_distance - 0.1 * reference_avg_distance
    threshold_high = reference_avg_distance + 0.1 * reference_avg_distance

    # Filter IDs within the 10% threshold
    result_df = distance_matrix[
        (distance_matrix['id_start'] != reference_value) &  # Exclude the reference value itself
        (distance_matrix['distance'] >= threshold_low) &
        (distance_matrix['distance'] <= threshold_high)
    ]

    # Get the unique sorted list of values from the id_start column
    sorted_ids_within_threshold = sorted(result_df['id_start'].unique())

    return sorted_ids_within_threshold

# Example usage
# Assuming 'result_unrolled' is the DataFrame obtained from the previous question
reference_value = 1001400  # Replace this with the specific reference value you want to use
ids_within_threshold = find_ids_within_ten_percentage_threshold(result_unrolled, reference_value)

print(ids_within_threshold)


# # Question 4: Calculate Toll Rates

# In[33]:


def calculate_toll_rate(distance_matrix):
    # Check if 'distance' column exists and is numeric
    if 'distance' not in distance_matrix.columns or not pd.api.types.is_numeric_dtype(distance_matrix['distance']):
        raise ValueError("The 'distance' column is missing or not numeric.")

    # Check for NaN values in the 'distance' column
    if distance_matrix['distance'].isna().any():
        raise ValueError("There are NaN values in the 'distance' column.")

    # Create new columns for toll rates for each vehicle type
    distance_matrix['moto'] = 0.8 * distance_matrix['distance']
    distance_matrix['car'] = 1.2 * distance_matrix['distance']
    distance_matrix['rv'] = 1.5 * distance_matrix['distance']
    distance_matrix['bus'] = 2.2 * distance_matrix['distance']
    distance_matrix['truck'] = 3.6 * distance_matrix['distance']

    return distance_matrix

# Example usage
# Assuming 'result_unrolled' is the DataFrame obtained from the previous question
result_with_toll_rates = calculate_toll_rate(result_unrolled)

print(result_with_toll_rates)


# # Question 5: Calculate Time-Based Toll Rates

# In[43]:


import pandas as pd
import numpy as np

# Assuming the merged DataFrame is named 'merged_df'

def calculate_time_based_toll_rates(df):
    # Create a copy of the input DataFrame to avoid modifying the original
    dataset_copy = dataset.copy()

    # Create columns for start_day and end_day
    dataset_copy['start_day'] = dataset_copy['start_time'].dt.day_name()
    dataset_copy['end_day'] = dataset_copy['end_time'].dt.day_name()

    # Define time ranges and discount factors
    weekday_ranges = [
        (pd.to_datetime('00:00:00').time(), pd.to_datetime('10:00:00').time()),
        (pd.to_datetime('10:00:00').time(), pd.to_datetime('18:00:00').time()),
        (pd.to_datetime('18:00:00').time(), pd.to_datetime('23:59:59').time())
    ]

    weekend_ranges = [
        (pd.to_datetime('00:00:00').time(), pd.to_datetime('23:59:59').time())
    ]

    weekday_discounts = [0.8, 1.2, 0.8]
    weekend_discount = 0.7

    # Apply discounts based on time ranges
    for (start, end), discount in zip(weekday_ranges, weekday_discounts):
        weekday_condition = (
            (dataset_copy['start_day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])) &
            (dataset_copy['start_time'].dt.time >= start) & (dataset_copy['end_time'].dt.time <= end)
        )
        dataset_copy.loc[weekday_condition, ['moto', 'car', 'rv', 'bus', 'truck']] *= discount

    for (start, end) in weekend_ranges:
        weekend_condition = (
            (dataset_copy['start_day'].isin(['Saturday', 'Sunday'])) &
            (dataset_copy['start_time'].dt.time >= start) & (dataset_copy['end_time'].dt.time <= end)
        )
        dataset_copy.loc[weekend_condition, ['moto', 'car', 'rv', 'bus', 'truck']] *= weekend_discount

    return dataset_copy

merged_dataset = pd.merge(dataset-1, dataset-2, how='inner', on=['id_start', 'id_end'])
merged_dataset = pd.merge(merged_dataset, dataset-3, how='inner', on=['id_start', 'id_end'])

# Example usage
# Assuming 'merged_df' is the DataFrame obtained from merging the three datasets
result_with_time_based_rates = calculate_time_based_toll_rates(merged_dataset)

# Print the result
print(result_with_time_based_rates)


# In[ ]:




