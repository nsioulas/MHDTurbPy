"""
Module providing Superposed epoch analysis functionality.

"""


import pandas as pd
import numpy as np

def threshold_condition_check(df, condit_variables, conditions, threshold_values):
    """
    This function takes a dataframe and checks for conditions on certain variables and returns the indices of the rows where all conditions are met.
    Inputs:
        df: Dataframe containing the data to be checked
        condit_variables: List of strings containing the names of the variables to be checked
        conditions: List of lists of strings containing the conditions to be checked for each variable. Each inner list corresponds to the conditions for a specific variable.
        threshold_values: List of lists of values containing the threshold values for the conditions. Each inner list corresponds to the threshold values for a specific variable.
    Output:
        target_rows: A list of indices of the rows where all conditions are met
    """
    # Turn to float just in case
    df = df.astype(float)

    # Create target column and initialize to 0
    df['target'] = 0

    # Initialize count column and set to 0
    df['count'] = 0

    # Iterate over condition variables and threshold values
    for i, variable in enumerate(condit_variables):
        # Get condition and threshold value
        for j,condition in enumerate(conditions[i]):
            threshold_value = threshold_values[i][j]

            # Evaluate condition
            if condition == ">=":
                df.loc[(df[variable] >= threshold_value), 'count'] += 1
            elif condition == "<=":
                df.loc[(df[variable] <= threshold_value), 'count'] += 1
            elif condition == "==":
                df.loc[(df[variable] == threshold_value), 'count'] += 1
            elif condition == "!=":
                df.loc[(df[variable] != threshold_value), 'count'] += 1
            else:
                raise ValueError(f"Invalid condition: {condition}")
    

    # Get indices of all rows that meet all conditions
    rows, columns = np.array(conditions).shape
    total_elements = rows * columns

    # Assign value of 1 
    df.loc[df['count'] == total_elements, 'target'] = 1
    
    return df[df['target'] == 1].index.values, df[df['target'] == 1]


def SEA(df, condit_variables, threshold_values, conditions, time_around, which_one, mean_or_median, resample_rate):
    """
    This function performs a superposed epoch analysis (SEA) on a given dataframe, df.
    The function finds events in the dataframe based on user-specified condition variables, threshold values, and conditions.
    The function then extracts a certain variable (specified by the user) around the identified events, and resamples the data to a user-specified rate.
    
    Parameters:
    - df (pd.DataFrame): The dataframe to perform the SEA on.
    
    - condit_variables (list of str): A list of column names in df that the conditions will be based on.
    
    - threshold_values (list of numbers): A list of threshold values for each of the condition variables.
    
    - conditions (list of str): A list of conditions that will be applied to each of the condition variables.
    - time_around (str): The amount of time before and after an event to extract data for.
    - which_one (str): The column name of the variable to extract data for.
    
    - mean_or_median (bool): Specifies whether to take the mean or median of the extracted data.
    
   -  resample_rate (str): The rate to resample the extracted data to.
    
    Returns:
    tuple: A tuple of two numpy arrays, xvals and yvals, where xvals are the time values and yvals are the extracted variable values.
    """
    
    # Find indices of events in the dataframe
    event_indices,_        = threshold_condition_check(df, condit_variables, conditions, threshold_values)

    # Convert time_around to a Timedelta object
    time_around_timedelta = pd.Timedelta(time_around)

    # Calculate start and end times for each event
    start_times           = event_indices - time_around_timedelta
    end_times             = event_indices + time_around_timedelta

    # Filter out events that fall outside of the time range of the dataframe
    valid_indices         = np.logical_and(event_indices > (df.index[0] + time_around_timedelta), event_indices < (df.index[-1] - time_around_timedelta))

    # Find closest indices to start and end times
    start_indices         = df.index.get_indexer(start_times[valid_indices], method='nearest')
    end_indices           = df.index.get_indexer(end_times[valid_indices], method='nearest')

    if keep_vals := [
        list(df[which_one].iloc[start:end].values)
        for start, end in zip(start_indices, end_indices)
    ]:
        yvals = np.nanmedian(keep_vals,axis=0)
        # To resample dataframe to specifiec cadence
        start_time = df.index[0] - pd.Timedelta(time_around)
        end_time   = df.index[0] + pd.Timedelta(time_around)


        time_index = pd.date_range(start_time, end_time, periods=len(yvals))


        data = {'yvals': yvals}

        fin_df = pd.DataFrame(data, index=time_index)
        fin_df     = fin_df.resample(resample_rate).mean()
        duration   = (fin_df.index[-1] - fin_df.index[0])
        dur_sec    = duration / np.timedelta64(1, 's') / 2

        # Final x, y values
        yvals      = fin_df['yvals'].values
        xvals      = np.linspace(-dur_sec, dur_sec, len(yvals))

        # Also estimate std
        y_std = np.nanstd(keep_vals,axis=0)
        # Estimate standard error of the mean
        y_std_err_mean = y_std/np.sqrt(np.shape(keep_vals)[0])

    return xvals, yvals, y_std, y_std_err_mean 