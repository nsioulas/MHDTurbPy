"""
Module providing Superposed epoch analysis functionality.

"""


import pandas as pd
import numpy as np
from joblib import Parallel, delayed



def threshold_condition_check(df,
                              condit_variables,
                              conditions,
                              threshold_values):
    """
    This function takes a dataframe and checks for conditions on certain variables and returns the indices of the rows where all conditions are met.
    Inputs:
        df: Dataframe containing the data to be checked
        condit_variables: List of strings containing the names of the variables to be checked
        conditions: List of lists of strings containing the conditions to be checked for each variable. Each inner list corresponds to the conditions for a specific variable.
        threshold_values: List of lists of values containing the threshold values for the conditions. Each inner list corresponds to the threshold values for a specific variable.
    Output:
        results: A dictionary where the key is a tuple of the condition variable, condition, and threshold value, and the value is a dataframe of the rows where the condition was met
    """
    # Turn to float just in case
    df = df.astype(float)
    results = {}

    # Iterate over condition variables, conditions, and threshold values
    for i, variable in enumerate(condit_variables):
        for condition in conditions[i]:
            for threshold_value in threshold_values[i]:
                print(threshold_value)

                # Create a copy of the dataframe for this specific check
                df_check = df.copy()

                # Create target column and initialize to False
                df_check['target'] = False

                # Evaluate condition
                if condition == ">=":
                    df_check.loc[(df_check[variable] >= threshold_value), 'target'] = True
                elif condition == "<=":
                    df_check.loc[(df_check[variable] <= threshold_value), 'target'] = True
                elif condition == "==":
                    df_check.loc[(df_check[variable] == threshold_value), 'target'] = True
                elif condition == "!=":
                    df_check.loc[(df_check[variable] != threshold_value), 'target'] = True
                else:
                    raise ValueError(f"Invalid condition: {condition}")

                # Add the results to the dictionary
                results[(variable, condition, threshold_value)] = df_check[df_check['target']]

    return results



def SEA(df, condit_variables, threshold_values, conditions, time_around, which_one, mean_or_median, resample_rate):
    """
    Same description as previous function
    """
    
    # Get dictionary of results for all threshold values
    results_dict = threshold_condition_check(df, condit_variables, conditions, threshold_values)

    # Create a dictionary to store results for each threshold value
    SEA_results = {}

    # Loop over each threshold value
    for key, result_df in results_dict.items():

        # Find indices of events in the dataframe
        event_indices = result_df.index.values

        # Convert time_around to a Timedelta object
        time_around_timedelta = pd.Timedelta(time_around)

        # Calculate start and end times for each event
        start_times = event_indices - time_around_timedelta
        end_times = event_indices + time_around_timedelta

        # Filter out events that fall outside of the time range of the dataframe
        valid_indices = np.logical_and(event_indices > (df.index[0] + time_around_timedelta), event_indices < (df.index[-1] - time_around_timedelta))

        # Find closest indices to start and end times
        start_indices = df.index.get_indexer(start_times[valid_indices], method='nearest')
        end_indices = df.index.get_indexer(end_times[valid_indices], method='nearest')

        if keep_vals := [
            list(df[which_one].iloc[start:end].values)
            for start, end in zip(start_indices, end_indices)
        ]:
            yvals = np.nanmedian(keep_vals,axis=0) if mean_or_median == 'median' else np.nanmean(keep_vals,axis=0)
            
            # To resample dataframe to specified cadence
            start_time = df.index[0] - pd.Timedelta(time_around)
            end_time = df.index[0] + pd.Timedelta(time_around)

            time_index = pd.date_range(start_time, end_time, periods=len(yvals))

            data = {'yvals': yvals}

            fin_df = pd.DataFrame(data, index=time_index)
            fin_df = fin_df.resample(resample_rate).mean()
            duration = (fin_df.index[-1] - fin_df.index[0])
            dur_sec = duration / np.timedelta64(1, 's') / 2

            # Final x, y values
            yvals = fin_df['yvals'].values
            xvals = np.linspace(-dur_sec, dur_sec, len(yvals))

            # Also estimate std
            y_std = np.nanstd(keep_vals,axis=0)
            # Estimate standard error of the mean
            y_std_err_mean = y_std/np.sqrt(np.shape(keep_vals)[0])

            # Store the result for this threshold value
            SEA_results[key] = (xvals, yvals, y_std, y_std_err_mean)

    return SEA_results

