import pandas as pd
import numpy as np
import haversine
import multiprocess

N_THREADS = 8

def haversine_distance(x):
    a_lat, a_lon, b_lat, b_lon = x_test
    return haversine.haversine((a_lat, a_lon), (b_lat, b_lon))

def apply_multithreaded(data, func):
    pool = multiprocess.Pool(N_THREADS)
    data = data.values
    result = pool.map(func, data)
    pool.close()
    return result

def preprocess_data(df):
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].apply(lambda x: 1 if x == "Y" else 0)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
    df["day_of_week"] = [t.dayofweek for t in df["pickup_datetime"]]

    # create distance features
    df["dist_l1"] = np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) + np.abs(df["pickup_longitude"] - df["dropoff_longitude"])
    df["dist_l2"] = np.sqrt((df["pickup_latitude"] - df["dropoff_latitude"]) ** 2 + (df["pickup_longitude"] - df['dropoff_longitude']) ** 2)
    df["dist_haversine"] = apply_multithreaded(df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']], haversine_distance)

    # create direction of travel features
    df["delta_lat"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["delta_long"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["angle"] = (180 / np.pi) * np.arctan2(df["delta_lat"], df["delta_long"]) + 180

    # creating traffic features
    df['day'] = df['pickup_datetime'].dt.date
    df['hour'] = df['pickup_datetime'].dt.hour
    daily_traffic = df.groupby('day')['day'].count()  # Count the number of trips on each day
    hourly_traffic = df.groupby('hour')['hour'].count()  # Count the number of trips in each hour
    df['daily_count'] = df['day'].apply(lambda day: daily_traffic[day])
    df['hourly_count'] = df['hour'].apply(lambda hour: hourly_traffic[hour])

    # creating time estimate features
    df['haversine_speed_km/s'] = df['dist_haversine'] / (df['trip_duration'] / 3600)  # Calculate haversine speed for training set
    hourly_speed = df.groupby('hour')['haversine_speed'].mean()  # Find average haversine_speed for each hour in the training set
    hourly_speed_fill = df['haversine_speed'].mean()  # Get mean across whole dataset for filling unknowns
    df_hourly_speed = df['hour'].apply(lambda hour: hourly_speed[hour] if hour in hourly_speed else hourly_speed_fill)
    df['haversine_speed_estim_km/h'] = df['dist_haversine'] / df_hourly_speed

    return df

if __name__ == "__main__":    
    print('Reading data into memory ...')

    # Read in the input dataset
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
    df_test["trip_duration"] = -1
    print('Read {} training data and {} testing data, preprocessing ...'.format(df_train.shape, df_test.shape))

    # preprocess data
    print('Starting preprocess data ...')
    df_full = pd.concat([df_train, df_test], axis=0)
    df_preprocessed = preprocess_data(df_full)

    # split to new_train and new_test
    train_len = len(df_train)
    new_train = df_preprocessed.iloc[:train_len, :]
    new_test = df_preprocessed.iloc[train_len:, :]
    
    # dropping columns for both train and test
    new_train.drop(["id","pickup_datetime","dropoff_datetime","day"], axis=1, inplace=True)
    new_test.drop(["pickup_datetime","dropoff_datetime","day","trip_duration"], axis=1, inplace=True)

    # saving df train and test to csv
    new_train.to_csv("../input/train_preprocessed.csv", index=False)
    print("Train data saved!")
    new_test.to_csv("../input/test_preprocessed.csv", index=False)
    print("Test data saved!") 

