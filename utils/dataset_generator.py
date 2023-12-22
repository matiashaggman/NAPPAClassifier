import os
import pandas as pd
import numpy as np

from utils.dataset_classes import NappaDataset, SleepRecording

def read_and_process_hypnogram(hypno_path):
    
    """
    Reads and processes a hypnogram file.
    
    Args:
        hypno_path (str): The file path to the hypnogram file.
    
    Returns:
        pd.DataFrame: A DataFrame with processed datetime and sleep stage information.
    """
        
    # Read the Start Time to extract the start date
    with open(hypno_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Start Time:" in line:
                start_date = line.split(':')[1].strip().split()[0]
                break

    # Skip the header (and the first artefact) and read the data
    hypnogram_df = pd.read_table(hypno_path, sep=' ', skiprows=8, names=['datetime', 'sleep stage'])
    # Reformat the time stamp structure (remove redundant decimals)
    hypnogram_df['datetime'] = hypnogram_df['datetime'].str.replace(',000;', '')

    # In some hypnograms, date is not included in the timestamp.
    # If 'time' column has only clock values without date, prepend the start_date to each timestamp.
    if " " not in hypnogram_df.iloc[0]['datetime']:
        hypnogram_df['datetime'] = start_date + ' ' + hypnogram_df['datetime']
        # Convert 'datetime' column to pandas datetime
        hypnogram_df['datetime'] = pd.to_datetime(hypnogram_df['datetime'], format='%d.%m.%Y %H:%M:%S')
        
        # Detect potential midnight rollovers by looking for large negative jumps in time differences
        rollovers = (hypnogram_df['datetime'].diff() < pd.Timedelta(0)).cumsum()
        # Change the date after midnight
        hypnogram_df['datetime'] += pd.to_timedelta(rollovers, unit='D')
        
    else:
        # If timestamps also contain date, we're all set (this is in most cases).
        hypnogram_df['datetime'] = pd.to_datetime(hypnogram_df['datetime'], format='%d.%m.%Y %H:%M:%S')
    
    hypnogram_df['sleep stage'] = hypnogram_df['sleep stage'].str.strip()  # Strip any extra spaces

    # interpolate inter-epoch artefacts (n=3 in our dataset)
    hypnogram_df['sleep stage'] = hypnogram_df['sleep stage'].replace('A', method='ffill')

    return hypnogram_df


def align_features_with_labels(sensor_df, hypnogram_df):
    """
    Aligns sensor features with hypnogram labels based on timestamps.
    
    Args:
        sensor_df (pd.DataFrame): DataFrame with sensor features.
        hypnogram_df (pd.DataFrame): DataFrame with hypnogram information.
    
    Returns:
        Tuple: Aligned features and hypnogram DataFrames.
    """
    hypno_time = hypnogram_df['datetime']

    # Find common start index
    start_idx_sensor = sensor_df.index.searchsorted(hypno_time[0])
    start_idx_hypnogram = hypno_time.searchsorted(sensor_df.index[0])

    # Set commong start index
    aligned_sensor_df = sensor_df[start_idx_sensor:]
    aligned_hypnogram_df= hypnogram_df[start_idx_hypnogram:]
    
    # Find common end index
    min_length = min(len(aligned_sensor_df), len(aligned_hypnogram_df))

    # Set common end index
    aligned_sensor_df = aligned_sensor_df[:min_length]
    aligned_hypnogram_df = aligned_hypnogram_df[:min_length]

    return aligned_sensor_df, aligned_hypnogram_df


def resample_features(acc_df, gyro_df):
    """
    Resamples sensor features to match the hypnogram's 30-second epochs.
    
    Args:
        acc_df (pd.DataFrame): DataFrame with accelerometer data.
        gyro_df (pd.DataFrame): DataFrame with gyroscope data.
    
    Returns:
        pd.DataFrame: A DataFrame with resampled features.
    """

    # Resample gyroscope features to exactly 30s intervals to match hypnogram labels
    resampled_gyro_df = gyro_df.resample('30S').mean().interpolate(method='nearest')

    resampled_acc_df = pd.DataFrame(index=resampled_gyro_df.index)
    resampled_acc_df[['feature2(m/sec)']] = np.nan

    # Downsample accelerometer features to 30s intervals to match gyroscope features
    for gyro_time in resampled_gyro_df.index:
        start_time = gyro_time 
        end_time = gyro_time + pd.Timedelta(seconds=30)
        window_data = acc_df.loc[start_time:end_time]
        if not window_data.empty:
            resampled_acc_df.loc[gyro_time] = window_data[['feature2(m/sec)']].mean()
    
    # Construct the feature DataFrame
    feature_df = pd.concat([resampled_acc_df, resampled_gyro_df], axis=1)
    return feature_df

def read_and_process_features(acc_path, gyro_path):
    """
    Reads sensor features from CSV files and processes them for classification.
    
    Args:
        acc_path (str): Path to the accelerometer CSV file.
        gyro_path (str): Path to the gyroscope CSV file.
    
    Returns:
        pd.DataFrame: A DataFrame with sensor features ready for classification.
    """
    # Read both feature files as dataframes
    accData = pd.read_csv(acc_path, skipinitialspace=True)
    gyroData = pd.read_csv(gyro_path, skipinitialspace=True)

    # Sensor clock is in UTC+0. Convert timestamps to Helsinki time.
    accTime = pd.to_datetime(accData['UTCTimestamp(ms)'], unit='ms', utc=True).dt.tz_convert('Europe/Helsinki').dt.tz_localize(None)
    gyroTime = pd.to_datetime(gyroData['UTCTimestamp(ms)'], unit='ms', utc=True).dt.tz_convert('Europe/Helsinki').dt.tz_localize(None)

    # Set time based indexing and select features for classifier:
    # feature2(m/sec): Activity, feature3_Y(unitless): respiration max ACF, feature4_Y(Hz): respiration rate
    # feature5_Y(grad/sec): respiration peaks median, feature6_Y(grad/sec): respiration peaks standard deviation
    accData = accData.set_index(accTime)[['feature2(m/sec)']]
    gyroData = gyroData.set_index(gyroTime)[['feature3_Y(unitless)', 'feature4_Y(Hz)', 'feature5_Y(grad/sec)', 'feature6_Y(grad/sec)']]

    # Resample features to constant 30s to match hypnogram
    feature_df = resample_features(accData, gyroData)

    return feature_df

def compile_recording(acc_path, gyro_path, hypno_path=None):

    """
    Compiles a single sleep recording from sensor and hypnogram data.

    Args:
        acc_path (str): Path to the accelerometer data file.
        gyro_path (str): Path to the gyroscope data file.
        hypno_path (str): Path to the hypnogram data file.

    Returns:
        SleepRecording: An instance of SleepRecording with features and labels.
    """
    # Load selected five sensor features (activity, autocorrelation, respiration rate, peak median, peak std)
    sensor_features_df = read_and_process_features(acc_path, gyro_path)

    # Extract the subject id from hypnogram filename
    id = int(hypno_path.split('.txt')[0].split('\\')[-1])

    hypnogram_df = read_and_process_hypnogram(hypno_path)

    aligned_sensor_features_df, aligned_hypnogram_df = align_features_with_labels(sensor_features_df, hypnogram_df)
    timestamps = aligned_sensor_features_df.index

    features = aligned_sensor_features_df.to_numpy()
    labels = aligned_hypnogram_df['sleep stage'].to_numpy()

    return SleepRecording(features=features, labels=labels, id=id, timestamps=timestamps)

def scan_directory(path):
    """
    Scans a directory for accelerometer, gyroscope, and hypnogram files.
    
    Args:
        path (str): Directory path to scan.
    
    Returns:
        Tuple: Paths to the accelerometer, gyroscope, and hypnogram files.
    """
    acc_file, gyro_file, hypno_file = None, None, None

    for file in os.listdir(path):
        if file.endswith('.csv') and 'AccFeatures' in file:
            acc_file = file
        elif file.endswith('.csv') and 'GyroFeatures' in file:
            gyro_file = file
        elif file.endswith('.txt'):
            hypno_file = file
    
    return (acc_file, gyro_file, hypno_file)

def create_dataset(root):
    """
    Creates a dataset by scanning a parent directory for subdirectories containing all relevant sleep recording files.
    
    Args:
        root (str): The root directory containing subdirectories of sleep recordings.
    
    Returns:
        NappaDataset: A dataset containing multiple SleepRecording instances.
    """ 
    directories = [entry.name for entry in os.scandir(root) if entry.is_dir()]
    recordings = []

    for directory in directories:

        path = os.path.join(root, directory) + '\\'
        
        acc_file, gyro_file, hypno_file = scan_directory(path)

        recording = compile_recording(acc_path=os.path.join(path, acc_file),
                                        gyro_path=os.path.join(path, gyro_file),
                                        hypno_path=os.path.join(path, hypno_file))
        recordings.append(recording)


    return NappaDataset(data=recordings)