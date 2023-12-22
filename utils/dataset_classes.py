import numpy as np
import pickle
import torch

class SleepRecording:
    """
    A class to represent a single sleep recording.
    """
    def __init__(self, features, labels, id=None, age=None, timestamps=None, sampling_interval=None):
        """
        Initialize a SleepRecording object.

        Args:
          features (np.ndarray): Sensor data representing the features extracted from the sleep recording.
          labels (np.ndarray): The labels corresponding to each epoch in the sleep recording.
          id (int, optional): A unique identifier for the subject of the sleep recording.
          age (int, optional): The age of the subject in months.
          timestamps (np.ndarray, optional): Timestamps corresponding to each epoch in the sleep recording.
          sampling_interval (float, optional): The time interval between data points in the recording.
        """
        self.id = id
        self.age = age
        self.features = features
        self.labels = labels
        self.timestamps = timestamps
        self.start = timestamps[0]
        self.end = timestamps[-1]
        self.duration = self.end - self.start
        self.sampling_interval = sampling_interval

    def __len__(self):
        return self.features.shape[0]

    
class NappaDataset(torch.utils.data.Dataset):
    """
    A dataset class to hold multiple SleepRecording objects for batched processing.
    """
    def __init__(self, data):
        """
        Initialize a NappaDataset object.

        Args:
          data (list or str): If list, should contain SleepRecording objects. If str, should be the path to a pickled file of the dataset.
        """
        self.normalization = None
        self.recordings = []

        if all(isinstance(item, SleepRecording) for item in data):
            self.recordings = data
        elif isinstance(data, str) and data.endswith('.pkl'):
            with open(data, 'rb') as f:
                obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)
        else:
            raise ValueError("Data must be a list of SleepRecording instances or a path to a pickled NappaDataset object.")

    @property
    def features(self) -> np.ndarray:
        """
        Returns a concatenated numpy array of features from all SleepRecording objects in the dataset.
        """
        return np.concatenate([rec.features for rec in self.recordings], axis=0)

    @property
    def labels(self) -> np.ndarray:
        """
        Returns a concatenated numpy array of labels from all SleepRecording objects in the dataset.
        """
        return np.concatenate([rec.labels for rec in self.recordings], axis=0)

    @property
    def ids(self) -> np.ndarray:
        """
        Returns a numpy array of subject IDs from all SleepRecording objects in the dataset.
        """
        return np.array([rec.id for rec in self.recordings])

    @property
    def ages(self) -> np.ndarray:
        """
        Returns a numpy array of subject ages from all SleepRecording objects in the dataset.
        """
        return np.array([rec.age for rec in self.recordings])

    def save(self, filename: str):
        """
        Save the dataset to a pickle file.

        Args:
          filename (str): The path where the dataset should be saved.
        """
        if filename.endswith('.pkl'):
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError("Unsupported file format. Use .pkl for saving.")

    def setSubjectAges(self, ages: dict) -> 'NappaDataset':
        """
        Update the ages for subjects based on a provided mapping from ID to age.

        Args:
          ages (dict): A dictionary mapping subject ID to age in months.

        Returns:
          NappaDataset: The dataset with updated ages.
        """
        for subject in self.recordings:
            subject.age = ages.get(subject.id, subject.age)
        return self

    def labelsToNumeric(self, mapping: dict) -> 'NappaDataset':
        """
        Convert string labels (Sleep stages 'N3', 'N2', etc.) to numeric labels based on a provided mapping.

        Args:
          mapping (dict): A dictionary mapping string labels to numeric labels.

        Returns:
          NappaDataset: The dataset with numeric labels.
        """
        if self.labels.dtype != np.dtype('long'):
            for rec in self.recordings:
                rec.labels = np.array([mapping[label] for label in rec.labels], dtype='long')
        return self

    def getById(self, id: int) -> SleepRecording:
            """
            Retrieve a SleepRecording by subject ID.

            Args:
            id (int): The subject ID.

            Returns:
            SleepRecording: The recording corresponding to the given ID, or None if not found.
            """
            return next((rec for rec in self.recordings if rec.id == id), None)

    def dropById(self, id: int) -> 'NappaDataset':
        """
        Remove a SleepRecording from the dataset by ID.

        Args:
          id (int): The subject ID.

        Returns:
          NappaDataset: The dataset with the specified recording removed.
        """
        self.recordings = [rec for rec in self.recordings if rec.id != id]
        return self

    def sortById(self) -> 'NappaDataset':
        """
        Sort the dataset by subject ID.

        Returns:
          NappaDataset: The sorted dataset.
        """
        self.recordings.sort(key=lambda recording: recording.id)
        return self

    def sortByLength(self) -> 'NappaDataset':
        """
        Sort the dataset by the length of the recordings.

        Returns:
          NappaDataset: The sorted dataset.
        """
        self.recordings.sort(key=lambda recording: len(recording))
        return self

    def __len__(self):
        return len(self.recordings)
    
    def __getitem__(self, index):
        return self.recordings[index]
    
    def __str__(self):
        ret = [
            f'Number of sleep recordings: {len(self)}\n',
            f'Subject age range: {np.min(self.ages)} - {np.max(self.ages)} (months) \n' if None not in self.ages  else '',
            f'Number of data points: {self.features.shape[0]}\n',
            f'Normalization: {self.normalization}\n']
        return ''.join(ret)