import numpy as np

from .datacloud import DataCloud
from .sample import Sample


class AutoCloud:

    def __init__(self, chebyshev_parameter: float):
        self.chebyshev_parameter = chebyshev_parameter
        self.data_clouds: list[DataCloud] = []

    def verify_merge(self, set_cloud_i: set[Sample], set_cloud_j: set[Sample]) -> bool:
        """
        Verifies if two DataClouds can be merged, given their sets of samples.

        Parameters
        ----------
        set_cloud_i : set[Sample]
            The first DataCloud to check.
        set_cloud_j : set[Sample]
            The second DataCloud to check.

        Returns
        -------
        bool
            True if the clouds can be merged, False otherwise.
        """
        intersection_i_j = set_cloud_i & set_cloud_j
        s_i_minus_j = set_cloud_i - intersection_i_j
        s_j_minus_i = set_cloud_j - intersection_i_j
        if (len(intersection_i_j) > len(s_i_minus_j)) or (
            len(intersection_i_j) > len(s_j_minus_i)
        ):
            return True
        return False

    def run_single_sample(self, sample: Sample) -> np.ndarray:
        if len(self.data_clouds) == 1:
            self.data_clouds[0] += sample
        else:
            has_joined = False
            for cloud_index, cloud in enumerate(self.data_clouds):
                points = cloud.points
                new_points = np.vstack((points, sample))
                new_mean = cloud.calculate_new_mean(x=sample, old_mean=cloud.mean)
                new_variance = cloud.calculate_new_variance(
                    x=sample, new_mean=new_mean, old_variance=cloud.variance
                )
                eccentricity = cloud.calculate_eccentricity(
                    points=new_points,
                    new_mean=new_mean,
                    new_variance=new_variance,
                )
                if eccentricity <= (
                    (self.chebyshev_parameter**2 + 1) / len(new_points)
                ):
                    cloud = cloud + sample
                    self.data_clouds[cloud_index] = cloud
                    has_joined = True
            if not has_joined:
                new_cloud = DataCloud(
                    x=sample,
                )
                self.data_clouds.append(new_cloud)
            new_clouds: list[DataCloud] = []
            for i in range(0, len(self.data_clouds) - 1):
                for j in range(i + 1, len(self.data_clouds)):
                    if self.verify_merge(
                        set(self.data_clouds[i].points),
                        set(self.data_clouds[j].points),
                    ):
                        new_cloud = self.data_clouds[i] + self.data_clouds[j]
                        new_clouds.append(new_cloud)
                    else:
                        new_clouds.extend([self.data_clouds[i], self.data_clouds[j]])
            self.data_clouds = new_clouds

    def run(self, samples: list[Sample]) -> None:
        """
        Runs the AutoCloud algorithm on a list of samples.

        Parameters
        ----------
        samples : list[Sample]
            The list of samples to process.
        """
        for sample in samples:
            self.run_single_sample(sample)
