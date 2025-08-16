import numpy as np

from datacloud import DataCloud
from.sample import Sample


class AutoCloud:

    def __init__(self, chebyshev_parameter: float):
        self.chebyshev_parameter = chebyshev_parameter
        self.data_clouds: list[DataCloud] = []

    def verify_merge(self, cloud_i: DataCloud, cloud_j: DataCloud) -> bool:

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
