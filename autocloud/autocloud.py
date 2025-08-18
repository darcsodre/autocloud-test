import logging
import numpy as np

from .datacloud import DataCloud
from .sample import Sample


LOGGER = logging.getLogger(__name__)


class AutoCloud:

    def __init__(self, chebyshev_parameter: float):
        self.chebyshev_factor = chebyshev_parameter**2 + 1
        self.data_clouds: list[DataCloud] = []
        self.internal_counter = 0

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
        if self.internal_counter == 1:
            self.data_clouds[0] += sample
        else:
            has_joined = False
            for cloud_index, cloud in enumerate(self.data_clouds):
                num_points = len(cloud) + 1
                new_mean = cloud.calculate_new_mean(x=sample, old_mean=cloud.mean)
                new_variance = cloud.calculate_new_variance(
                    x=sample,
                    new_mean=new_mean,
                    old_mean=cloud.mean,
                    old_variance=cloud.variance,
                )
                eccentricity = cloud.calculate_eccentricity(
                    num_points=num_points,
                    mean=new_mean,
                    variance=new_variance,
                    point=sample.data,
                )
                # LOGGER.debug(
                #     f"Sample {sample.sample_id} - Cloud {cloud_index}: "
                #     f"Eccentricity: {eccentricity}, Chebyshev Factor: {self.chebyshev_factor / num_points}"
                # )
                # Calculo do threshold
                if eccentricity <= (self.chebyshev_factor / num_points):
                    cloud = cloud + sample
                    self.data_clouds[cloud_index] = cloud
                    has_joined = True
            new_cloud: DataCloud | None = None
            if not has_joined:
                new_cloud = DataCloud(
                    x=sample,
                )
            new_clouds: list[DataCloud] = []
            merged_clouds: set[int] = set()
            for i in range(0, len(self.data_clouds) - 1):
                for j in range(i + 1, len(self.data_clouds)):
                    if i in merged_clouds and j in merged_clouds:
                        continue
                    if self.verify_merge(
                        self.data_clouds[i].set_data_points,
                        self.data_clouds[j].set_data_points,
                    ):
                        new_cloud = self.data_clouds[i] + self.data_clouds[j]
                        new_clouds.append(new_cloud)
                        merged_clouds.add(i)
                        merged_clouds.add(j)
                    else:
                        if i not in merged_clouds:
                            new_clouds.append(self.data_clouds[i])
                            merged_clouds.add(i)
                        if j not in merged_clouds:
                            new_clouds.append(self.data_clouds[j])
                            merged_clouds.add(j)
            if len(new_clouds) == 0:
                new_clouds = self.data_clouds
            self.data_clouds = new_clouds
            self.data_clouds.append(new_cloud) if new_cloud else None
        self.internal_counter += 1

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
