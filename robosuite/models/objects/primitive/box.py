import numpy as np

from robosuite.models.objects import PrimitiveObject
from robosuite.utils.mjcf_utils import get_size


class BoxObject(PrimitiveObject):
    """
    A box object.

    Args:
        size (3-tuple of float): (half-x, half-y, half-z) size parameters for this box object
    """

    def __init__(
        self,
        name,
        size=None,
        size_max=None,
        size_min=None,
        density=1000,
        friction=None,
        rgba=None,
        solref=[0.001, 1],   
        solimp=[0.9, 0.95, 0.001, 0.5, 2],
        material=None,
        joints="default",
        obj_type="all",
        duplicate_collision_geoms=True,
        priority=0, ###############
    ):
        size = get_size(size, size_max, size_min, [0.07, 0.07, 0.09], [0.03, 0.03, 0.09])
        super().__init__(
            name=name,
            size=size,
            rgba=rgba,
            density=density,
            friction=friction,
            solref=solref,
            solimp=solimp,
            material=material,
            joints=joints,
            obj_type=obj_type,
            duplicate_collision_geoms=duplicate_collision_geoms,
            priority=priority, ###############
        )

    def sanity_check(self):
        """
        Checks to make sure inputted size is of correct length

        Raises:
            AssertionError: [Invalid size length]
        """
        assert len(self.size) == 3, "box size should have length 3"

    def _get_object_subtree(self):
        return self._get_object_subtree_(ob_type="box")

    @property
    def bottom_offset(self):
        return np.array([0, 0, -1 * self.size[2]])

    @property
    def top_offset(self):
        return np.array([0, 0, self.size[2]])

    @property
    def horizontal_radius(self):
        return np.linalg.norm(self.size[0:2], 2)
