from pathlib import Path
import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

np.set_printoptions(suppress=True)

project_root = Path(__file__).parent
robot_library = project_root / "robots/library"

robot_usd_path = robot_library / "ABB/CRB15000_10kg_152_v1/CRB15000_10kg_152/CRB15000_10kg_152.usd"
assert robot_usd_path.exists()
robot_mount_prim_path = f"/World/robot_mount"
robot_prim_path = f"{robot_mount_prim_path}/robot"
add_reference_to_stage(str(robot_usd_path), robot_prim_path)

flange_prim_path = f"{robot_prim_path}/link_6/flange"

RESOLUTION = (1280, 720)

def create_camera(prim_path: str) -> Camera:
    camera = Camera(
        prim_path=prim_path,
        name="eoat_camera",
        resolution=RESOLUTION,
        translation=(0, 0, 0),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, -90, 0]), degrees=True),
    )

    return camera

def set_camera_intrinsics_from_k_matrix(camera: Camera, k_matrix: np.ndarray):
    fx = k_matrix[0, 0]
    fy = k_matrix[1, 1]
    cx = k_matrix[0, 2]
    cy = k_matrix[1, 2]
    
    width, height = camera.get_resolution()
    
    # Set focal length (using fx as reference)
    camera.set_focal_length(fx)
    
    # Calculate apertures based on focal lengths and resolution
    # Assuming unit pixel size, aperture = resolution * pixel_size
    # Since focal_length = aperture / pixel_size, we get pixel_size = aperture / focal_length
    # Default aperture values can be calculated from the current focal length
    current_focal = camera.get_focal_length()
    horizontal_aperture = width * current_focal / fx
    vertical_aperture = height * current_focal / fy
    
    camera.set_horizontal_aperture(horizontal_aperture)
    camera.set_vertical_aperture(vertical_aperture)

world = World()

world.scene.add_default_ground_plane() # type: ignore

robot = SingleArticulation(robot_prim_path)

camera = create_camera(f"{flange_prim_path}/camera")

world.reset()

robot.initialize()
camera.initialize()
camera.add_distance_to_camera_to_frame({})
camera.add_distance_to_image_plane_to_frame({})
camera.set_clipping_range(0.01, 1000)

k = camera.get_intrinsics_matrix()
print("Original K matrix:")
print(k)

# Example: Set custom K matrix
custom_k = np.array([[800, 0, 640],
                     [0, 800, 360],
                     [0, 0, 1]], dtype=np.float32)

print("\nSetting custom K matrix:")
print(custom_k)

set_camera_intrinsics_from_k_matrix(camera, custom_k)

# Verify the change
new_k = camera.get_intrinsics_matrix()
print("\nNew K matrix after setting:")
print(new_k)

i = 0
while True:
    robot.set_joint_positions(np.deg2rad(np.array([np.sin(i/100)*90, 10, 10, 10, 10, np.sin(i/100)*45])))
    world.step(render=True)
    i += 1