import matplotlib.pyplot as plt

import trajectory as Trajectory
import error as Error

################################# argument #####################################
scenario = "09"
gt_path="trajectories/Reference/reference_" + scenario + ".txt" #Hilti
gt_KITTI_path="trajectories/KITTI_gt/" + scenario + ".txt"      #KITTI
LOAM_path="trajectories/A-LOAM/A-LOAM_kitti_" + scenario + ".txt"
Lego_LOAM_path="trajectories/LeGO-LOAM_LiDAR_only/LeGO-LOAM_LiDAR_only_" + scenario + ".txt"
Fast_LIO_path = "trajectories/Fast_LIO/fast_lio_" + scenario + ".txt"
Faster_LIO_path = "trajectories/Faster_LIO/faster_lio_" + scenario + ".txt"
LIO_SAM_path = "trajectories/LIO-SAM/lio_sam_" + scenario + ".txt"
BALM_path = "trajectories/BALM/BALM_kitti_"+ scenario + ".txt"
BALM_backend_path = "trajectories/BALM-backend/BAML-backend_kitti_"+ scenario + ".txt"


gt_traj=Trajectory.Trajectory(gt_KITTI_path, "Reference", opt=5)
test_traj=Trajectory.Trajectory(LOAM_path, "LOAM", opt=1)
# test_traj=Trajectory.Trajectory(Lego_LOAM_path, "Lego_LOAM", opt=2)
# test_traj=Trajectory.Trajectory(LIO_SAM_path, "LIO_SAM", opt=3)
# test_traj=Trajectory.Trajectory(Fast_LIO_path, "Fast_LIO", opt=4)
# test_traj=Trajectory.Trajectory(Faster_LIO_path, "Faster_LIO", opt=4)
#test_traj=Trajectory.Trajectory(BALM_path, "BALM", opt=1)
#test_traj=Trajectory.Trajectory(BALM_backend_path, "BALM-backend", opt=1)

error = Error.Error(reference=gt_traj, estimate=test_traj)
################################################################################

#check data
# print("trajectory data size :", gt_traj.trajectory.shape)
# print("orientation data size :", gt_traj.orientation.shape)
# print("time data size :", gt_traj.time.shape)
# print("name :", gt_traj.name)
# print("length :", gt_traj.length)
# print(gt_traj.trajectory[:3, :])
# print(gt_traj.orientation[:3])

#plot Trajectory
Trajectory.plot2D('xy', gt_traj, test_traj)
Trajectory.plot2D('xz', gt_traj, test_traj)
Trajectory.plotXYZ(gt_traj, test_traj)
Trajectory.plot3D(gt_traj, test_traj)
# Trajectory.plotQuat(gt_traj, test_traj)

#plot Error
Error.plotAPE(error)
Error.plotRPE(error)
Error.plotAPEStats(error)
Error.plotRPEStats(error)
plt.show()