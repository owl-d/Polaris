import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import rosbag

import quaternion as quat
import math


class Trajectory:
    def __init__(self, file_name, name, opt=0):
        """Get trajectory and pose data from a file

        Args:
            file_name (string): data's file name 
        """
        print("Reading {}".format(file_name))
        self.is_None = False
        if file_name.endswith('.txt') or file_name.endswith('.csv'):
            #name = file_name.split('/')[2].split('.')[0].split('_')[2]

            trajectory, orientation, time_dur = [], [], []
            length = 0
            self.is_first = 1
            f = open(file_name, mode='r')
            for line in f:
                data = line.split(' ')
                data[-1]=data[-1].replace("\n","")
                length += 1

                if len(data) > 8: data=data[:-1]
                for i in range(len(data)):
                    data[i]=float(data[i])
                
                if self.is_first:
                    self.is_first=False
                    init_data = data
                    init_rot_mtx = (quat.Quaternion([init_data[4], init_data[5], init_data[6], init_data[7]])**-1).rotation()
                    init_quat_inv = quat.Quaternion(init_data[4:])**-1
                
                if opt == 0 : # Hilti Referece
                    cor_trajectory = np.dot(init_rot_mtx, [[data[1]-init_data[1]], [data[2]-init_data[2]], [data[3]-init_data[3]]])
                    trajectory.append(cor_trajectory.reshape(3))
                    cor_orientation = quat.Quaternion([data[4], data[5], data[6], data[7]]) * init_quat_inv
                    cor_orientation = quat.Quaternion(cor_orientation.normalize())
                    orientation.append(cor_orientation)
                    time_dur.append((data[0] - init_data[0])*1e+9) #nano_sec
                elif opt == 1 : # A-LOAM
                    trajectory.append([-data[2], -data[1], -data[3]])
                    orientation.append(quat.Quaternion([data[4], data[5], data[6], data[7]]))
                    time_dur.append(data[0] - init_data[0])
                elif opt == 2 : # LeGO-LOAM
                    trajectory.append([-data[1], -data[3], -data[2]])
                    orientation.append(quat.Quaternion([data[6], data[4], data[5], data[7]]))
                    time_dur.append(data[0] - init_data[0])
                elif opt == 3 : # LIO-SAM
                    trajectory.append([-data[2], -data[1], -data[3]])
                    orientation.append(quat.Quaternion([data[4], data[5], data[6], data[7]]))
                    time_dur.append((data[0] - init_data[0])*1e+9)
                elif opt == 4 : # Faster-LIO, Fast-LIO
                    cor_trajectory = np.dot(init_rot_mtx, [[data[1]-init_data[1]], [data[2]-init_data[2]], [data[3]-init_data[3]]])
                    trajectory.append(cor_trajectory.reshape(3))
                    cor_orientation = quat.Quaternion([data[5], data[4], data[6], data[7]]) * init_quat_inv
                    cor_orientation = quat.Quaternion(cor_orientation.normalize())
                    orientation.append(cor_orientation)
                    time_dur.append((data[0] - init_data[0])*1e+9)
                elif opt == 5 : #KIIT ref
                    cor_trajectory = np.dot(init_rot_mtx, [[data[1]-init_data[1]], [-data[3]+init_data[3]], [data[2]-init_data[2]]])
                    trajectory.append(cor_trajectory.reshape(3))
                    cor_orientation = quat.Quaternion([data[4], data[5], data[6], data[7]]) * init_quat_inv
                    cor_orientation = quat.Quaternion(cor_orientation.normalize())
                    orientation.append(cor_orientation)
                    time_dur.append((data[0] - init_data[0])*1e+8) #nano_sec

            f.close()


        # elif file_name.endswith('.bag'):
        #     with rosbag.Bag(file_name) as bag:
        #         trajectory, orientation, name, time_dur, length = self._read_bag(bag)

        else:
            print("unsupported type of data file")
            self.is_None = True

        self.trajectory = np.array(trajectory)
        self.orientation = np.array(orientation)
        # self.raw_orientation = np.array(raw_orientation)

        self.time = np.array(time_dur)
        self.name = name
        self.length = length

        self.is_gt = False
        if self.name == 'gt' or self.name == 'ground_truth' or self.name == '/ground_truth' or self.name == '/Reference' : self.is_gt = True
        print("{} with length {}".format(self.name, self.length))

    def _read_bag(self, bag):
        traj, rot, time = [], [], []
        for topic, msg, _ in bag.read_messages():
            poses = msg.poses
        for msg in poses:
            traj.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            rot.append(quat.Quaternion([msg.pose.orientation.x, msg.pose.orientation.y,
                                        msg.pose.orientation.z, msg.pose.orientation.w]))
            time.append((msg.header.stamp - poses[0].header.stamp).to_nsec())
        return traj, rot, topic, time, len(poses)

    def pose_matrix(self, index):
        return np.vstack([np.hstack([self.orientation[index].rotation(), self.trajectory[index].reshape(3, 1)]),
                          np.array([0, 0, 0, 1])])

def plotXYZ(gt, trajs):
    plt.figure(figsize=(6, 10))
    plt.subplot(3, 1, 1)
    for traj in trajs:
        plt.plot(traj.time, traj.trajectory[:, 0], label=traj.name)
    if gt: plt.plot(gt.time, gt.trajectory[:, 0], label=gt.name, ls='--')
    plt.ylabel('x[m]')
    plt.legend()

    plt.subplot(3, 1, 2)
    for traj in trajs:
        plt.plot(traj.time, traj.trajectory[:, 1], label=traj.name)
    if gt: plt.plot(gt.time, gt.trajectory[:, 1], label=gt.name, ls='--')
    plt.ylabel('y[m]')
    plt.legend()

    plt.subplot(3, 1, 3)
    for traj in trajs:
        plt.plot(traj.time, traj.trajectory[:, 2], label=traj.name)
    if gt: plt.plot(gt.time, gt.trajectory[:, 2], label=gt.name, ls='--')
    plt.ylabel('z[m]')
    plt.xlabel('time[nano_sec]')
    plt.legend()

def plot2D(option, gt, trajs):
    plt.figure(figsize=(6, 5))
    plt.title('Top-View')
    if option == 'xy':
        for traj in trajs:
            plt.plot(traj.trajectory[:, 0], traj.trajectory[:, 1], label=traj.name)
        if gt: plt.plot(gt.trajectory[:, 0], gt.trajectory[:, 1], label=gt.name, ls='--')
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()
    if option == 'xz':
        for traj in trajs:
            plt.plot(traj.trajectory[:, 0], traj.trajectory[:, 2], label=traj.name)
        if gt: plt.plot(gt.trajectory[:, 0], gt.trajectory[:, 2], label=gt.name, ls='--')
        plt.xlabel("x[m]")
        plt.ylabel("z[m]")
        plt.legend()

def plot3D(gt, trajs):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    for traj in trajs:
        ax.scatter(traj.trajectory[:, 0], traj.trajectory[:, 1], traj.trajectory[:, 2], label=traj.name)
    if gt: ax.scatter(gt.trajectory[:, 0], gt.trajectory[:, 1], gt.trajectory[:, 2], label=gt.name)
    ax.legend()
    ax.set_zlim3d(-40, 40)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def plotQuat(gt, trajs):
    traj_list, gt_list = [], []
    plt.figure(figsize=(6, 10))
    plt.subplot(4, 1, 1)
    for i in range(trajs.length):
        traj_list.append(abs(trajs.orientation[i].x))
    if gt:
        for i in range(gt.length): gt_list.append(abs(gt.orientation[i].x))
    plt.plot(gt.time, gt_list, label=gt.name, ls='--')    
    plt.plot(trajs.time, traj_list, label=trajs.name)
    plt.ylabel('x')
    plt.xlabel('time[nano_sec]')
    plt.legend()

    traj_list.clear()
    gt_list.clear()
    plt.subplot(4, 1, 2)
    for i in range(trajs.length):
        traj_list.append(abs(trajs.orientation[i].y))
    if gt:
        for i in range(gt.length): gt_list.append(abs(gt.orientation[i].y))
    plt.plot(gt.time, gt_list, label=gt.name, ls='--')    
    plt.plot(trajs.time, traj_list, label=trajs.name)
    plt.ylabel('y')
    plt.xlabel('time[nano_sec]')
    plt.legend()

    traj_list.clear()
    gt_list.clear()
    plt.subplot(4, 1, 3)
    for i in range(trajs.length):
        traj_list.append(abs(trajs.orientation[i].z))
    if gt:
        for i in range(gt.length): gt_list.append(abs(gt.orientation[i].z))
    plt.plot(gt.time, gt_list, label=gt.name, ls='--')    
    plt.plot(trajs.time, traj_list, label=trajs.name)
    plt.ylabel('z')
    plt.xlabel('time[nano_sec]')
    plt.legend()

    traj_list.clear()
    gt_list.clear()
    plt.subplot(4, 1, 4)
    for i in range(trajs.length):
        traj_list.append(abs(trajs.orientation[i].w))
    if gt:
        for i in range(gt.length): gt_list.append(abs(gt.orientation[i].w))
    plt.plot(gt.time, gt_list, label=gt.name, ls='--')    
    plt.plot(trajs.time, traj_list, label=trajs.name)
    plt.ylabel('w')
    plt.xlabel('time[nano_sec]')
    plt.legend()



if __name__=="__main__":

    ################################# argument #####################################
    scenario = "exp14"
    gt_path="trajectories/Reference/reference_" + scenario + ".txt"
    LOAM_path="trajectories/A-LOAM/A-LOAM_" + scenario + ".txt"
    Lego_LOAM_path="trajectories/LeGO-LOAM_LiDAR_only/LeGO-LOAM_LiDAR_only_" + scenario + ".txt"
    Fast_LIO_path = "trajectories/Fast_LIO/fast_lio_" + scenario + ".txt"
    Faster_LIO_path = "trajectories/Faster_LIO/faster_lio_" + scenario + ".txt"
    LIO_SAM_path = "trajectories/LIO-SAM/lio_sam_" + scenario + ".txt"

    gt_traj=Trajectory(gt_path, "reference")
    # test_traj=Trajectory(LOAM_path, "LOAM", opt=1)
    test_traj=Trajectory(Lego_LOAM_path, "Lego_LOAM", opt=2)
    # test_traj=Trajectory(LIO_SAM_path, "LIO_SAM", opt=3)
    # test_traj=Trajectory(Fast_LIO_path, "Fast_LIO", opt=4)
    # test_traj=Trajectory(Faster_LIO_path, "Faster_LIO", opt=4)
    ################################################################################

    # print("gt trajectory data size :", gt_traj.trajectory.shape)
    # print("gt orientation data size :", gt_traj.orientation.shape)
    # print("gt time data size :", gt_traj.time.shape)
    # print("gt name :", gt_traj.name)
    # print("gt length :", gt_traj.length)
    # print("gt trajectory 3 row :\n", gt_traj.trajectory[:3, :])
    # print("gt orientation 3 row :\n", gt_traj.orientation[0:3])
    # print("gt duration 3 row :", gt_traj.time[0:3])
    # print("\ntest trajectory data size :", test_traj.trajectory.shape)
    # print("test orientation data size :", test_traj.orientation.shape)
    # print("test time data size :", test_traj.time.shape)
    # print("test name :", test_traj.name)
    # print("test length :", test_traj.length)
    # print("test trajectory 3 row :\n", test_traj.trajectory[:3, :])
    # print("test orientation 3 row :\n", test_traj.orientation[0:3])
    # print("test duration 3 row :", test_traj.time[0:3])

    #plot
    plotXYZ(gt_traj, test_traj)
    plot2D('xy', gt_traj, test_traj)
    plot2D('xz', gt_traj, test_traj)
    plot3D(gt_traj, test_traj)
    plotQuat(gt_traj, test_traj)
    plt.show()