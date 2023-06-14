import matplotlib.pyplot as plt
import numpy as np


class Trajectory():

    def __init__(self, file_path, name):
        
        # read txt file
        self.is_first = 1
        f = open(file_path, mode='r')
        for line in f:
            data = line.split(' ')
            data[-1]=data[-1].replace("\n","")

            for i in range(len(data)):
                data[i]=float(data[i])
            
            if self.is_first:
                self.trajectory = np.array(data[1:4]).reshape(1,3)
                self.orientation = np.array(data[4:]).reshape(1,4)
                self.time = np.array([0]).reshape(1,1)
                self.is_first=False
                self.start_time = data[0]
                self.init_position = np.array(data[1:4]).reshape(1,3)
                self.init_orientation = np.array(data[4:]).reshape(1,4)
            else:
                self.trajectory = np.append(self.trajectory, np.array(data[1:4]-self.init_position).reshape(1,3), axis=0)
                self.orientation = np.append(self.orientation, np.array(data[4:]-self.init_orientation).reshape(1,4), axis=0)
                # self.time = np.append(self.time, np.array((data[0]-self.start_time).to_nsec()).reshape(1,1), axis=0)
                self.time = np.append(self.time, np.array(data[0]-self.start_time).reshape(1,1), axis=0)

                
        f.close()

        self.name = name
        self.length = int(self.trajectory.shape[0])

    def pose_matrix(self, index):
        return np.vstack([np.hstack([self.orientation[index].rotation(), self.trajectory[index].reshape(3, 1)]),
                          np.array([0, 0, 0, 1])])


#Visualization
#data structure : [timestamp position(x y z) quaternion(x y z w)]

def plot2D_xy(gt, data):
    plt.figure(figsize=(6, 5))
    plt.title('Top-View(XY)')
    plt.plot(data.trajectory[:,0], data.trajectory[:,1], label='GroundTruth')
    if gt: plt.plot(gt.trajectory[:, 0], gt.trajectory[:, 1], label=gt.name, ls='--')
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()
    # plt.show()

def plot2D_xz(gt, data):
    plt.figure(figsize=(6, 5))
    plt.title('Top-View(XZ)')
    plt.plot(data.trajectory[:,0], data.trajectory[:,2], label='GroundTruth')
    if gt: plt.plot(gt.trajectory[:, 0], gt.trajectory[:, 2], label=gt.name, ls='--')
    plt.xlabel("x[m]")
    plt.ylabel("z[m]")
    plt.legend()
    # plt.show()

def plotXYZ(gt, data):
    plt.figure(figsize=(6, 10))
    plt.subplot(3, 1, 1)
    plt.plot(data.time, data.trajectory[:, 0], label=data.name)
    if gt: plt.plot(gt.time, gt.trajectory[:, 0], label=gt.name, ls='--')
    plt.ylabel('x[m]')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(data.time, data.trajectory[:, 1], label=data.name)
    if gt: plt.plot(gt.time, gt.trajectory[:, 0], label=gt.name, ls='--')
    plt.ylabel('y[m]')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(data.time, data.trajectory[:, 2], label=data.name)
    if gt: plt.plot(gt.time, gt.trajectory[:, 0], label=gt.name, ls='--')
    plt.ylabel('z[m]')
    plt.legend()

def plot3D(gt, data):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data.trajectory[:, 0], data.trajectory[:, 1], data.trajectory[:, 2], label=data.name)
    if gt: ax.scatter(gt.trajectory[:, 0], gt.trajectory[:, 1], gt.trajectory[:, 2], label=gt.name)
    ax.legend()
    ax.set_zlim3d(-40, 40)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


if __name__=="__main__":

    Hilti_gt_path = "/home/doyu/dataset/Hilti/2022/"
    gt_path="exp04_construction_upper_level.txt"
    test_path="exp05_construction_upper_level_2.txt"

    gt_traj=Trajectory(Hilti_gt_path+gt_path, "reference")
    test_traj=Trajectory(Hilti_gt_path+test_path, "test")

    print("trajectory data size :", gt_traj.trajectory.shape)
    print("orientation data size :", gt_traj.orientation.shape)
    print("time data size :", gt_traj.time.shape)
    print("name :", gt_traj.name)
    print("length :", gt_traj.length)
    print(gt_traj.trajectory[:3, :])
    print(gt_traj.orientation[:3, :])

    #plot
    plot2D_xy(gt_traj, test_traj)
    plot2D_xz(gt_traj, test_traj)
    plotXYZ(gt_traj, test_traj)
    plot3D(gt_traj, test_traj)
    plt.show()

