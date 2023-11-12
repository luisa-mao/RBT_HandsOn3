import math
import numpy as np
import copy

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2

def rotation_matrix(axis, angle):
  """
  Create a 3x3 rotation matrix which rotates about a specific axis

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians

  Returns:
    3x3 rotation matrix as a numpy array
  """
  rot_mat = np.eye(3)
  cos_theta = np.cos(angle)
  sin_theta = np.sin(angle)
  ux, uy, uz = axis

  rot_mat = np.array([[cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
                        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
                        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]])
    
  return rot_mat

def homogenous_transformation_matrix(axis, angle, v_A):
  """
  Create a 4x4 transformation matrix which transforms from frame A to frame B

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians
    v_A:   Vector. The vector translation from A to B defined in frame A

  Returns:
    4x4 transformation matrix as a numpy array
  """
  cos_theta = np.cos(angle)
  sin_theta = np.sin(angle)
  ux, uy, uz = axis

  rotation_matrix = np.array([[cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
                              [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
                              [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]])

  T = np.eye(4)
  T[:3, :3] = rotation_matrix
  T[:3, 3] = v_A

  return T

def fk_hip(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the hip
  frame given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the hip frame in the base frame
  """
  hip_angle, shoulder_angle, elbow_angle = joint_angles

  
  d1 = 0  
  a1 = 0  
  alpha1 = 0  

  d2 = 0  
  a2 = 0  
  alpha2 = 0  

  d3 = 0  
  a3 = 0  
  alpha3 = 0  

  T1 = np.array([
      [np.cos(hip_angle), -np.sin(hip_angle), 0, 0],
      [np.sin(hip_angle), np.cos(hip_angle), 0, 0],
      [0, 0, 1, d1],
      [0, 0, 0, 1]
  ])

  T2 = np.array([
      [np.cos(shoulder_angle), -np.sin(shoulder_angle), 0, 0],
      [0, 0, -1, -d2],
      [np.sin(shoulder_angle), np.cos(shoulder_angle), 0, 0],
      [0, 0, 0, 1]
  ])

  T3 = np.array([
      [np.cos(elbow_angle), -np.sin(elbow_angle), 0, 0],
      [0, 0, 1, d3],
      [-np.sin(elbow_angle), -np.cos(elbow_angle), 0, 0],
      [0, 0, 0, 1]
  ])

  hip_frame = np.dot(np.dot(T1, T2), T3)

  return hip_frame

def fk_shoulder(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the shoulder
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the shoulder frame in the base frame
  """

  # remove these lines when you write your solution
  
  # default_sphere_location = np.array([[0.15, 0.0, -0.1]])
  # shoulder_frame = np.block(
  #   [[np.eye(3), default_sphere_location.T], 
  #    [0, 0, 0, 1]])
  # return shoulder_frame

  hip_angle, shoulder_angle, elbow_angle = joint_angles

  d1 = 0  
  a1 = 0  
  alpha1 = 0  

  d2 = 0  
  a2 = 0  
  alpha2 = 0  

  T1 = np.array([
      [np.cos(hip_angle), -np.sin(hip_angle), 0, 0],
      [np.sin(hip_angle), np.cos(hip_angle), 0, 0],
      [0, 0, 1, d1],
      [0, 0, 0, 1]
  ])

  T2 = np.array([
      [np.cos(shoulder_angle), -np.sin(shoulder_angle), 0, 0],
      [0, 0, -1, -d2],
      [np.sin(shoulder_angle), np.cos(shoulder_angle), 0, 0],
      [0, 0, 0, 1]
  ])

  shoulder_frame = np.dot(T1, T2)

  
  shoulder_frame[0, 3] = 0.15  
  shoulder_frame[1, 3] = 0.0  
  shoulder_frame[2, 3] = -0.1  

  return shoulder_frame
  
def fk_elbow(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the elbow
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the elbow frame in the base frame
  """

  # remove these lines when you write your solution
  # default_sphere_location = np.array([[0.15, 0.1, -0.1]])
  # elbow_frame = np.block(
  #   [[np.eye(3), default_sphere_location.T], 
  #    [0, 0, 0, 1]])
  # return elbow_frame
  hip_angle, shoulder_angle, elbow_angle = joint_angles

  
  d1 = 0  
  a1 = 0  
  alpha1 = 0  

  d2 = 0  
  a2 = 0  
  alpha2 = 0  

  d3 = 0  
  a3 = 0 
  alpha3 = 0  

  T1 = np.array([
      [np.cos(hip_angle), -np.sin(hip_angle), 0, 0],
      [np.sin(hip_angle), np.cos(hip_angle), 0, 0],
      [0, 0, 1, d1],
      [0, 0, 0, 1]
  ])

  T2 = np.array([
      [np.cos(shoulder_angle), -np.sin(shoulder_angle), 0, 0],
      [0, 0, -1, -d2],
      [np.sin(shoulder_angle), np.cos(shoulder_angle), 0, 0],
      [0, 0, 0, 1]
  ])

  elbow_frame = np.dot(np.dot(T1, T2), np.array([[np.eye(3), [0.15, 0.1, -0.1]], [0, 0, 0, 1]]))

  return elbow_frame

def fk_foot(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the foot given 
  the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the end effector frame in the base frame
  """

  # remove these lines when you write your solution
  # default_sphere_location = np.array([[0.15, 0.2, -0.1]])
  # end_effector_frame = np.block(
  #   [[np.eye(3), default_sphere_location.T], 
  #    [0, 0, 0, 1]])
  # return end_effector_frame
  hip_angle, shoulder_angle, elbow_angle = joint_angles

  
  d1 = 0 
  a1 = 0  
  alpha1 = 0  

  d2 = 0  
  a2 = 0  
  alpha2 = 0  

  d3 = 0  
  a3 = 0  
  alpha3 = 0  

  T1 = np.array([
      [np.cos(hip_angle), -np.sin(hip_angle), 0, 0],
      [np.sin(hip_angle), np.cos(hip_angle), 0, 0],
      [0, 0, 1, d1],
      [0, 0, 0, 1]
  ])

  T2 = np.array([
      [np.cos(shoulder_angle), -np.sin(shoulder_angle), 0, 0],
      [0, 0, -1, -d2],
      [np.sin(shoulder_angle), np.cos(shoulder_angle), 0, 0],
      [0, 0, 0, 1]
  ])

  T3 = np.array([
      [np.cos(elbow_angle), -np.sin(elbow_angle), 0, 0],
      [0, 0, 1, d3],
      [-np.sin(elbow_angle), -np.cos(elbow_angle), 0, 0],
      [0, 0, 0, 1]
  ])

  end_effector_frame = np.dot(np.dot(np.dot(T1, T2), T3), np.array([[np.eye(3), [0.15, 0.2, -0.1]], [0, 0, 0, 1]]))
  return end_effector_frame
