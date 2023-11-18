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
  hip_frame = homogenous_transformation_matrix([0, 0, 1], hip_angle, [0, 0, 0])

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
  hip_matrix = fk_hip(joint_angles)
  # calculate the position of the shoulder frame in the base frame
  shoulder_frame = np.dot(hip_matrix, homogenous_transformation_matrix([0, 1, 0], shoulder_angle, [0, -HIP_OFFSET, 0]))
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

  # calculate the position of the elbow frame in the base frame
  hip_angle, shoulder_angle, elbow_angle = joint_angles
  shoulder_to_hip = fk_shoulder(joint_angles)
  elbow_to_shoulder = homogenous_transformation_matrix([0, 1, 0], elbow_angle, [0, 0, UPPER_LEG_OFFSET])
  elbow_frame = np.dot(shoulder_to_hip, elbow_to_shoulder)
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
  hip_ange, shoulder_angle, elbow_angle = joint_angles
  elbow_to_base = fk_elbow(joint_angles)
  foot_to_elbow = homogenous_transformation_matrix([0, 0, 0], 0, [0, 0, LOWER_LEG_OFFSET])
  end_effector_frame = np.dot(elbow_to_base, foot_to_elbow)
  return end_effector_frame