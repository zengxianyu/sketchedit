import os
import torch
import scipy.ndimage
import pdb
import cv2
import mediapipe as mp
import numpy as np
import pytorch3d.ops

# For static images:
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5)


def mp_landmark(image, ratio=1.25):
    h,w,c = image.shape
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    #results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    results = face_mesh.process(image)
    if results is None or results.multi_face_landmarks is None:
        return None
    if len(results.multi_face_landmarks) < 1:
        return None
    lms = []
    for kp in results.multi_face_landmarks[0].landmark:
        lms.append([kp.x*w, kp.y*h])
    lms = np.array(lms)
    return lms


def corresponding_points_alignment(
    Xt,
    Yt,
    eps: float = 1e-9,
):
    # make sure we convert input Pointclouds structures to tensors
    n, dim = Xt.shape
    num_points = n

    # compute the centroids of the point sets
    Xmu = Xt.mean(0)[None]
    Ymu = Yt.mean(0)[None]

    # mean-center the point sets
    Xc = Xt - Xmu
    Yc = Yt - Ymu

    if (num_points < (dim + 1)):
        warnings.warn(
            "The size of one of the point clouds is <= dim+1. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    # compute the covariance XYcov between the point sets Xc, Yc
    XYcov = Xc.T@Yc

    # decompose the covariance matrix XYcov
    U, S, V = np.linalg.svd(XYcov)
    V = V.T

    # find the rotation matrix by composing U and V again
    R = U@(V.T)

    # estimate the scaling component of the transformation
    trace_ES = S.sum()
    Xcov = (Xc * Xc).sum()#/max(num_points,1)

    # the scaling component
    s = trace_ES / max(Xcov, eps)

    # translation component
    T = Ymu[0, :] - s * (Xmu@R)[0, :]

    return R, T, s


if __name__ == "__main__":
    path_in = "/media/zeng/Samsung_T51/10_tiktok"
    from tqdm import tqdm
    #path_in = "./images"
    #file_list = os.listdir(path_in)
    #file_list = [f for f in file_list if f.endswith(".jpg")]

    with open("/media/zeng/Samsung_T51/txt/512_v1v2.txt","r") as f:
        file_list = f.readlines()
    file_list = [n.strip("\n") for n in file_list]

    file_list = [f"{path_in}/{f}" for f in file_list]

    lms_avg = np.load("./lms_avg_dense.npy")
    c_avg = lms_avg.mean(0)[None,...]
    lms_avg_ctr = lms_avg-c_avg
    H,W = 256, 256

    lms_all = []

    #with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
      for idx, file in tqdm(enumerate(file_list)):
        image = cv2.imread(file)
        h,w,c = image.shape
        lms = mp_landmark(image)
        R, T, s = corresponding_points_alignment(lms, lms_avg)
        image0 = image
        rot_mat = np.concatenate((R.T*s,T[...,None]),1)
        image = cv2.warpAffine(image, rot_mat, (256,256))
        rot_mat = np.concatenate((R/s,-R/s@T[...,None]),1)
        mask = np.ones_like(image)
        # Y = X@R*s+T
        # X = (Y-T)@R.T*1/s
        image = cv2.warpAffine(image, rot_mat, (w,h))
        mask = cv2.warpAffine(mask, rot_mat, (w,h))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel)
        #lms_rot = lms@R.T+T

        for x,y  in lms:
            image = cv2.circle(image, (int(x),int(y)), 2, [255,0,0], 2)
        image = image*mask+image0*(1-mask)
        cv2.imwrite('./output/' + str(idx) + '.png', image)
        pdb.set_trace()
