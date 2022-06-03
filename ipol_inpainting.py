"""
Inpainting using total variation with regularization factor minimization
"""

import numpy as np
import cvxpy
import cv2


def total_variation(img):
    """
    find the sum of the total variation and the regularization factor of a given image
    :param img: the image
    :return: the total variation
    """
    global damaged, mask, lamda
    dx = cvxpy.vec((img[1:, :-1] - img[:-1, :-1]))
    dy = cvxpy.vec((img[:-1, 1:] - img[:-1, :-1]))
    D = cvxpy.vstack((dx, dy))
    norm = cvxpy.norm(D, p=2, axis=0)  # norm l2
    # multiply with mask to get rid of black pixels, black pixel = 0
    known_diff_norm = cvxpy.multiply((img - damaged), 1 - (mask / 255)) ** 2
    return cvxpy.sum(norm) + (lamda / 2) * cvxpy.sum(known_diff_norm)


def image_inpaint(damaged, mask, verbose=False, max_iters=100000):
    """
    inpaint the damaged image given the mask of the damaged regions
    using total variation minimization
    :param damaged: the damaged image
    :param mask: the mask of the damaged regions where black pixels denoted such regions
    :param verbose: whether the output of SCS is needed
    :param max_iters: the maximum number of iterations of SCS
    :return: the inpainted image
    """
    rows, cols = np.where(mask == 0)  # where pixel is black
    # prepare problem inputs
    x = cvxpy.Variable(damaged.shape)
    objective = cvxpy.Minimize(total_variation(x))
    know = x[rows, cols] == damaged[rows, cols]  # only change damaged pixels
    constraints = [0 <= x, x <= 255, know]
    # solve
    problem = cvxpy.Problem(objective, constraints)
    problem.solve(solver=cvxpy.SCS, verbose=verbose, max_iters=max_iters)
    return x.value


lamda = 50
damaged = cv2.imread('images/damaged.jpg', cv2.IMREAD_GRAYSCALE)
mask = cv2.imread('images/mask.jpg', cv2.IMREAD_GRAYSCALE)
inpainted = image_inpaint(damaged, mask, verbose=True, max_iters=1000)
cv2.imwrite("images/ipol_inpainted.jpg", inpainted)
