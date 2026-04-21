import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
if __name__ == "__main__":
    fa = FrankaArm()

    fa.reset_joints()
    fa.goto_joints([-0.0324427, -0.19648132, -0.21504541, -2.00733578, -1.66687944,  1.77458245, -0.47951309]) #up
    fa.goto_joints([-0.03502555, -0.1531992,  -0.21829118, -2.07086378, -1.66688138,  1.77438752, -0.39818296]) #pick
    fa.close_gripper()
    fa.goto_joints([-0.0324427, -0.19648132, -0.21504541, -2.00733578, -1.66687944,  1.77458245, -0.47951309]) #up
    fa.goto_joints([0.00903874, -0.27583963,  0.14265124, -2.07148438, -1.54980572,  1.41853647, -0.55047169]) #out
    fa.reset_joints()    
    fa.goto_joints([0.03365152, -0.04303588, -0.06966402, -2.73393386, -0.02052933,  2.70163491, 0.82759907]) # down
    fa.open_gripper()

      