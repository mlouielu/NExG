import matplotlib
import numpy as np

import os
import sys
import random
nxg_path = os.environ.get('NXG_PATH')
sys.path.append(nxg_path + 'core')
sys.path.append(nxg_path + 'configuration-setup')
sys.path.append(nxg_path + 'core/rrtAlgorithms/src/')

from evaluation_isen import EvaluationInvSen
from evaluation_isen_nongr import EvaluationInvSenNonGr


def animation_gr():
    # evalObject = EvaluationInvSen(dynamics='OtherBenchC2')
    # evalObject.setSimsBound(40)
    # dataObject = evalObject.setDataObject()
    # dataObject.setSteps(201)

    # For correction periods 1 and 3
    # dest_traj = dataObject.generateTrajectories(r_states=[[0.74224485, 1.1416229]])[0]
    # dests = [[0.64465355, -0.45927538]]
    # i_state = [[1.17860713, 0.89770518]]
    # time_step = 180
    # evalObject.reachDestInvSen(dests=dests, d_time_steps=[time_step], threshold=0.01,
    #                            correction_steps=[1], scaling_factors=[0.1], i_state=i_state)

    # dests = [[1.13453814, 0.17141537]]
    # i_state = [[0.58046985, 0.88060913]]
    # time_step = 63  # Tried with 77 and 85, orig 51
    # evalObject.reachDestInvSen(dests=dests, d_time_steps=[time_step], threshold=0.005,
    #                            correction_steps=[1], scaling_factors=[0.05], i_state=i_state)

    # For comparison with neuralex
    # dests = [[1.0508023,  -0.50354116]]
    # i_state = [[0.73370841, 0.52553085]]
    # time_step = 139
    # evalObject.reachDestInvSen(dests=dests, d_time_steps=[time_step], threshold=0.003,
    #                            correction_steps=[2], scaling_factors=[0.2], i_state=i_state)

    # For axis-alignment -1
    # dests = [[1.0264224,  0.44886683]]
    # i_state = [[1.04854324, 0.91869626]]
    # time_step = 59  # orig 60
    # evalObject.reachDestInvSen(dests=dests, d_time_steps=[time_step], threshold=0.02,
    #                            correction_steps=[1], scaling_factors=[0.05], i_state=i_state)

    # For axis-alignment -2
    # dests = [[ 0.99629181, -0.03765987]]
    # i_state = [[0.87942858, 1.18787155]]
    # time_step = 90
    # evalObject.reachDestInvSen(dests=dests, d_time_steps=[time_step], threshold=0.02,
    #                            correction_steps=[1], scaling_factors=[0.1], i_state=i_state)

    # For boundary
    # evalObject = EvaluationInvSen(dynamics='OtherBenchC9')
    # evalObject.setSimsBound(40)
    # dataObject = evalObject.setDataObject()
    # dataObject.setSteps(201)
    #
    # dests = [[-0.33247966, -0.62283303,  0.36890848, -0.18661113]]
    # i_state = [[0.9510333,  -0.98411012, -0.29126114,  0.78761868]]
    # time_step = 130
    # evalObject.reachDestInvSen(dests=dests, d_time_steps=[time_step], threshold=0.03,
    #                            correction_steps=[1], scaling_factors=[0.2], i_state=i_state)

    evalObject = EvaluationInvSen(dynamics='OtherBenchC9Tanh')
    evalObject.setSimsBound(30)
    dataObject = evalObject.setDataObject()
    dataObject.setSteps(201)
    #
    # dests = [[-0.87598929,  0.49474659,  0.33004758, -0.07435994]]
    # i_state = [[-0.72921585, -0.14265032,  0.45697502, -0.45421423]]
    # time_step = 85  # orig 103
    # evalObject.reachDestInvSen(dests=dests, d_time_steps=[time_step], threshold=0.005,
    #                            correction_steps=[1], scaling_factors=[0.2], i_state=i_state)

    # For different time steps
    dests = [[-0.77858913,  0.39901871, -0.09664246, -0.36331666]]
    i_state = [[-0.5653285,  -0.40737782,  0.61172352, -0.32746681]]
    time_step = 75  # orig 88 (checked with 110, 75)
    evalObject.reachDestInvSen(dests=dests, d_time_steps=[time_step], threshold=0.005,
                               correction_steps=[1], scaling_factors=[0.2], i_state=i_state)


def animation_nongr():
    evalObject = EvaluationInvSenNonGr(dynamics='OtherBenchC2', norm_status=True)
    evalObject.setSimsBound(30)
    dataObject = evalObject.setDataObject()
    dataObject.setSteps(201)

    dest_traj = dataObject.generateTrajectories(samples=1)[0]
    ref_traj = dataObject.generateTrajectories(samples=1)[0]
    time_step = 139
    print(dest_traj[0], dest_traj[time_step])
    dest = dest_traj[time_step]
    dests = [dest]
    print(ref_traj[0], ref_traj[time_step])
    i_state = [ref_traj[0]]
    dests = [[1.0508023,  -0.50354116]]
    i_state = [[0.73370841, 0.52553085]]
    evalObject.reachDestInvSen(dests=dests, d_time_steps=[time_step], threshold=0.005,
                               correction_steps=[1], scaling_factors=[1], i_state=i_state)

    # [0.66606523 0.57838785][0.9136342 - 0.36513212]
    # [0.75612213 0.98470033][1.20102879 - 0.35505115]

def animation_gr_falsify():
    evalObject = EvaluationInvSen(dynamics='OtherBenchC2')
    evalObject.setSimsBound(40)
    dataObject = evalObject.setDataObject()
    dataObject.setSteps(151)
    dataObject.setLowerBound([0.8, 0.9])
    dataObject.setUpperBound([1.2, 1.2])
    lowerBound = [1.5, 0.2]
    upperBound = [1.55, 0.25]
    evalObject.setUnsafeSet([lowerBound, upperBound])
    stl_time_interval = [0.7, 0.9]

    ref_traj = dataObject.generateTrajectories(samples=1)[0]
    dest = evalObject.generateRandomUnsafeStates(1)
    dests = dest
    print(dests)
    evalObject.reachDestInvSen(dests=dests, d_time_steps=stl_time_interval, threshold=0.0, correction_steps=[2],
                               scaling_factors=[0.2], i_state=[ref_traj[0]])

if __name__ == '__main__':
    animation_gr()
    # animation_nongr()
    # animation_gr_falsify()
