import random

import os
import sys
nxg_path = os.environ.get('NXG_PATH')
print(nxg_path)
sys.path.append(nxg_path + '/core')
sys.path.append(nxg_path + '/configuration-setup')

from evaluation_isen import EvaluationInvSen
from learningModule import DataConfiguration, CreateTrainNN
import os.path
from os import path


def reachDest(benchmarkName, sims_f, dconfig_fname, model_fname):
    evalObject = EvaluationInvSen(dynamics=benchmarkName)
    evalObject.setSimsBound(30)
    dataObject = evalObject.setDataObject(dconfig_fname)
    print(dataObject.getStepSize())
    dest_traj = dataObject.generateTrajectories(samples=1)[0]
    ref_traj = dataObject.generateTrajectories(samples=1)[0]
    for t in range(1):
        time_step = round(random.uniform(0.3, 1.5), 2)
        print(time_step)
        print(dest_traj[0], dest_traj[int(time_step/dataObject.getStepSize())])
        dest = dest_traj[int(time_step/dataObject.getStepSize())]
        dests = [dest]
        print(ref_traj[0], ref_traj[int(time_step/dataObject.getStepSize())])
        evalObject.reachDestInvSenGr(dests=dests, d_time_steps=[time_step], threshold=0.003, correction_steps=[10],
                                   scaling_factors=[0.1], i_state=[ref_traj[0]], model_fname=model_fname, sims_f=sims_f)
        f_sims = evalObject.getFSimulationsCount()
        print(f_sims)


def train_network(benchmarkName, sims_f, d_config_fname, model_fname):
    epochs = 30
    dataObject = DataConfiguration(dynamics=benchmarkName, gradient_run=True)
    # dataObject.setNeighbors(5)
    dataObject.setStepSize(0.01)
    dataObject.setSteps(200)
    dataObject.setSamples(40)
    dataObject.setLowerBound([0.5, 0.5])
    dataObject.setUpperBound([1.5, 1.5])
    dataObject.generateTrajectories(scaling=0.01, sims_f=sims_f)
    # dataObject.showTrajectories(xindex=0, yindex=1)
    dataObject.createData(jumps=[1, 2, 5, 7, 11, 13, 17, 19], data_config_fname=d_config_fname)

    nnObject1 = CreateTrainNN(dynamics=benchmarkName, dnn_rbf='RBF')
    nnObject1.createInputOutput(data_object=dataObject, inp_vars=['x', 'xp', 'vp', 't'], out_vars=['v'])
    nnObject1.setEpochs(epochs)
    nnObject1.trainTestNN(optim='SGD', loss_fn='mae', layers=1, neurons=512, model_fname=model_fname)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    benchmarkName = 'OtherBenchC1'

    model_f_name = nxg_path + '/vp-2-v-' + benchmarkName + '.h5'
    dconfig_f_name = nxg_path + '/dobj-' + benchmarkName + '.txt'
    sims_f_name = nxg_path + '/sims-' + benchmarkName + '.txt'
    # if path.exists(sims_f_name):
    #     os.remove(sims_f_name)
    sims_f = open(sims_f_name, 'a')
    train_network(benchmarkName, sims_f, dconfig_f_name, model_f_name)
    reachDest(benchmarkName, sims_f, dconfig_f_name, model_f_name)
    sims_f.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
