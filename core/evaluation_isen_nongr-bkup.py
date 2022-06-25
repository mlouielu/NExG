from evaluation import Evaluation
from sampler import generateRandomStates
from matplotlib.path import Path
import matplotlib.patches as patches
import time
import numpy as np
import matplotlib.pyplot as plt
from frechet import norm
from evaluation_plot import plotInvSenResults, plotInvSenResultsAnimate, plotInvSenStaliroResults, plotInvSenReachSetCoverage
# version 1 RRT
from rrtv1 import RRTV1

# version 2 RRT
import sys
sys.path.append('./rrtAlgorithms/src/')
from rrt.rrt import RRT
from search_space.search_space import SearchSpace
from utilities.plotting import Plot


class EvaluationInvSenNonGr(Evaluation):

    def __init__(self, dynamics='None', layers=1, neurons=512, dnn_rbf='RBF', act_fn='ReLU', norm_status=False):
        Evaluation.__init__(self, dynamics=dynamics, sensitivity='Inv', dnn_rbf=dnn_rbf, layers=layers,
                            neurons=neurons, act_fn=act_fn, grad_run=False, norm_status=norm_status)
        self.f_simulations_count = 0
        self.f_dist = None
        self.f_rel_dist = None
        self.staliro_run = False
        self.usafelowerBoundArray = []
        self.usafeupperBoundArray = []
        self.usafe_centroid = None
        self.time_steps = []
        self.always_spec = False
        self.f_simulations = []
        self.best_trajectory = []

    def getFSimulationsCount(self):
        return self.f_simulations_count

    def getFDistance(self):
        return self.f_dist

    def getFRelDistance(self):
        return self.f_rel_dist

    def setStaliroRun(self):
        self.staliro_run = True

    def setUnsafeSet(self, lowerBound, upperBound):
        self.usafelowerBoundArray = lowerBound
        self.usafeupperBoundArray = upperBound
        self.usafe_centroid = np.zeros(self.data_object.dimensions)
        for dim in range(self.data_object.dimensions):
            self.usafe_centroid[dim] = (lowerBound[dim] + upperBound[dim])/2
        self.staliro_run = True

    def generateRandomUnsafeStates(self, samples):
        states = generateRandomStates(samples, self.usafelowerBoundArray, self.usafeupperBoundArray)
        return states

    def check_for_usafe_contain_eventual(self, traj):
        # print(" Checking for containment " + str(state))
        time_steps = self.time_steps
        # time_steps = [d_time_step]
        found_time_step = -1
        for time_step in time_steps:
            is_contained = True
            state = traj[time_step]
            for dim in range(self.data_object.dimensions):
                l_bound = self.usafelowerBoundArray[dim]
                u_bound = self.usafeupperBoundArray[dim]
                if l_bound < state[dim] < u_bound:
                    continue
                else:
                    is_contained = False
                    break

            if is_contained:
                found_time_step = time_step
                # print(" Found time step " + str(found_time_step))
                break

        return found_time_step

    def check_for_usafe_contain_always(self, traj):
        # print(" Checking for containment " + str(state))
        time_steps = self.time_steps
        # time_steps = [d_time_step]
        found_time_step = -1
        is_contained = True
        for time_step in time_steps:
            state = traj[time_step]
            for dim in range(self.data_object.dimensions):
                l_bound = self.usafelowerBoundArray[dim]
                u_bound = self.usafeupperBoundArray[dim]
                if l_bound < state[dim] < u_bound:
                    continue
                else:
                    is_contained = False
                    break

            if is_contained is False:
                break

        if is_contained:
            found_time_step = time_steps[1]
            print(" Trajectory found ")

        return found_time_step

    def compute_robust_state_wrt_axes(self, state):

        robustness = 100.0

        for dim in range(self.data_object.dimensions):
            l_bound = self.usafeupperBoundArray[dim]
            u_bound = self.usafeupperBoundArray[dim]

            dist_1 = abs(state[dim] - l_bound)
            dist_2 = abs(u_bound - state[dim])

            if dist_1 < robustness:
                robustness = dist_1
            if dist_2 < robustness:
                robustness = dist_2

        return robustness

    def compute_robust_wrt_axes(self, traj):

        found_time_step = self.check_for_usafe_contain_eventual(traj)

        robustness = 100

        if found_time_step != -1:
            state = traj[found_time_step]
            robustness = self.compute_robust_state_wrt_axes(state)
        else:
            # state = traj[self.time_steps[int(len(self.time_steps)/2)]]
            for time_step in self.time_steps:
                state = traj[time_step]
                cur_robustness = self.compute_robust_state_wrt_axes(state)
                if cur_robustness < robustness:
                    robustness = cur_robustness

        if found_time_step != -1:
            robustness = robustness * -1

        return robustness

    def predict_falsifying_time_step(self, dest, traj, mid=True):
        d_time_step = -1
        if mid is True:
            d_time_step = int((self.time_steps[0] + self.time_steps[len(self.time_steps)-1])/2)
            # d_time_step = random.randint(d_time_steps[0], d_time_steps[1])
        else:
            min_dist = 100
            for t_idx in range(self.time_steps[0], self.time_steps[len(self.time_steps)-1]):
                current_dist = norm(dest-traj[t_idx], 2)
                if current_dist < min_dist:
                    min_dist = current_dist
                    d_time_step = t_idx
        return d_time_step

    def reachDestInvSen(self, dests=None, d_time_steps=None, threshold=0.01, correction_steps=[1],
                          scaling_factors=[1], i_state=None, true_inv_sen=None):

        ref_traj = self.data_object.generateTrajectories(r_states=i_state)[0]
        dest = dests[0]

        if d_time_steps is None:
            min_dist = 200.0
            min_idx = 0
            for idx in range(len(ref_traj)):
                curr_dist = norm(dest - ref_traj[idx], 2)
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    min_idx = idx
            print(min_dist, min_idx)
            d_time_step = min_idx
        elif len(d_time_steps) == 1:
            d_time_step = d_time_steps[0]
        else:
            for t_idx in range(d_time_steps[0]+1, d_time_steps[1]+1):
                self.time_steps.append(t_idx)
            d_time_step = self.predict_falsifying_time_step(dest, ref_traj, False)
        print(d_time_step)

        assert self.data_object is not None

        trained_model = self.getModel()
        if trained_model is None:
            return

        paths_list = [[[ref_traj[d_time_step], dest]]]

        for paths in paths_list:
            # print(paths)
            for s_factor in scaling_factors:

                for steps in correction_steps:
                    if steps == 1:
                        start_time = time.time()
                        self.reachDestInvBaseline(ref_traj=ref_traj, paths=paths, d_time_step=d_time_step,
                                                  threshold=threshold, model_v=trained_model, sims_bound=self.sims_bound,
                                                  scaling_factor=s_factor, dynamics=self.dynamics, true_inv_sen=true_inv_sen)
                        print("Time taken: " + str(time.time() - start_time))
                    else:
                        start_time = time.time()
                        self.reachDestInvNonBaseline(ref_traj=ref_traj, paths=paths, d_time_step=d_time_step,
                                                     threshold=threshold,  model_v=trained_model, correction_steps=steps,
                                                     scaling_factor=s_factor, sims_bound=self.sims_bound,
                                                     dynamics=self.dynamics)
                        print("Time taken: " + str(time.time() - start_time))

    '''
    ReachDestination for correction period 1 (without axes aligned).
    '''
    def reachDestInvBaseline(self, ref_traj, paths, d_time_step, threshold, model_v, sims_bound, scaling_factor, dynamics, true_inv_sen=None):

        assert dynamics is not None
        n_paths = len(paths)
        x_val = ref_traj[0]
        trajectories = [ref_traj]
        rrt_dests = []
        dimensions = self.data_object.getDimensions()

        dest_traj_start_pt = None
        if true_inv_sen is not None:
            dest_traj_start_pt = ref_traj[0] + true_inv_sen
            print(x_val, dest_traj_start_pt)

        for path_idx in range(n_paths):
            path = paths[path_idx]
            xp_val = path[0]
            dest = path[1]
            rrt_dests.append(dest)
            print("***** path idx " + str(path_idx) + " correction steps 1")
            v_val = dest - x_val

            # print("Actual v norm " + str(norm(actual_v, 2)))
            vp_val = dest - xp_val
            print(dest, xp_val)
            t_val = d_time_step
            vp_norm = norm(vp_val, 2)
            vp_val_normalized = [val / vp_norm for val in vp_val]  # Normalized
            dist = vp_norm
            print("Starting distance: " + str(dist))
            original_distance = dist
            min_dist = dist
            # best_state = x_val
            sims_count = 1
            min_simulation = sims_count
            best_trajectory = ref_traj
            vp_vals = [vp_val]
            v_vals = [v_val]

            while dist > threshold and sims_count < sims_bound:
                if original_distance < 0.005:
                    break

                if self.staliro_run is True and sims_count < 2:

                    if self.check_for_usafe_contain_eventual(ref_traj) != -1:
                        print("*********** Initial sample falsified ************")
                        break

                if self.norm_status is False:
                    data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val, t_val)
                else:
                    data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val_normalized, t_val)

                predicted_v = self.evalModel(input=data_point, eval_var='v', model=model_v)
                # print("predicted v norm " + str(norm(predicted_v, 2)))
                if true_inv_sen is not None:
                    if self.norm_status is False:
                        # print("delta is " + str((norm(predicted_v - true_inv_sen, 2) - 0.0001 * norm(true_inv_sen, 2))))
                        predicted_v_scaled = [val * scaling_factor for val in predicted_v]
                    else:
                        true_inv_sen_norm = norm(true_inv_sen, 2)
                        # predicted_v_norm = predicted_v * true_inv_sen_norm
                        # print("delta is " + str((norm(predicted_v_norm - true_inv_sen, 2) - 0.0001 * norm(true_inv_sen, 2))))
                        predicted_v_scaled = [val * scaling_factor * true_inv_sen_norm for val in predicted_v]
                else:
                    if self.norm_status is False:
                        predicted_v_scaled = [val * scaling_factor for val in predicted_v]
                    else:
                        predicted_v_scaled = [val * scaling_factor * vp_norm for val in predicted_v]
                new_init_state = [self.check_for_bounds(x_val + predicted_v_scaled)]
                new_traj = self.data_object.generateTrajectories(r_states=new_init_state)[0]
                x_val = new_traj[0]

                if true_inv_sen is not None and dest_traj_start_pt is not None:
                    true_inv_sen = dest_traj_start_pt - x_val
                # print("Actual v norm " + str(norm(actual_v, 2)))
                xp_val = new_traj[d_time_step]

                v_val = predicted_v_scaled
                vp_val = dest - xp_val

                vp_norm = norm(vp_val, 2)
                vp_val_normalized = [val / vp_norm for val in vp_val]  # Normalized

                dist = norm(vp_val, 2)
                t_val = d_time_step
                vp_vals.append(vp_val)
                v_vals.append(predicted_v_scaled)

                trajectories.append(new_traj)
                sims_count = sims_count + 1
                # A falsifying trajectory found
                if self.staliro_run and self.check_for_usafe_contain_eventual(new_traj) != -1:
                    best_trajectory = new_traj
                    # print("Found the time step **** ")
                    break

                # if dist < min_dist:
                min_dist = dist
                min_simulation = sims_count
                best_trajectory = new_traj

            if self.staliro_run:
                min_dist = self.compute_robust_wrt_axes(best_trajectory)
                print("Best robustness " + str(min_dist))
                print("Final simulation: " + str(sims_count))
            else:
                print("Final distance " + str(dist))
                print("Final relative distance " + str(dist/original_distance))
                print("Min relative distance " + str(min_dist/original_distance))
                print("Min simulation: " + str(min_simulation))
                print("Final simulation: " + str(sims_count))
                print("Min distance " + str(min_dist))

            self.f_simulations_count = sims_count
            self.f_dist = min_dist
            self.f_rel_dist = min_dist/original_distance
            print(len(trajectories))
            self.f_simulations = trajectories
            self.best_trajectory = best_trajectory

            if self.staliro_run:
                plotInvSenStaliroResults(trajectories, d_time_step, best_trajectory, self.usafeupperBoundArray,
                self.usafelowerBoundArray, self.data_object)
            else:
                plotInvSenResults(trajectories, rrt_dests, d_time_step, dimensions, best_trajectory)
                # plotInvSenResultsAnimate(trajectories, rrt_dests, d_time_step, v_vals, vp_vals)


    '''
    Reach Destination implementation with course correction.
    '''
    def reachDestInvNonBaseline(self, ref_traj, paths, d_time_step, threshold, model_v, correction_steps, sims_bound,
                                scaling_factor, rand_area=None, dynamics=None):

        n_paths = len(paths)
        x_val = ref_traj[0]
        dimensions = self.data_object.getDimensions()
        # xp_vals_list = []
        # x_vals_list = []
        # xp_val = paths[0][0]
        # print(x_val, xp_val)
        trajectories = [ref_traj]
        rrt_dests = []
        for path_idx in range(n_paths):
            path = paths[path_idx]
            xp_val = path[0]
            dest = path[1]
            rrt_dests.append(dest)
            print("***** path idx " + str(path_idx) + " s_factor " + str(scaling_factor) + " correction steps " +
                  str(correction_steps))
            sims_count = 1
            x_vals = []
            xp_vals = []
            v_val = dest - x_val
            vp_val = dest - xp_val
            vp_norm = norm(vp_val, 2)
            dist = vp_norm
            original_distance = dist
            print("Starting distance: " + str(dist))
            min_dist = dist
            best_trajectory = ref_traj
            min_simulation = sims_count

            while dist > threshold and sims_count <= sims_bound:

                if self.staliro_run is True and sims_count < 2:

                    if self.always_spec is False and self.check_for_usafe_contain_eventual(ref_traj) != -1:
                        print("*********** Initial sample falsified ************")
                        break
                    elif self.always_spec is True and self.check_for_usafe_contain_always(ref_traj) != -1:
                        print("*********** Initial sample falsified ************")
                        break

                x_vals.append(x_val)
                xp_vals.append(xp_val)
                t_val = d_time_step
                vp_val_normalized = [val / vp_norm for val in vp_val]  # Normalized
                vp_val_scaled = [val * scaling_factor for val in vp_val]
                step = 0
                prev_pred_dist = None
                # pred_dist = dist

                while step < correction_steps:
                    data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val_normalized, t_val)
                    predicted_v = self.evalModel(input=data_point, eval_var='v', model=model_v)
                    predicted_v_scaled = [val * scaling_factor * vp_norm for val in predicted_v]
                    new_init_state = [self.check_for_bounds(x_val + predicted_v_scaled)]
                    x_val = new_init_state[0]
                    xp_val = xp_val + vp_val_scaled
                    v_val = predicted_v_scaled
                    vp_val = dest - xp_val
                    vp_norm = norm(vp_val, 2)
                    x_vals.append(x_val)
                    xp_vals.append(xp_val)
                    # pred_dist = vp_norm
                    vp_val_normalized = [val/vp_norm for val in vp_val]
                    # print("new distance: " + str(dist))
                    t_val = d_time_step
                    step += 1
                    # if prev_pred_dist is not None and prev_pred_dist < pred_dist:
                    #     x_vals.pop()
                    #     xp_vals.pop()
                    #     break
                    # prev_pred_dist = pred_dist

                sims_count = sims_count + 1

                new_traj = self.data_object.generateTrajectories(r_states=[x_vals[len(x_vals) - 1]])[0]
                x_val = new_traj[0]
                xp_val = new_traj[d_time_step]
                vp_val = dest - xp_val
                vp_norm = norm(vp_val, 2)
                dist = vp_norm

                trajectories.append(new_traj)

                # if dist < min_dist:
                min_dist = dist
                best_trajectory = new_traj
                min_simulation = sims_count

                # A falsifying trajectory found
                if self.staliro_run and self.always_spec is False and self.check_for_usafe_contain_eventual(new_traj) != -1:
                    best_trajectory = new_traj
                    # print("Found the time step **** ")
                    break
                elif self.staliro_run and self.always_spec is True and self.check_for_usafe_contain_always(new_traj) != -1:
                    best_trajectory = new_traj
                    print("Found the time step **** ")
                    break
                elif self.staliro_run and self.always_spec is True and self.check_for_usafe_contain_always(new_traj) == -1:
                    new_dest = self.generateRandomUnsafeStates(1)[0]
                    new_time_step = self.predict_falsifying_time_step(new_dest, new_traj, False)
                    new_dist = norm(new_traj[new_time_step]-new_dest, 2)
                    if new_dist < min_dist:
                        print("Setting new time step to " + str(new_time_step))
                        d_time_step = new_time_step
                        best_trajectory = new_traj
                        min_dist = new_dist
                        dest = new_dest

            min_rel_dist = min_dist / original_distance

            if self.staliro_run:
                min_dist = self.compute_robust_wrt_axes(best_trajectory)
                print("Best robustness " + str(min_dist))

            if self.staliro_run is False:
                print("Final distance " + str(dist))
                print("Final relative distance " + str(dist / original_distance))
                print("Min relative distance: " + str(min_rel_dist))
                print("Min simulation: " + str(min_simulation))
            print("Final simulation: " + str(sims_count))
            self.f_simulations_count = sims_count
            self.f_dist = min_dist
            self.f_rel_dist = min_rel_dist

            # self.plotInvSenResultsRRTv2(trajectories, dest, d_time_step, dimensions, rand_area, path_idx, dynamics)

            if self.staliro_run:
                plotInvSenStaliroResults(trajectories, d_time_step, best_trajectory, self.usafeupperBoundArray,
                                         self.usafelowerBoundArray, self.data_object)
            else:
                plotInvSenResults(trajectories, rrt_dests, d_time_step, dimensions, best_trajectory)

    def partition_init_set(self, partitions=5):
        x_index = 0
        y_index = 1
        i_x_min = self.data_object.lowerBoundArray[x_index]
        i_x_max = self.data_object.upperBoundArray[x_index]
        i_y_min = self.data_object.lowerBoundArray[y_index]
        i_y_max = self.data_object.upperBoundArray[y_index]

        x_dim_partition_size = (i_x_max - i_x_min)/partitions
        y_dim_partition_size = (i_y_max - i_y_min)/partitions
        lowerBounds = []
        upperBounds = []

        x_upper = i_x_min

        parity = 0
        for idx in range(partitions):
            x_lower = x_upper
            x_upper = x_lower+x_dim_partition_size

            y_upper = i_y_min

            lowerBounds_temp = []
            upperBounds_temp = []
            for idy in range(partitions):
                y_lower = y_upper
                y_upper = y_lower + y_dim_partition_size
                lowerBound = [x_lower, y_lower]
                upperBound = [x_upper, y_upper]
                lowerBounds_temp.append(lowerBound)
                upperBounds_temp.append(upperBound)

            if parity == 0:
                parity = 1
            elif parity == 1:
                lowerBounds_temp = lowerBounds_temp[::-1]
                upperBounds_temp = upperBounds_temp[::-1]
                parity = 0

            for idz in range(len(lowerBounds_temp)):
                lowerBounds.append(lowerBounds_temp[idz])
                upperBounds.append(upperBounds_temp[idz])

        # print(lowerBounds, upperBounds)
        assert len(lowerBounds) == len(upperBounds)
        return lowerBounds, upperBounds
        # print(lowerBounds)
        # print(upperBounds)

        # dataObject.setLowerBound([0.5, 0.5])
        # dataObject.setUpperBound([1.5, 1.5])
    def coverage(self, d_time_steps, samples=0, partitions=5):

        lowerBounds = None
        upperBounds = None
        if samples == 0:
            lowerBounds, upperBounds = self.partition_init_set(partitions=partitions)
            r_states = []
            for idx in range(len(lowerBounds)):
                r_state = generateRandomStates(3, lowerBounds[idx], upperBounds[idx])
                for state in r_state:
                    r_states.append(state)
            # print(r_states)
            ref_trajs = self.data_object.generateTrajectories(r_states=r_states)
            ref_trajs.append(ref_trajs[0])
        else:
            ref_trajs = self.data_object.generateTrajectories(samples=samples)
            ref_trajs.append(ref_trajs[0])

        # ref_traj = self.data_object.generateTrajectories(samples=1)
        # scaling = 0.2
        # vecs_in_unit_circle = self.data_object.generate_points_in_circle(n_samples=samples)
        #
        # # print(vecs_in_unit_circle)
        # delta_vecs = []
        # for vec in vecs_in_unit_circle:
        #     delta_vec = [val * scaling for val in vec]
        #     delta_vecs.append(delta_vec)
        # # print(delta_vecs)
        # r_state = ref_traj[0]
        # ref_trajs = []
        # for delta_vec in delta_vecs:
        #     # print(r_state, delta_vec)
        #     neighbor_state = [r_state[i] + delta_vec[i] for i in range(len(delta_vec))]
        #     neighbor_trajectory = self.data_object.generateTrajectories(r_states=neighbor_state)[0]
        #     ref_trajs.append(neighbor_trajectory)
        # ref_trajs.append(ref_traj[0])

        print(len(ref_trajs))

        for d_time_step in d_time_steps:
            f_trajs = []
            best_trajs = []

            f_abs_dist = 0.0
            f_sims = 0
            for idx in range(len(ref_trajs)-1):
                # print(ref_trajs[idx+1])
                print(" **** idx **** " + str(idx))
                dests = [ref_trajs[idx+1][d_time_step]]
                self.reachDestInvSen(dests=dests, d_time_steps=[d_time_step], threshold=0.004, i_state=[ref_trajs[idx][0]])
                f_abs_dist = f_abs_dist + self.f_dist
                f_sims = f_sims + self.f_simulations_count
                trajectories = self.f_simulations
                best_trajectory = self.best_trajectory
                f_trajs.append(trajectories)
                best_trajs.append(best_trajectory)
            print(len(f_trajs), len(best_trajs))
            print(" Mean final distance " + str(f_abs_dist/len(ref_trajs)))
            print(" Mean simulations count " + str(f_sims/len(ref_trajs)))
            plotInvSenReachSetCoverage(f_trajs, d_time_step, self.data_object, lowerBounds, upperBounds)

