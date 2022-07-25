from evaluation import Evaluation
import time
import numpy as np
import matplotlib.pyplot as plt
from frechet import norm
from evaluation_plot import plotInvSenResults, plotInvSenResultsAnimate, plotInvSenStaliroResults
from sampler import generateRandomStates

from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate
import random

# version 1 RRT
from rrtv1 import RRTV1

# version 2 RRT
import sys
# sys.path.append('./rrtAlgorithms/src/')
# from rrt.rrt import RRT
# from search_space.search_space import SearchSpace
# from utilities.plotting import Plot


def random_points_in_polygon(polygon, k):
    "Return list of k points chosen uniformly at random inside the polygon."
    areas = []
    transforms = []
    for t in triangulate(polygon):
        areas.append(t.area)
        (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
        transforms.append([x1 - x0, x2 - x0, y2 - y0, y1 - y0, x0, y0])
    points = []
    for transform in random.choices(transforms, weights=areas, k=k):
        x, y = [random.random() for _ in range(2)]
        if x + y > 1:
            p = Point(1 - x, 1 - y)
        else:
            p = Point(x, y)
        points.append(affine_transform(p, transform))
    return points


class EvaluationInvSen(Evaluation):

    def __init__(self, dynamics='None', layers=1, neurons=512, dnn_rbf='RBF', act_fn='ReLU', norm_status=True):
        Evaluation.__init__(self, dynamics=dynamics, sensitivity='Inv', dnn_rbf=dnn_rbf, layers=layers,
                            neurons=neurons, act_fn=act_fn, grad_run=True, norm_status=norm_status)
        self.f_simulations_count = 0
        self.f_dist = None
        self.f_rel_dist = None
        self.staliro_run = False
        self.usafelowerBoundArray = []
        self.usafeupperBoundArray = []
        self.usafeVerts = []
        self.usafe_centroid = None
        self.time_steps = []
        self.always_spec = False
        self.f_simulations = None
        self.best_trajectory = None

    def getFSimulationsCount(self):
        return self.f_simulations_count

    def getFDistance(self):
        return self.f_dist

    def getFRelDistance(self):
        return self.f_rel_dist

    def setStaliroRun(self):
        self.staliro_run = True

    def setUnsafeSet(self, bounds=None, verts=None, staliro_run=True):
        if bounds is not None:
            assert self.data_object.dimensions == len(bounds[0])
            assert self.data_object.dimensions == len(bounds[1])
            self.usafelowerBoundArray = bounds[0]
            self.usafeupperBoundArray = bounds[1]
            self.usafe_centroid = np.zeros(self.data_object.dimensions)
            for dim in range(self.data_object.dimensions):
                self.usafe_centroid[dim] = (bounds[0][dim] + bounds[1][dim])/2
        elif verts is not None:   # Used in coverage for now
            self.usafeVerts = verts
        self.staliro_run = staliro_run

    def generateRandomUnsafeStates(self, samples):
        states = []
        if len(self.usafelowerBoundArray) > 0 and len(self.usafeupperBoundArray) > 0:
            states = generateRandomStates(samples, self.usafelowerBoundArray, self.usafeupperBoundArray)
        elif self.usafeVerts is not None:
            poly = Polygon(self.usafeVerts)

            states = []
            minx, miny, maxx, maxy = poly.bounds
            while len(states) < samples:
                pnt = [random.uniform(minx, maxx), random.uniform(miny, maxy)]
                if poly.contains(Point(pnt[0], pnt[1])):
                    states.append(pnt)

        return states

    def check_state_containment_in_set(self, r_state, r_set):
        is_contained = True
        lower_bound_array = r_set[0]
        upper_bound_array = r_set[1]

        for dim in range(self.data_object.dimensions):
            l_bound_dim = lower_bound_array[dim]
            u_bound_dim = upper_bound_array[dim]
            if l_bound_dim < r_state[dim] < u_bound_dim:
                continue
            else:
                is_contained = False
                break

        return is_contained

    def check_for_usafe_contain_always(self, traj):
        # print(" Checking for containment " + str(state))
        time_steps = self.time_steps
        # time_steps = [d_time_step]
        found_time_step = -1
        is_contained = True
        usafe_set = [self.usafelowerBoundArray, self.usafeupperBoundArray]
        for time_step in time_steps:
            state = traj[time_step]

            is_contained = self.check_state_containment_in_set(state, usafe_set)

            if is_contained is False:
                break

        if is_contained:
            found_time_step = time_steps[1]
            print(" Trajectory found ")

        return found_time_step

    def check_for_usafe_contain_eventual(self, traj):
        # print(" Checking for containment " + str(state))
        time_steps = self.time_steps
        # time_steps = [d_time_step]
        found_time_step = -1
        usafe_set = [self.usafelowerBoundArray, self.usafeupperBoundArray]
        for time_step in time_steps:
            state = traj[time_step]

            is_contained = self.check_state_containment_in_set(state, usafe_set)

            if is_contained:
                found_time_step = time_step
                # print(" Found time step " + str(found_time_step))
                break

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
        if self.always_spec is True:
            # print("Here")
            found_time_step = self.check_for_usafe_contain_always(traj)

        # robustness = 100

        if found_time_step != -1:
            state = traj[found_time_step]
            robustness = self.compute_robust_state_wrt_axes(state)
            robustness = robustness * -1
        else:
            # state = traj[self.time_steps[int(len(self.time_steps)/2)]]
            r_for_internal_states = []
            r_for_external_states = []
            usafe_set = [self.usafelowerBoundArray, self.usafeupperBoundArray]
            for time_step in self.time_steps:
                state = traj[time_step]
                cur_robustness = self.compute_robust_state_wrt_axes(state)
                if self.check_state_containment_in_set(state, usafe_set):
                    r_for_internal_states.append(cur_robustness)
                else:
                    r_for_external_states.append(cur_robustness)

            # print(len(r_for_internal_states), r_for_internal_states)
            # print(len(r_for_external_states), r_for_external_states)
            if len(r_for_internal_states) > 0 and self.always_spec is False:
                robustness = -1 * np.max(r_for_internal_states)
            elif len(r_for_internal_states) > 0 and self.always_spec is True and len(r_for_internal_states) >= 0.6 * len(r_for_external_states):
                robustness = -1 * np.min(r_for_internal_states)
            else:
                if self.always_spec is False:
                    robustness = np.max(r_for_external_states)
                else:
                    robustness = np.min(r_for_external_states)

        # if found_time_step != -1:
        #     robustness = robustness * -1

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
                # print(dest, traj[t_idx], current_dist)
                if current_dist < min_dist:
                    min_dist = current_dist
                    d_time_step = t_idx
        return d_time_step

    def reachDestInvSenGr(self, dests=None, d_time_steps=None, threshold=0.01, correction_steps=[1],
                          scaling_factors=[0.01], i_state=None, true_inv_sen=None, model_fname=None, sims_f=None):

        ref_traj = self.data_object.generateTrajectories(r_states=i_state)[0]
        dest = dests[0]
        self.data_object.setGradientRun(False)  # To avoid generating neighboring trajectories like in the training

        assert self.data_object is not None

        trained_model = self.getModel(model_fname)
        if trained_model is None:
            return

        if d_time_steps is None:
            print(" Provide a valid time step ")
            return
        elif len(d_time_steps) == 1:
            if isinstance(d_time_steps[0], int):
                d_time_step = d_time_steps[0]
            else:
                d_time_step = int(d_time_steps[0]/self.data_object.getStepSize())
        else:
            self.time_steps.clear()
            if isinstance(d_time_steps[0], int):
                for t_idx in range(d_time_steps[0], d_time_steps[1]):
                    self.time_steps.append(t_idx)
            else:
                for t_idx in range(int(d_time_steps[0]/self.data_object.getStepSize()), int(d_time_steps[1]/self.data_object.getStepSize())):
                    self.time_steps.append(t_idx)
            print(self.time_steps)
            # print(self.time_steps)
            d_time_step = self.predict_falsifying_time_step(dest, ref_traj, False)

        print("*** time step **** " + str(d_time_step))
        paths_list = [[[ref_traj[d_time_step], dest]]]

        for paths in paths_list:
            # print(paths)
            for s_factor in scaling_factors:

                for steps in correction_steps:
                    if steps == -1:
                        print(" *** Axes Aligned *** \n")
                        start_time = time.time()
                        self.reachDestInvAxesAligned(ref_traj=ref_traj, paths=paths, d_time_step=d_time_step,
                                                        threshold=threshold, model_v=trained_model,
                                                        sims_bound=self.sims_bound,
                                                        scaling_factor=s_factor, dynamics=self.dynamics)
                        print("Time taken: " + str(time.time() - start_time))
                    else:
                        start_time = time.time()
                        self.reachDestInvOriginal(ref_traj=ref_traj, paths=paths, d_time_step=d_time_step,
                                                     threshold=threshold,  model_v=trained_model, correction_steps=steps,
                                                     scaling_factor=s_factor, sims_bound=self.sims_bound,
                                                     dynamics=self.dynamics, true_inv_sen=true_inv_sen, sims_f=sims_f)
                        print("Time taken: " + str(time.time() - start_time))

    '''
    Reach Destination implementation with course correction.
    '''
    def reachDestInvOriginal(self, ref_traj, paths, d_time_step, threshold, model_v, correction_steps, sims_bound,
                                scaling_factor, true_inv_sen=None, rand_area=None, dynamics=None, sims_f=None):

        n_paths = len(paths)
        x_val = ref_traj[0]
        dimensions = self.data_object.getDimensions()
        if sims_f is not None:
            sims_f.write(str(ref_traj))
            sims_f.write('\n')
        trajectories = [ref_traj]
        rrt_dests = []
        print("Norm status: " + str(self.norm_status))
        print("True inv sen: " + str(true_inv_sen))

        dest_traj_start_pt = None
        if true_inv_sen is not None:
            dest_traj_start_pt = ref_traj[0] + true_inv_sen
            print(x_val, dest_traj_start_pt)

        for path_idx in range(n_paths):
            vp_vals = []
            v_vals = []
            rrt_path = paths[path_idx]
            xp_val = rrt_path[0]
            dest = rrt_path[1]
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
            print("Starting distance: " + str(dist))
            original_distance = dist
            min_dist = dist
            best_trajectory = ref_traj
            min_simulation = sims_count

            while dist > threshold and sims_count <= sims_bound:

                if original_distance < 0.02:
                    print("Starting distance is very less ")
                    break

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
                vp_val_scaled = [val * scaling_factor for val in vp_val_normalized]
                step = 0
                vp_vals.append(vp_val)
                v_vals.append(v_val)
                # prev_pred_dist = None

                while step < correction_steps:
                    data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val_normalized, t_val)
                    predicted_v = self.evalModel(input=data_point, eval_var='v', model=model_v)

                    if true_inv_sen is not None:
                        true_inv_sen_norm = norm(true_inv_sen, 2)
                        predicted_v_scaled = [val * scaling_factor * true_inv_sen_norm for val in predicted_v]
                    else:
                        predicted_v_scaled = [val * scaling_factor * vp_norm for val in predicted_v]

                    new_init_state = [self.check_for_bounds(x_val + predicted_v_scaled)]

                    x_val = new_init_state[0]

                    if true_inv_sen is not None and dest_traj_start_pt is not None:
                        true_inv_sen = dest_traj_start_pt - x_val

                    xp_val = xp_val + vp_val_scaled
                    v_val = predicted_v_scaled
                    # vp_val = dest - xp_val
                    x_vals.append(x_val)
                    xp_vals.append(xp_val)
                    t_val = d_time_step
                    step += 1

                sims_count += 1

                new_traj = self.data_object.generateTrajectories(r_states=[x_vals[len(x_vals) - 1]])[0]
                if sims_f is not None:
                    sims_f.write(str(new_traj))
                    sims_f.write('\n')
                x_val = new_traj[0]
                xp_val = new_traj[d_time_step]
                vp_val = dest - xp_val
                vp_norm = norm(vp_val, 2)
                dist = vp_norm
                # vp_val_temp = [val * scaling_factor for val in vp_val_normalized]
                vp_vals.append(vp_val)
                v_vals.append(v_val)

                if true_inv_sen is not None and dest_traj_start_pt is not None:
                    true_inv_sen = dest_traj_start_pt - x_val

                trajectories.append(new_traj)

                if dist < min_dist:
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
                elif self.staliro_run and self.check_for_usafe_contain_always(new_traj) == -1:
                    new_dest = self.generateRandomUnsafeStates(1)[0]
                    new_time_step = self.predict_falsifying_time_step(new_dest, new_traj, False)
                    new_dist = norm(new_traj[new_time_step]-new_dest, 2)
                    if new_dist < min_dist:
                        # print("Setting new time step to " + str(new_time_step))
                        d_time_step = new_time_step
                        best_trajectory = new_traj
                        min_dist = new_dist
                        dest = new_dest
                        xp_val = new_traj[d_time_step]
                        vp_val = np.array(dest) - xp_val
                        vp_norm = norm(vp_val, 2)
                        dist = vp_norm

            if dist > original_distance:
                dist = original_distance

            min_rel_dist = min_dist / original_distance

            if self.staliro_run:
                min_dist = self.compute_robust_wrt_axes(best_trajectory)
                print("Best robustness " + str(min_dist))
                print("Final distance " + str(dist))
                # for idx in range(self.time_steps[0], self.time_steps[1]):
                #     print(best_trajectory[idx])

            if self.staliro_run is False:
                print("Final distance " + str(dist))
                print("Final relative distance " + str(dist / original_distance))
                print("Min relative distance: " + str(min_rel_dist))
                print("Min simulation: " + str(min_simulation))
            print("Final simulation: " + str(sims_count))
            self.f_simulations_count = sims_count
            self.f_dist = min_dist
            self.f_rel_dist = min_rel_dist

            self.f_simulations = trajectories
            self.best_trajectory = best_trajectory

            # self.plotInvSenResultsRRTv2(trajectories, dest, d_time_step, dimensions, rand_area, path_idx, dynamics)

            # if self.staliro_run and len(trajectories) > 1:
            #     plotInvSenStaliroResults(trajectories, d_time_step, best_trajectory, self.usafeupperBoundArray,
            #                              self.usafelowerBoundArray, self.data_object)
            #     # plotInvSenResultsAnimate(trajectories, None, d_time_step, v_vals, vp_vals)
            # elif not self.staliro_run:
            #     plotInvSenResults(trajectories, rrt_dests, d_time_step, dimensions, best_trajectory)
            #     # plotInvSenResultsAnimate(trajectories, rrt_dests, d_time_step, v_vals, vp_vals)

    def get_vp_direction_axes_aligned(self, dimensions, scaling_factor, dest, xp_val):
        i_matrix = np.eye(dimensions)
        vp_direction_dist = None
        vp_direction = None
        for i_m_idx in range(dimensions):

            vec_1 = i_matrix[i_m_idx]
            vec_1_scaled = [val * scaling_factor for val in vec_1]
            temp_xp_val = xp_val + vec_1_scaled
            temp_dist = norm(dest - temp_xp_val, 2)
            if vp_direction_dist is None or temp_dist < vp_direction_dist:
                vp_direction_dist = temp_dist
                vp_direction = vec_1_scaled

            vec_2 = i_matrix[i_m_idx]
            vec_2_scaled = [val * -scaling_factor for val in vec_2]
            temp_xp_val = xp_val + vec_2_scaled
            temp_dist = norm(dest - temp_xp_val, 2)
            if temp_dist < vp_direction_dist:
                vp_direction_dist = temp_dist
                vp_direction = vec_2_scaled

        return vp_direction

    '''
    ReachDestination with Axis aligned for correction period 1
    '''
    def reachDestInvAxesAligned(self, ref_traj, paths, d_time_step, threshold, model_v, sims_bound, scaling_factor,
                                   dynamics=None):

        dimensions = self.data_object.getDimensions()
        n_paths = len(paths)
        x_val = ref_traj[0]
        trajectories = [ref_traj]
        rrt_dests = []
        for path_idx in range(n_paths):
            path = paths[path_idx]
            xp_val = path[0]
            dest = path[1]
            rrt_dests.append(dest)
            print("***** path idx " + str(path_idx) + " s_factor " + str(scaling_factor) + " correction steps 1")
            sims_count = 1
            v_val = dest - x_val
            print(xp_val, dest)

            # vp_val = dest - xp_val
            vp_direction = self.get_vp_direction_axes_aligned(dimensions, scaling_factor, dest, xp_val)
            vp_val = vp_direction
            print("vp direction " + str(vp_direction))
            vp_norm = norm(vp_val, 2)
            dist = norm(dest-xp_val, 2)
            min_simulation = sims_count
            t_val = d_time_step
            vp_val = [val / vp_norm for val in vp_val]  # Normalized
            vp_val_scaled = [val * scaling_factor for val in vp_val]
            print("Starting distance: " + str(dist))
            original_distance = dist
            min_dist = dist
            best_trajectory = ref_traj
            vp_vals = [vp_val_scaled]
            v_vals = [v_val]

            while dist > threshold and sims_count < sims_bound:
                data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val, t_val)
                predicted_v = self.evalModel(input=data_point, eval_var='v', model=model_v)
                predicted_v_scaled = [val * scaling_factor for val in predicted_v]
                new_init_state = [self.check_for_bounds(x_val + predicted_v_scaled)]
                new_traj = self.data_object.generateTrajectories(r_states=new_init_state)[0]
                x_val = new_traj[0]
                xp_val = new_traj[d_time_step]
                v_val = predicted_v_scaled
                vp_direction = self.get_vp_direction_axes_aligned(dimensions, scaling_factor, dest, xp_val)
                vp_val = vp_direction
                vp_norm = norm(vp_val, 2)
                dist = norm(dest-xp_val, 2)
                vp_val = [val / vp_norm for val in vp_val]  # Normalized
                vp_val_scaled = [val * scaling_factor for val in vp_val]
                t_val = d_time_step
                vp_vals.append(vp_val_scaled)
                v_vals.append(predicted_v_scaled)
                sims_count = sims_count + 1

                trajectories.append(new_traj)

                if dist < min_dist:
                    min_dist = dist
                    best_trajectory = new_traj
                    min_simulation = sims_count

            print("Final relative distance " + str(dist/original_distance))
            print("Min relative distance " + str(min_dist/original_distance))
            print("Min dist: " + str(min_dist))
            print("Min simulation: " + str(min_simulation))
            print("Final simulation: " + str(sims_count))
            self.f_simulations_count = sims_count
            self.f_dist = min_dist
            self.f_rel_dist = min_dist/original_distance

            plotInvSenResults(trajectories, rrt_dests, d_time_step, dimensions, best_trajectory)
            # plotInvSenResultsAnimate(trajectories, rrt_dests, d_time_step, v_vals, vp_vals)
            # self.plotInvSenResultsRRTv2(trajectories, dest, d_time_step, dimensions, rand_area, path_idx,dynamics)
