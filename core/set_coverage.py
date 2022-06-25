from evaluation_isen import EvaluationInvSen
from sampler import generateRandomStates
from evaluation_plot import plotInvSenInitSetCoverage, plotInvSenReachSetCoverage
from frechet import norm
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np


class CoverageInvSen(EvaluationInvSen):

    def __init__(self, dynamics='None', layers=1, neurons=512, dnn_rbf='RBF', act_fn='ReLU', norm_status=True):
        EvaluationInvSen.__init__(self, dynamics=dynamics, dnn_rbf=dnn_rbf, layers=layers,
                            neurons=neurons, act_fn=act_fn, norm_status=norm_status)

    def partition_init_set(self, partitions=5):
        x_index = 0
        y_index = 1
        i_x_min = self.data_object.lowerBoundArray[x_index]
        i_x_max = self.data_object.upperBoundArray[x_index]
        i_y_min = self.data_object.lowerBoundArray[y_index]
        i_y_max = self.data_object.upperBoundArray[y_index]

        x_dim_partition_size = (i_x_max - i_x_min ) /partitions
        y_dim_partition_size = (i_y_max - i_y_min ) /partitions
        lowerBounds = []
        upperBounds = []

        x_upper = i_x_min

        parity = 0
        for idx in range(partitions):
            x_lower = x_upper
            x_upper = x_lower +x_dim_partition_size

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
    def coverage_init_set(self, d_time_steps, samples=0, partitions=5):

        self.data_object.setGradientRun(False)
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

            dests = []
            for idx in range(len(ref_trajs) - 1):
                dests.append(ref_trajs[idx + 1][d_time_step])

            dests_end_points = np.array(dests)
            hull = ConvexHull(dests_end_points)
            # print(hull.vertices)
            points = []
            for idx in hull.vertices:
                points.append(dests_end_points[idx])
            points.append(dests_end_points[hull.vertices[0]])
            dests_end_points = points

            for idx in range(len(ref_trajs) - 1):
                print(" **** idx **** " + str(idx))
                self.reachDestInvSen(dests=[dests[idx]], d_time_steps=[d_time_step], threshold=0.004,
                                     correction_steps=[10], scaling_factors=[0.05], i_state=[ref_trajs[idx][0]])
                f_abs_dist = f_abs_dist + self.f_dist
                f_sims = f_sims + self.f_simulations_count
                trajectories = self.f_simulations
                best_trajectory = self.best_trajectory
                f_trajs.append(trajectories)
                best_trajs.append(best_trajectory)
            print(len(f_trajs), len(best_trajs))
            print(" Mean final distance " + str(f_abs_dist / len(ref_trajs)))
            print(" Mean simulations count " + str(f_sims / len(ref_trajs)))

            # self.setUnsafeSet(verts=usafe_end_points, staliro_run=False)

            plotInvSenInitSetCoverage(f_trajs, d_time_step, dests_end_points, self.data_object, lowerBounds, upperBounds)

    def compute_boundary_points_from_templates(self, t_val, scaling_factor, correction_steps, template_dirs):

        x_index = 0
        y_index = 1
        i_x_min = self.data_object.lowerBoundArray[x_index]
        i_x_max = self.data_object.upperBoundArray[x_index]
        i_y_min = self.data_object.lowerBoundArray[y_index]
        i_y_max = self.data_object.upperBoundArray[y_index]
        i_centroid = [(i_x_min + i_x_max) / 2, (i_y_min + i_y_max) / 2]
        orig_ref_traj = self.data_object.generateTrajectories(r_states=[i_centroid])[0]
        # print(ref_traj)

        init_set = [self.data_object.lowerBoundArray, self.data_object.upperBoundArray]

        model_v = self.getModel()

        xp_vals = []
        for template_dir in template_dirs:
            # print(template_dir)
            x_val = orig_ref_traj[0]
            xp_val = orig_ref_traj[t_val]
            while self.check_state_containment_in_set(x_val, init_set):
                vp_val = template_dir
                vp_norm = norm(template_dir, 2)
                vp_val_normalized = [val / vp_norm for val in vp_val]  # Normalized
                # print(vp_val_normalized)
                vp_val_scaled = [val * scaling_factor * vp_norm for val in vp_val_normalized]
                # print(vp_val_scaled)
                v_val = vp_val
                step = 0

                while step < correction_steps:
                    data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val_normalized, t_val)

                    predicted_v = self.evalModel(input=data_point, eval_var='v', model=model_v)

                    predicted_v_scaled = [val * scaling_factor * vp_norm for val in predicted_v]

                    # new_init_state = [self.check_for_bounds(x_val + predicted_v_scaled)]

                    x_val = x_val + predicted_v_scaled

                    xp_val = xp_val + vp_val_scaled
                    v_val = predicted_v_scaled
                    step = step + 1

                ref_traj = self.data_object.generateTrajectories(r_states=[x_val])[0]
                # print(x_val)
                xp_val = ref_traj[t_val]
            x_val = self.check_for_bounds(x_val)
            # print("x_val is " + str(x_val))
            # print(" new x_val is " + str(self.check_for_bounds(x_val)))
            xp_val = self.data_object.generateTrajectories(r_states=[x_val])[0][t_val]
            print(xp_val)
            xp_vals.append(xp_val)
        return xp_vals

    def get_enclosing_rect(self, boundary_points):
        boundary_point = boundary_points[0]
        lowerBounds = []
        upperBounds = []
        for dim in range(2):
            lowerBounds.append(boundary_point[dim])
            upperBounds.append(boundary_point[dim])

        for idx in range(1, len(boundary_points)):
            xp_val = boundary_points[idx]
            for dim in range(2):
                if xp_val[dim] < lowerBounds[dim]:
                    lowerBounds[dim] = xp_val[dim]
                elif xp_val[dim] > upperBounds[dim]:
                    upperBounds[dim] = xp_val[dim]
        return [lowerBounds, upperBounds]

    def coverage_reach_set(self, d_time_steps, threshold, samples, rect=False):
        self.data_object.setGradientRun(False)
        t_val = d_time_steps[0]
        scaling_factor = 0.01
        correction_steps = 10

        template_dirs = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
        # template_dirs = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]]

        usafe_end_points = self.compute_boundary_points_from_templates(t_val, scaling_factor, correction_steps, template_dirs)
        # usafe_end_points.append(usafe_end_points[0])

        usafe_rect_bounds = None
        if rect is True:
            usafe_rect_bounds = self.get_enclosing_rect(usafe_end_points)
            self.setUnsafeSet(bounds=usafe_rect_bounds, staliro_run=False)
        else:
            # https://codereview.stackexchange.com/questions/69833/generate-sample-coordinates-inside-a-polygon
            usafe_end_points = np.array(usafe_end_points)
            hull = ConvexHull(usafe_end_points)
            print(hull.vertices)
            points = []
            for idx in hull.vertices:
                points.append(usafe_end_points[idx])
            points.append(usafe_end_points[hull.vertices[0]])
            usafe_end_points = points
            self.setUnsafeSet(verts=usafe_end_points, staliro_run=False)
            # plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)
            # plt.show()

        dests = self.generateRandomUnsafeStates(samples)

        # reachable_count = 0
        f_trajs = []
        good_indices = []
        # unreachable_dests = []
        # reachable_dests = []
        # unreachable_init_states = []
        for idx in range(len(dests)):
            print(" ******* idx ******* " + str(idx))
            dest = dests[idx]
            ref_traj = self.data_object.generateTrajectories(samples=1)[0]
            self.reachDestInvSen(dests=[dest], d_time_steps=[t_val], threshold=threshold,
                                 correction_steps=[correction_steps], scaling_factors=[scaling_factor * 5],
                                 i_state=[ref_traj[0]])
            if self.f_dist < threshold and self.f_simulations_count < 50:
                # reachable_count = reachable_count + 1
                good_indices.append(idx)
                # reachable_dests.append(dest)
            # else:
            #     # unreachable_dests.append(dest)
            f_trajs.append(self.best_trajectory)

        print(" Number of reachable states " + str(len(good_indices)))

        if rect is True:
            plotInvSenReachSetCoverage(data_object=self.data_object, dests=dests, good_dest_indices=good_indices,
                                       f_trajs=f_trajs, usafe_rect_bounds=usafe_rect_bounds)
        else:
            plotInvSenReachSetCoverage(data_object=self.data_object, dests=dests, good_dest_indices=good_indices,
                                       f_trajs=f_trajs, usafe_verts=usafe_end_points)


