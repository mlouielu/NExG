
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib import animation
import sys
from frechet import norm
from itertools import product, combinations
import os


def get_verts_from_bounds(rect_bounds):
    lowerBounds = rect_bounds[0]
    upperBounds = rect_bounds[1]
    # if len(lowerBounds) > 2:
    #     return "Not supported dimensions > 2"
    x_index = 0
    y_index = 1

    x_min = lowerBounds[x_index]
    x_max = upperBounds[x_index]
    y_min = lowerBounds[y_index]
    y_max = upperBounds[y_index]

    verts = [
        (x_min, y_min),  # left, bottom
        (x_max, y_min),  # left, top
        (x_max, y_max),  # right, top
        (x_min, y_max),  # right, bottom
        (x_min, y_min),  # ignored
    ]

    return verts


def plotFwdSenTrajectoriesNew(data_object, ref_trajs, predicted_trajs, actual_trajs, max_time, x_vals=None, xp_vals=None):
        x_index = 0
        y_index = 1

        i_verts = get_verts_from_bounds([data_object.lowerBoundArray, data_object.upperBoundArray])

        fig, ax = plt.subplots()

        init_p = Polygon(i_verts, edgecolor='blue', fill=False, lw=2, label='Initial set')
        ax.add_patch(init_p)

        n_trajs = len(predicted_trajs)

        # ax.plot(ref_trajs[0][0:max_time, x_index], ref_trajs[0][0:max_time, y_index], color='k',
        #         label='Anchor trajectory')
        ax.plot(predicted_trajs[n_trajs-1][0:max_time, x_index], predicted_trajs[n_trajs-1][0:max_time, y_index],
                color='r', label='Predicted trajectory')
        ax.plot(actual_trajs[n_trajs-1][0:max_time, x_index], actual_trajs[n_trajs-1][0:max_time, y_index], color='g',
                label='Actual trajectory')
        for idx in range(0, n_trajs):
            # ref_traj = ref_trajs[idx]
            # ax.plot(ref_traj[0:max_time, x_index], ref_traj[0:max_time, y_index], color='b')
            predicted_traj = predicted_trajs[idx]
            actual_traj = actual_trajs[idx]
            ax.plot(predicted_traj[0:max_time, x_index], predicted_traj[0:max_time, y_index], color='r')
            ax.plot(actual_traj[0:max_time, x_index], actual_traj[0:max_time, y_index], color='g')

        # if x_vals is not None and len(x_vals) > 0:
        #     ax.scatter(x_vals[0][0], x_vals[0][1], color='g', label='Initial states')
            # ax.scatter(xp_vals[0][0], xp_vals[0][1], color='r', label='Pred destination states')
            # for idx in range(1, len(x_vals)):
            #       ax.scatter(x_vals[idx][0], x_vals[idx][1], color='g')
            #       ax.scatter(xp_vals[idx][0], xp_vals[idx][1], color='r')

        ax.set_xlabel('x' + str(x_index))
        ax.set_ylabel('x' + str(y_index))
        plt.legend()
        plt.show()


def plotInvSenInitSetCoverage(trajectories, d_time_step, dests_end_points, data_object, lowerBounds, upperBounds):

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 4)]

    x_index = 0
    y_index = 1

    i_verts = get_verts_from_bounds([data_object.lowerBoundArray, data_object.upperBoundArray])

    fig, ax = plt.subplots()

    init_p = Polygon(i_verts, edgecolor='blue', fill=False, lw=2, label='Initial set')
    ax.add_patch(init_p)

    dests_end_points = np.array(dests_end_points)
    dest_p = Polygon(dests_end_points, edgecolor='k', fill=False, lw=2, label='Destination set')
    ax.add_patch(dest_p)

    if lowerBounds is not None:
        for idx in range(len(lowerBounds)):
            lowerBound = lowerBounds[idx]
            upperBound = upperBounds[idx]
            i_verts = get_verts_from_bounds([lowerBound, upperBound])
            init_p = Polygon(i_verts, edgecolor='blue', fill=False)
            ax.add_patch(init_p)

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(trajectories))]
    # colors = ['red', 'black', 'blue', 'brown', 'green']

    n_trajectories = len(trajectories)
    # ax.plot(trajectories[0][0][:, x_index], trajectories[0][0][:, y_index], color='magenta', label='Starting trajectory')

    for idx in range(n_trajectories):
        trajs = trajectories[idx]
        for idy in range(len(trajs)):
            trajectory = trajs[idy]
            pred_init = trajectory[0]
            pred_destination = trajectory[d_time_step]
            ax.scatter(pred_init[x_index], pred_init[y_index], color=colors[idx])
            ax.scatter(pred_destination[x_index], pred_destination[y_index], color=colors[idx])
        # ax.plot(best_trajectories[idx][:, x_index], best_trajectories[idx][:, y_index], color='g', label='Best trajectory')
                # plt.plot(trajectory[:, x_index], trajectory[:, y_index], 'g')

        # plt.plot(trajectories[1][:, x_index], trajectories[1][:, y_index], 'g', label='Intermediate trajectory')
        # plt.plot(best_trajectory[:, x_index], best_trajectory[:, y_index], 'b', label='Final trajectory')

        # plt.plot(destination[x_index], destination[y_index], 'ko', label='Destination z')

        # for destination in destinations:
        #     plt.plot(destination[x_index], destination[y_index], 'ko')
    plt.legend(fontsize=12)
    plt.show()


def plotInvSenReachSetCoverage(data_object, dests, good_dest_indices, f_trajs, usafe_verts=None, usafe_rect_bounds=None):

    assert len(dests) == len(f_trajs)

    if len(good_dest_indices) == 0:
        return

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 4)]

    x_index = 0
    y_index = 1
    i_verts = get_verts_from_bounds([data_object.lowerBoundArray, data_object.upperBoundArray])

    fig, ax = plt.subplots()
    #
    init_p = Polygon(i_verts, edgecolor='blue', fill=False, lw=2, label='Initial set')
    ax.add_patch(init_p)

    if usafe_verts is not None:
        u_verts = np.array(usafe_verts)
    else:
        u_verts = get_verts_from_bounds(usafe_rect_bounds)
    # print(u_verts)
    p = Polygon(u_verts, edgecolor='k', fill=False, lw=2, label='Reachable set')
    ax.add_patch(p)

    if data_object.getDynamics() == "OtherBenchC1":
        ax.set_xlim([0.2, 1.55])
        ax.set_ylim([-1.5, 1.55])
    elif data_object.getDynamics() == "OtherBenchC2":
        ax.set_xlim([0, 2.0])
        ax.set_ylim([-0.8, 1.25])
    else:
        ax.set_xlim([0.25, 1.25])
        ax.set_ylim([-0.25, 1.25])

    list_idx = 0
    reachable_dests = []
    unreachable_dests = []
    good_init_states = []
    other_init_states = []
    # print(good_dest_indices)
    for idx in range(len(dests)):
        if list_idx < len(good_dest_indices) and (idx == good_dest_indices[list_idx]):
            reachable_dests.append(dests[idx])
            good_init_states.append(f_trajs[idx][0])
            list_idx = list_idx + 1
        else:
            unreachable_dests.append(dests[idx])
            other_init_states.append(f_trajs[idx][0])

    for dest in reachable_dests:
        ax.scatter(dest[x_index], dest[y_index], color='green')

    for i_state in good_init_states:
        ax.scatter(i_state[x_index], i_state[y_index], color='green')
        # ax.plot(traj[:, x_index], traj[:, y_index], color='magenta')

    for dest in unreachable_dests:
        ax.scatter(dest[x_index], dest[y_index], color='red')

    for i_state in other_init_states:
        ax.scatter(i_state[x_index], i_state[y_index], color='red')

    if data_object.getDynamics() == "OtherBenchC1":
        plt.text(0.35, -0.2, "Total states " + str(len(dests)), fontsize=12)
        plt.text(0.35, -0.4, "Reachable states " + str(len(good_dest_indices)), fontsize=12)
    elif data_object.getDynamics() == "OtherBenchC2":
        plt.text(0.05, 0.2, "Total states " + str(len(dests)), fontsize=12)
        plt.text(0.05, 0, "Reachable states " + str(len(good_dest_indices)), fontsize=12)
    elif data_object.getDynamics() == "OtherBenchC3":
        plt.text(0.3, 0.4, "Total states " + str(len(dests)), fontsize=12)
        plt.text(0.3, 0.25, "Reachable states " + str(len(good_dest_indices)), fontsize=12)

    plt.legend(fontsize=12)
    plt.show()


def plotInvSenResults(trajectories, destinations, d_time_step, dimensions, best_trajectory, x_vals=None, xp_vals=None):
    n_trajectories = len(trajectories)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]

    destination = destinations[0]

    if dimensions == 3:
        x_index = 0
        y_index = 1
        z_index = 2
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlabel('x')
        ax.set_xlabel('y')
        ax.set_xlabel('z')
        ax.plot3D(trajectories[0][:, x_index], trajectories[0][:, y_index], trajectories[0][:, z_index],
                  color=colors[7], label='Reference trajectory')
        ax.scatter3D(trajectories[0][0, x_index], trajectories[0][0, y_index], trajectories[0][0, z_index],
                     color='green', label='states at time 0')
        ax.scatter3D(trajectories[0][d_time_step, x_index], trajectories[0][d_time_step, y_index],
                     trajectories[0][d_time_step, z_index], color='red', label='states at time t')

        if x_vals is not None:
            for idx in range(len(x_vals)):
                x_val = x_vals[idx]
                xp_val = xp_vals[idx]
                ax.scatter3D(x_val[x_index], x_val[y_index], x_val[z_index], color='green')
                ax.scatter3D(xp_val[x_index], xp_val[y_index], xp_val[z_index], color='red')
        else:
            for idx in range(1, n_trajectories - 1):
                trajectory = trajectories[idx]
                pred_init = trajectory[0]
                pred_destination = trajectory[d_time_step]
                ax.scatter3D(pred_init[x_index], pred_init[y_index], pred_init[z_index], color='green')
                ax.scatter3D(pred_destination[x_index], pred_destination[y_index], pred_destination[z_index],
                             color='red')

        ax.plot3D(best_trajectory[:, x_index], best_trajectory[:, y_index],
                  best_trajectory[:, z_index], color='blue', label='Final trajectory')

        ax.scatter3D(destination[x_index], destination[y_index], destination[z_index], color='black',
                     label='Destination z')

        # For plotting cube initial set
        # r = [0.2, 0.5]
        # for s, e in combinations(np.array(list(product(r, r, r))), 2):
        #     if np.sum(np.abs(s - e)) == r[1] - r[0]:
        #         ax.plot3D(*zip(s, e), color="black")
        # ax.set_title("Cube")

        for destination in destinations:
            ax.scatter3D(destination[x_index], destination[y_index], destination[z_index], color='black')
        plt.legend()
        plt.show()

    else:
        plt.figure(1)
        x_indices = [0]
        y_indices = [1]

        # To plot an obstacle
        # c_center_x = 1.75
        # c_center_y = -0.25
        # c_size = 0.1
        # deg = list(range(0, 360, 5))
        # deg.append(0)
        # xl = [c_center_x + c_size * math.cos(np.deg2rad(d)) for d in deg]
        # yl = [c_center_y + c_size * math.sin(np.deg2rad(d)) for d in deg]
        # plt.plot(xl, yl, color='k')

        if dimensions == 4 or dimensions == 5:
            x_indices = [0, 1]
            y_indices = [3, 2]
        elif dimensions == 6:
            # ACC 3L: 0, 5, 1, 4, 2, 3
            # ACC 5L: 0, 5, 3, 2, 1, 4
            # ACC 7L, 10L: 0, 5, 1, 4, 2, 3
            x_indices = [0, 3, 1]
            y_indices = [5, 2, 4]
        elif dimensions == 12:
            x_indices = [0, 2, 4, 6, 8, 10]
            y_indices = [1, 3, 5, 7, 9, 11]

        for x_idx in range(len(x_indices)):
            x_index = x_indices[x_idx]
            y_index = y_indices[x_idx]
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            plt.plot(trajectories[0][:, x_index], trajectories[0][:, y_index], color='magenta',
                     label='Reference Trajectory')
            # starting_state = trajectories[0][d_time_step]
            plt.plot(trajectories[0][0, x_index], trajectories[0][0, y_index], 'g^', label='states at time 0')
            plt.plot(trajectories[0][d_time_step, x_index], trajectories[0][d_time_step, y_index], 'r^',
                     label='states at time t')

            if x_vals is not None:
                for idx in range(len(x_vals)):
                    x_val = x_vals[idx]
                    xp_val = xp_vals[idx]
                    plt.plot(x_val[x_index], x_val[y_index], 'g^')
                    plt.plot(xp_val[x_index], xp_val[y_index], 'r^')
                for idx in range(1, n_trajectories - 1):
                    trajectory = trajectories[idx]
                    plt.plot(trajectory[:, x_index], trajectory[:, y_index], 'g')

            else:
                for idx in range(1, n_trajectories - 1):
                    trajectory = trajectories[idx]
                    pred_init = trajectory[0]
                    pred_destination = trajectory[d_time_step]
                    plt.plot(pred_init[x_index], pred_init[y_index], 'g^')
                    plt.plot(pred_destination[x_index], pred_destination[y_index], 'r^')
                    plt.plot(trajectory[:, x_index], trajectory[:, y_index], 'g')

            plt.plot(trajectories[1][:, x_index], trajectories[1][:, y_index], 'g', label='Intermediate trajectory')
            plt.plot(best_trajectory[:, x_index], best_trajectory[:, y_index], 'b', label='Final trajectory')

            plt.plot(destination[x_index], destination[y_index], 'ko', label='Destination z')

            for destination in destinations:
                plt.plot(destination[x_index], destination[y_index], 'ko')
            plt.legend()
            plt.show()


def plotInvSenStaliroResults(trajectories, d_time_step, best_trajectory, usafeupperBoundArray, usafelowerBoundArray,
                             data_object):

    print("Here")
    x_index = 0
    y_index = 1

    u_verts = get_verts_from_bounds([usafelowerBoundArray, usafeupperBoundArray])

    i_verts = get_verts_from_bounds([data_object.lowerBoundArray, data_object.lowerBoundArray])

    codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
    ]

    u_path = Path(u_verts, codes)

    fig, ax = plt.subplots()

    u_patch = patches.PathPatch(u_path, facecolor='red', lw=1)
    ax.add_patch(u_patch)

    i_path = Path(i_verts, codes)

    i_patch = patches.PathPatch(i_path, facecolor='red', lw=1, fill=False)
    ax.add_patch(i_patch)

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    # colors = ['red', 'black', 'blue', 'brown', 'green']

    n_trajectories = len(trajectories)
    ax.plot(trajectories[0][:, x_index], trajectories[0][:, y_index], color='magenta', label='Reference trajectory')

    for idx in range(1, n_trajectories - 1):
        pred_init = trajectories[idx][0]
        ax.scatter(pred_init[x_index], pred_init[y_index], color='g')
        pred_dest = trajectories[idx][d_time_step]
        # ax.scatter(pred_dest[x_index], pred_dest[y_index], color='r')
        ax.plot(trajectories[idx][:, x_index], trajectories[idx][:, y_index], color='g')

    ax.plot(trajectories[1][:, x_index], trajectories[1][:, y_index], color='g', label='Intermediate trajectory')
    ax.plot(best_trajectory[:, x_index], best_trajectory[:, y_index], color=colors[1], label='Best trajectory')

    plt.rcParams.update({'font.size': 12})
    ax.set_xlabel('x' + str(x_index), fontsize=15)
    ax.set_ylabel('x' + str(y_index), fontsize=15)
    # ax.set_xlim(5, 10)
    # ax.set_ylim(5, 30)
    plt.legend()

    plt.show()


# https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib
# https://matplotlib.org/stable/api/animation_api.html
# https://matplotlib.org/stable/gallery/animation/dynamic_image.html
# http://blog.vikramank.com/2015/02/methods-animations-loop-animation-package-python/
# https://stackoverflow.com/questions/23049762/matplotlib-multiple-animate-multiple-lines
# https://stackoverflow.com/questions/38980794/python-matplotlib-funcanimation-save-only-saves-100-frames
# https://stackoverflow.com/questions/58263646/attributeerror-list-object-has-no-attribute-get-zorder
# http://louistiao.me/posts/notebooks/save-matplotlib-animations-as-gifs/
def plotInvSenResultsAnimate(trajectories, destinations, d_time_step, v_vals=None, vp_vals=None):

    staliro_run = False
    if destinations is None:
        staliro_run = True

    n_trajectories = len(trajectories)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]

    fig = plt.figure()
    # ax1 = plt.axes(xlim=(0.4, 1.8), ylim=(-0.7, 1.21))  # For bench2  - For correction period
    # ax1 = plt.axes(xlim=(0.55, 1.2), ylim=(-0.7, 1.0))  # For bench2
    # ax1 = plt.axes(xlim=(-1.0, 0.1), ylim=(-0.6, 0.0))  # For bench9-III (Tanh)
    ax1 = plt.axes(xlim=(-0.9, 0.1), ylim=(-0.38, -0.32))  # For bench9-III (Tanh)
    # ax1 = plt.axes(xlim=(-1.4, -0.2), ylim=(-0.5, 0.8))  # For bench9

    plt.xlabel('X')
    plt.ylabel('Y')

    u_set_lines = []
    i_set_lines = []
    lines = []
    dots = []
    vectors = []
    x_xp_dots = []
    if staliro_run is False:
        destination = destinations[0]

    lobj = ax1.plot([], [], lw=1, color=colors[4], label='Initial set')[0]
    i_set_lines.append(lobj)
    for index in range(3):
        lobj = ax1.plot([], [], lw=1, color=colors[4])[0]
        i_set_lines.append(lobj)

    if staliro_run is True:
        lobj = ax1.plot([], [], lw=1, color=colors[1], label='Unsafe set')[0]
        u_set_lines.append(lobj)
        for index in range(3):
            lobj = ax1.plot([], [], lw=1, color=colors[4])[0]
            u_set_lines.append(lobj)

    for index in range(n_trajectories):
        lobj = ax1.plot([], [], lw=1, color=colors[7])[0]
        lines.append(lobj)

    vobj = ax1.plot([], [], lw=3, color='red')[0]
    vectors.append(vobj)
    vobj = ax1.plot([], [], lw=3, color='red')[0]
    vectors.append(vobj)
    # ax1.cla()
    # vobj = ax1.arrow(0.0, 0.0, 0.0, 0.0,  head_width=2, head_length=2, fc='black', ec='black')
    # vectors.append(vobj)
    # vobj = ax1.arrow(0.0, 0.0, 0.0, 0.0,  head_width=2, head_length=2, fc='black', ec='black')
    # vectors.append(vobj)
    if staliro_run is False:
        dotObj = ax1.plot([], [], 'ko', label='destination')[0]  # destination
        dots.append(dotObj)
    dotObj = ax1.plot([], [], color='black', marker="*", markersize=6)[0]  # next state at 0
    dots.append(dotObj)
    dotObj = ax1.plot([], [], color='black', marker="*", markersize=6)[0]  # next state at d_time_step
    dots.append(dotObj)

    dObj = ax1.plot([], [], 'b.', label='initial state')[0]  # state at 0
    x_xp_dots.append(dObj)
    dObj = ax1.plot([], [], 'g.', label='state at time '+str('t'))[0]  # state at d_time_step
    x_xp_dots.append(dObj)

    for index in range(n_trajectories-1):
        dObj = ax1.plot([], [], 'b.')[0]  # state at 0
        x_xp_dots.append(dObj)
        dObj = ax1.plot([], [], 'g.')[0]  # state at d_time_step
        x_xp_dots.append(dObj)

    def init():
        for line in lines:
            line.set_data([], [])
        for dot in dots:
            dot.set_data([], [])
        for vec in vectors:
            vec.set_data([], [])
        for dot in x_xp_dots:
            dot.set_data([], [])
        for i_line in i_set_lines:
            i_line.set_data([], [])
        for u_line in u_set_lines:
            u_line.set_data([], [])
        return lines, dots, vectors, x_xp_dots, i_set_lines, u_set_lines

    frame_num = n_trajectories
    u_set_xlist = []
    u_set_ylist = []

    # test 1 - Bench2
    # x_index = 0
    # y_index = 1
    # i_set_xlist = [0.5, 1.2, 1.2, 0.5, 0.5]
    # i_set_ylist = [0.5, 0.5, 1.1, 1.1, 0.5]

    # test - Bench 2 - For Axis aligned
    # x_index = 0
    # y_index = 1
    # i_set_xlist = [0.5, 1.2, 1.2, 0.5, 0.5]
    # i_set_ylist = [0.5, 0.5, 1.2, 1.2, 0.5]
    #
    # u_set_xlist = [1.5, 1.55, 1.55, 1.5, 1.5]
    # u_set_ylist = [0.2, 0.2, 0.25, 0.25, 0.2]

    # test 2 - Bench9 Tanh
    x_index = 0
    y_index = 3
    i_set_xlist = [-1.0, -0.5, -0.5, -1.0, -1.0]
    i_set_ylist = [-0.5, -0.5, 0.0, 0.0, -0.5]

    # test 3 - Bench 9
    # x_index = 1
    # y_index = 2
    # i_set_xlist = [-1.0, -1.0, -0.5, -0.5, -1.0]
    # i_set_ylist = [-0.4, 0.0, 0.0, -0.4, -0.4]

    def animate(idx):
        dots_idx = 0

        graph_list = []
        # sc_1.set_offsets([destination[x_index], destination[y_index]])
        trajectory = trajectories[idx]
        for idz in range(4):
            i_set_lines[idz].set_data(i_set_xlist, i_set_ylist)

        if staliro_run is True:
            for idz in range(4):
                u_set_lines[idz].set_data(u_set_xlist, u_set_ylist)

        # dots[1].set_data(trajectory[0, x_index], trajectory[0, y_index])
        # dots[2].set_data(trajectory[d_time_step, x_index], trajectory[d_time_step, y_index])

        if staliro_run is False:
            dots[dots_idx].set_data(destination[x_index], destination[y_index])
            dots_idx += 1
        xlist = [trajectory[:, x_index]]
        ylist = [trajectory[:, y_index]]
        lines[idx].set_data(xlist, ylist)
        x_xp_dots[2*idx].set_data(trajectory[0, x_index], trajectory[0, y_index])
        x_xp_dots[2*idx+1].set_data(trajectory[d_time_step, x_index], trajectory[d_time_step, y_index])
        # for lnum, line in enumerate(lines):
        #     line.set_data(xlist, ylist)  # set data for each line separately.

        # bench 9 - tanh
        # scale_v = 2.0
        # scale_vp = 0.8
        # bench 9
        # scale_v = 5.0
        # scale_vp = 2.0
        # bench 2
        scale_v = 1.0
        scale_vp = 1.0
        if idx < frame_num-1:
            xlist = [trajectory[0, x_index]]
            xlist.append(trajectory[0, x_index] + scale_v * v_vals[idx+1][x_index])
            # xlist.append(0.1 * x_vals[idx+1][x_index])
            ylist = [trajectory[0, y_index]]
            ylist.append(trajectory[0, y_index] + scale_v * v_vals[idx+1][y_index])
            # xlist.append(0.1 * x_vals[idx+1][y_index])
            vectors[0].set_data(xlist, ylist)

            xlist = [trajectory[d_time_step, x_index]]
            xlist.append(trajectory[d_time_step, x_index] + scale_vp * vp_vals[idx][x_index])
            # xlist.append(xp_vals[idx + 1][x_index])
            ylist = [trajectory[d_time_step, y_index]]
            ylist.append(trajectory[d_time_step, y_index] + scale_vp * vp_vals[idx][y_index])
            # ylist.append(xp_vals[idx + 1][y_index])
            vectors[1].set_data(xlist, ylist)

            # dots[1].set_data(x_vals[idx+1][x_index], x_vals[idx+1][y_index])
            # v_val_norm = norm(v_vals[idx+1], 2)
            dots[dots_idx].set_data(trajectory[0, x_index] + scale_v * v_vals[idx+1][x_index], trajectory[0, y_index] + scale_v * v_vals[idx+1][y_index])
            dots_idx += 1
            dots[dots_idx].set_data(trajectory[d_time_step, x_index] + scale_vp * vp_vals[idx][x_index], trajectory[d_time_step, y_index] + scale_vp * vp_vals[idx][y_index])
            # dots[2].set_data(xp_vals[idx + 1][x_index], xp_vals[idx + 1][y_index])
        # else:
        #     xlist = []
        #     ylist = []
        #     vectors[0].set_data(xlist, ylist)
        #     vectors[1].set_data(xlist, ylist)
        #     dots[1].set_data([], [])
        #     dots[2].set_data([], [])

        graph_list.append(lines)
        graph_list.append(dots)
        graph_list.append(x_xp_dots)
        graph_list.append(vectors)
        graph_list.append(i_set_lines)
        graph_list = [item for sublist in graph_list for item in sublist]
        legend = plt.legend()
        return graph_list, legend

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frame_num, interval=1000, save_count=sys.maxsize, repeat=False)

    # def update(idx, trajectories, line):
    #     line.set_data(trajectories[:, x_index], y1[:num])
    #     return line,
    #
    # ani = animation.FuncAnimation(fig, update, frame_num, fargs=[trajectories, line, ], interval=200, blit=True)

    # anim.save('test.mp4', fps=5.0, dpi=200)
    anim.save('bench2.gif', writer='imagemagick', fps=1)
    fig.savefig('last_frame.png')
    plt.legend()
    plt.show()

