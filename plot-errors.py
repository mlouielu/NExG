# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt

# https://matplotlib.org/stable/gallery/color/named_colors.html
import matplotlib.colors as mcolors


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# when we used to divide with vp-norm
def plotDelta_old(benchmark):

    if benchmark == 'bench1':
        v_vals = [0.18, 0.11, 0.037, 0.033, 0.029, 0.022, 0.015, 0.007, 0.0037, 0.0033, 0.0026, 0.0018, 0.0011, 0.00036]
        delta_old_nex = [0.13, 0.11, 0.098, 0.098, 0.098, 0.098, 0.097, 0.097, 0.097, 0.097, 0.097, 0.097, 0.097, 0.097]
        delta_new_scale_05 = [0.05, 0.043, 0.023, 0.021, 0.019, 0.015, 0.010, 0.005, 0.0027, 0.0024, 0.0019, 0.0014, 0.0008, 0.00027]  # v = 0.18
        delta_new_scale_03 = [0.074, 0.025, 0.013, 0.012, 0.011, 0.009, 0.006, 0.003, 0.0017, 0.0015, 0.0012, 0.00087, 0.00052, 0.00018]  # v = 0.11
        delta_new_scale_01 = [0.10, 0.037, 0.008, 0.0076, 0.0068, 0.0055, 0.0039, 0.0021, 0.0011, 0.001, 0.00078, 0.00056, 0.00034, 0.00011]  # v = 0.037
        # delta_new_scale_009 = []  # v = 0.033
        # delta_new_scale_008 = []  # v = 0.029
        # delta_new_scale_006 = []  # v = 0.022
        delta_new_scale_005 = [0.12, 0.052, 0.01, 0.009, 0.0078, 0.0056, 0.0036, 0.0018, 0.0009, 0.0008, 0.0006, 0.00045, 0.00027, 0.00009]  # v = 0.018
        # delta_new_scale_004 = []  # v = 0.015
        # delta_new_scale_002 = []  # v = 0.007
        delta_new_scale_001 = [0.12, 0.051, 0.0075, 0.0064, 0.0054, 0.0037, 0.0023, 0.0011, 0.0006, 0.0005, 0.0004, 0.0003, 0.00018, 0.00006]  # v = 0.0037
        # delta_new_scale_0009 = []  # v = 0.0033
        # delta_new_scale_0007 = []  # v = 0.0026
        delta_new_scale_0005 = [0.12, 0.053, 0.0081, 0.0069, 0.0058, 0.004, 0.0024, 0.0012, 0.00057, 0.0005, 0.0004, 0.00028, 0.00017, 0.00006]  # v = 0.0018
        # delta_new_scale_0003 = []  # v = 0.0011
        delta_new_scale_0001 = [0.13, 0.056, 0.011, 0.0092, 0.0078, 0.0055, 0.0034, 0.0016, 0.00076, 0.00068, 0.00053, 0.00037, 0.00022, 0.00007]  # v = 0.00036
        plt.plot(v_vals, delta_old_nex, marker = 'o', label="old-neuralExp")
        # plt.plot(v_vals, delta_new_scale_05, marker='o', label="new-v-0.18")
        # plt.plot(v_vals, delta_new_scale_03, marker='o', label="new-v-0.11")
        # plt.plot(v_vals, delta_new_scale_01, marker='o', label="new-v-0.037")
        # plt.plot(v_vals, delta_new_scale_005, marker='o', label="new-v-0.018")
        # plt.plot(v_vals, delta_new_scale_001, marker='o', label="new-v-0.0037")
        # plt.plot(v_vals, delta_new_scale_0005, marker='o', label="new-v-0.0018")
        # plt.plot(v_vals, delta_new_scale_0001, marker='o', label="new-v-0.00036")
    else:
        v_vals = [0.52, 0.26, 0.08, 0.071, 0.063, 0.047, 0.031, 0.016, 0.008, 0.007, 0.0055, 0.004, 0.0023, 0.00078]
        delta_old_nex = [0.21, 0.098, 0.057, 0.056, 0.055, 0.054, 0.053, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052]
        delta_new_scale_05 = [0.085, 0.1, 0.058, 0.053, 0.048, 0.037, 0.026, 0.013, 0.0067, 0.006, 0.0047, 0.0034, 0.002, 0.00068]  # v = 0.52
        delta_new_scale_03 = [0.2, 0.056, 0.04, 0.036, 0.033, 0.026, 0.019, 0.0098, 0.005, 0.0045, 0.0035, 0.0025, 0.0015, 0.0005]  # v = 0.26
        delta_new_scale_01 = [0.34, 0.13, 0.017, 0.016, 0.015, 0.013, 0.01, 0.006, 0.0032, 0.0028, 0.0022, 0.0016, 0.001, 0.0003]  # v = 0.08
        # delta_new_scale_009 = []  # v = 0.071
        # delta_new_scale_008 = []  # v = 0.063
        # delta_new_scale_006 = []  # v = 0.047
        delta_new_scale_005 = [0.37, 0.16, 0.019, 0.016, 0.014, 0.01, 0.007, 0.0042, 0.0023, 0.0021, 0.0016, 0.0012, 0.0007, 0.00025]  # v = 0.04
        # delta_new_scale_004 = []  # v = 0.031
        # delta_new_scale_002 = []  # v = 0.016
        delta_new_scale_001 = [0.37, 0.17, 0.028, 0.023, 0.019, 0.013, 0.0076, 0.0035, 0.0018, 0.0016, 0.0012, 0.0009, 0.0005, 0.00018]  # v = 0.008
        # delta_new_scale_0009 = []  # v = 0.007
        # delta_new_scale_0007 = []  # v = 0.0055
        delta_new_scale_0005 = [0.36, 0.165, 0.027, 0.023, 0.019, 0.012, 0.0072, 0.0033, 0.0017, 0.0015, 0.0012, 0.0008, 0.0005, 0.00017]  # v = 0.004
        # delta_new_scale_0003 = []  # v = 0.0023
        delta_new_scale_0001 = [0.38, 0.18, 0.032, 0.027, 0.022, 0.015, 0.0084, 0.0037, 0.0018, 0.0016, 0.0012, 0.0009, 0.0005, 0.00017]  # v = 0.00078
        plt.plot(v_vals, delta_old_nex, marker='o', label="old-neuralExp")
        # plt.plot(v_vals, delta_new_scale_05, marker='o', label="new-v-0.52")
        # plt.plot(v_vals, delta_new_scale_03, marker='o', label="new-v-0.26")
        # plt.plot(v_vals, delta_new_scale_01, marker='o', label="new-v-0.08")
        # plt.plot(v_vals, delta_new_scale_005, marker='o', label="new-v-0.04")
        # plt.plot(v_vals, delta_new_scale_001, marker='o', label="new-v-0.008")
        # plt.plot(v_vals, delta_new_scale_0005, marker='o', label="new-v-0.004")
        # plt.plot(v_vals, delta_new_scale_0001, marker='o', label="new-v-0.00078")

    # plt.ylim(0, 10)
    # plt.fill_between(x, y1, y4, color='yellow')
    plt.legend()
    plt.show()


def plotMSE_old(benchmark):

    if benchmark == 'bench1':
        v_vals = [0.18, 0.11, 0.037, 0.033, 0.029, 0.022, 0.015, 0.007, 0.0037, 0.0033, 0.0026, 0.0018, 0.0011, 0.00036]
        delta_old_nex = [0.13, 0.11, 0.098, 0.098, 0.098, 0.098, 0.097, 0.097, 0.097, 0.097, 0.097, 0.097, 0.097, 0.097]
        delta_new_scale_05 = [0.1, 0.14, 0.23, 0.23, 0.24, 0.25, 0.25, 0.26, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27]  # v = 0.18
        delta_new_scale_03 = [0.15, 0.08, 0.13, 0.14, 0.14, 0.15, 0.16, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.18]  # v = 0.11
        delta_new_scale_01 = [0.2, 0.12, 0.08, 0.08, 0.09, 0.1, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11]  # v = 0.037
        # delta_new_scale_009 = []  # v = 0.033
        # delta_new_scale_008 = []  # v = 0.029
        # delta_new_scale_006 = []  # v = 0.022
        delta_new_scale_005 = [0.24, 0.17, 0.1, 0.1, 0.1, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09]  # v = 0.018
        # delta_new_scale_004 = []  # v = 0.015
        # delta_new_scale_002 = []  # v = 0.007
        delta_new_scale_001 = [0.23, 0.16, 0.08, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]  # v = 0.0037
        # delta_new_scale_0009 = []  # v = 0.0033
        # delta_new_scale_0007 = []  # v = 0.0026
        delta_new_scale_0005 = [0.24, 0.16, 0.08, 0.08, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]  # v = 0.0018
        # delta_new_scale_0003 = []  # v = 0.0011
        delta_new_scale_0001 = [0.25, 0.19, 0.11, 0.1, 0.1, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07]  # v = 0.00036
        plt.plot(v_vals, delta_old_nex, marker = 'o', label="old-neuralExp")
        plt.plot(v_vals, delta_new_scale_05, marker='o', label="new-v-0.18")
        plt.plot(v_vals, delta_new_scale_03, marker='o', label="new-v-0.11")
        plt.plot(v_vals, delta_new_scale_01, marker='o', label="new-v-0.037")
        plt.plot(v_vals, delta_new_scale_005, marker='o', label="new-v-0.018")
        plt.plot(v_vals, delta_new_scale_001, marker='o', label="new-v-0.0037")
        plt.plot(v_vals, delta_new_scale_0005, marker='o', label="new-v-0.0018")
        plt.plot(v_vals, delta_new_scale_0001, marker='o', label="new-v-0.00036")
    else:
        v_vals = [0.52, 0.26, 0.08, 0.071, 0.063, 0.047, 0.031, 0.016, 0.008, 0.007, 0.0055, 0.004, 0.0023, 0.00078]
        delta_old_nex = [0.21, 0.098, 0.057, 0.056, 0.055, 0.054, 0.053, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052]
        delta_new_scale_05 = [0.17, 0.37, 0.57, 0.59, 0.6, 0.62, 0.64, 0.66, 0.67, 0.67, 0.67, 0.67, 0.68, 0.68]  # v = 0.52
        delta_new_scale_03 = [0.4, 0.18, 0.39, 0.40, 0.41, 0.44, 0.46, 0.49, 0.5, 0.5, 0.5, 0.5, 0.51, 0.51]  # v = 0.26
        delta_new_scale_01 = [0.68, 0.45, 0.17, 0.18, 0.19, 0.22, 0.26, 0.3, 0.31, 0.32, 0.32, 0.33, 0.33, 0.33]  # v = 0.08
        # delta_new_scale_009 = []  # v = 0.071
        # delta_new_scale_008 = []  # v = 0.063
        # delta_new_scale_006 = []  # v = 0.047
        delta_new_scale_005 = [0.75, 0.53, 0.19, 0.16, 0.17, 0.17, 0.18, 0.21, 0.23, 0.23, 0.24, 0.24, 0.24, 0.25]  # v = 0.04
        # delta_new_scale_004 = []  # v = 0.031
        # delta_new_scale_002 = []  # v = 0.016
        delta_new_scale_001 = [0.75, 0.56, 0.28, 0.26, 0.25, 0.21, 0.19, 0.17, 0.17, 0.18, 0.18, 0.18, 0.18, 0.18]  # v = 0.008
        # delta_new_scale_0009 = []  # v = 0.007
        # delta_new_scale_0007 = []  # v = 0.0055
        delta_new_scale_0005 = [0.71, 0.55, 0.27, 0.25, 0.24, 0.21, 0.18, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17]  # v = 0.004
        # delta_new_scale_0003 = []  # v = 0.0023
        delta_new_scale_0001 = [0.76, 0.6, 0.32, 0.3, 0.28, 0.25, 0.21, 0.18, 0.18, 0.17, 0.17, 0.17, 0.17, 0.17]  # v = 0.00078
        # plt.plot(v_vals, delta_old_nex, marker='o', label="old-neuralExp")
        plt.plot(v_vals, delta_new_scale_05, marker='o', label="new-v-0.52")
        plt.plot(v_vals, delta_new_scale_03, marker='o', label="new-v-0.26")
        plt.plot(v_vals, delta_new_scale_01, marker='o', label="new-v-0.08")
        plt.plot(v_vals, delta_new_scale_005, marker='o', label="new-v-0.04")
        plt.plot(v_vals, delta_new_scale_001, marker='o', label="new-v-0.008")
        plt.plot(v_vals, delta_new_scale_0005, marker='o', label="new-v-0.004")
        plt.plot(v_vals, delta_new_scale_0001, marker='o', label="new-v-0.00078")

    # plt.ylim(0, 10)
    # plt.fill_between(x, y1, y4, color='yellow')
    plt.legend()
    plt.show()


def plotDelta(benchmark):

    bench1_v_vals = [0.18, 0.11, 0.037, 0.033, 0.029, 0.022, 0.015, 0.007, 0.0037, 0.0033, 0.0026, 0.0018, 0.0011, 0.00036]
    bench1_delta_old_nex_lb = [0.016, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015]
    bench1_delta_old_nex = [0.078, 0.056, 0.047, 0.047, 0.047, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046]
    bench1_delta_old_nex_ub = [0.19, 0.14, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11]
    # delta_new_scale_05 = [0.082, 0.093, 0.049, 0.045, 0.041, 0.032, 0.022, 0.011, 0.0058, 0.0052, 0.0041, 0.0029, 0.0017, 0.00058]  # v = 0.18
    # delta_new_scale_03 = [0.11, 0.05, 0.027, 0.025, 0.023, 0.018, 0.013, 0.0067, 0.0034, 0.0031, 0.0024, 0.0017, 0.001, 0.00035]  # v = 0.11
    # delta_new_scale_01 = [0.19, 0.067, 0.0096, 0.0086, 0.0077, 0.0061, 0.0044, 0.0024, 0.0013, 0.0011, 0.00089, 0.00064, 0.00039, 0.00013]  # v = 0.037
    # delta_new_scale_009 = []  # v = 0.033
    # delta_new_scale_008 = []  # v = 0.029
    # delta_new_scale_006 = []  # v = 0.022
    # delta_new_scale_005 = [0.24, 0.089, 0.013, 0.011, 0.0095, 0.0063, 0.0038, 0.0018, 0.0009, 0.0008, 0.0006, 0.00045, 0.00027, 0.00009]  # v = 0.018
    # delta_new_scale_004 = []  # v = 0.015
    # delta_new_scale_002 = []  # v = 0.007
    bench1_delta_new_scale_001 = [0.22, 0.089, 0.017, 0.015, 0.013, 0.0093, 0.0058, 0.0028, 0.0013, 0.0012, 0.0009, 0.0007, 0.0004, 0.00013]  # v = 0.0037
    # delta_new_scale_0009 = []  # v = 0.0033
    # delta_new_scale_0007 = []  # v = 0.0026
    # delta_new_scale_0005 = [0.25, 0.11, 0.024, 0.021, 0.018, 0.013, 0.0085, 0.0041, 0.0021, 0.0018, 0.0014, 0.001, 0.00062, 0.00021]  # v = 0.0018
    # delta_new_scale_0003 = []  # v = 0.0011
    # delta_new_scale_0001 = [0.21, 0.082, 0.014, 0.013, 0.011, 0.0078, 0.0052, 0.0026, 0.0013, 0.0012, 0.0009, 0.0007, 0.0004, 0.00014]  # v = 0.00036
    # plt.plot(v_vals, delta_new_scale_05, marker='o', label="new-v-0.18")
    # plt.plot(v_vals, delta_new_scale_03, marker='o', label="new-v-0.11")
    # plt.plot(v_vals, delta_new_scale_01, marker='o', label="new-v-0.037")
    # plt.plot(v_vals, delta_new_scale_005, marker='o', label="new-v-0.018")
    # plt.plot(v_vals, delta_new_scale_001, marker='o', label="new-v-0.0037")
    # plt.plot(v_vals, delta_new_scale_0005, marker='o', label="new-v-0.0018")
    # plt.plot(v_vals, delta_new_scale_0001, marker='o', label="new-v-0.00036")

    bench2_v_vals = [0.26, 0.08, 0.071, 0.063, 0.047, 0.031, 0.016, 0.008, 0.007, 0.0055, 0.004, 0.0023, 0.00078]
    bench2_delta_old_nex_lb = [0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    bench2_delta_old_nex = [0.10, 0.061, 0.06, 0.059, 0.058, 0.057, 0.056, 0.056, 0.056, 0.056, 0.056, 0.056, 0.056]
    bench2_delta_old_nex_ub = [0.43, 0.23, 0.13, 0.13, 0.13, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11]
    # delta_new_scale_05 = [0.062, 0.13, 0.086, 0.079, 0.072, 0.057, 0.039, 0.021, 0.011, 0.095, 0.0074, 0.0053, 0.0032, 0.0011]  # v = 0.52
    # delta_new_scale_03 = [0.19, 0.032, 0.054, 0.051, 0.047, 0.039, 0.028, 0.015, 0.0078, 0.0071, 0.0055, 0.004, 0.0024, 0.0008]  # v = 0.26
    # delta_new_scale_01 = [0.54, 0.20, 0.02, 0.017, 0.015, 0.012, 0.009, 0.005, 0.0026, 0.0024, 0.0019, 0.0014, 0.0083, 0.0003]  # v = 0.08
    # delta_new_scale_009 = []  # v = 0.071
    # delta_new_scale_008 = []  # v = 0.063
    # delta_new_scale_006 = []  # v = 0.047
    # delta_new_scale_005 = [0.59, 0.21, 0.02, 0.015, 0.012, 0.0076, 0.0053, 0.0032, 0.0018, 0.0016, 0.0013, 0.001, 0.0006, 0.0002]  # v = 0.04
    # delta_new_scale_004 = []  # v = 0.031
    # delta_new_scale_002 = []  # v = 0.016
    # bench2_delta_new_scale_001 = [0.6, 0.23, 0.03, 0.024, 0.02, 0.013, 0.0068, 0.0028, 0.0013, 0.0012, 0.001, 0.0007,
    #                               0.0004, 0.0001]  # v = 0.008
    bench2_delta_new_scale_001 = [0.23, 0.03, 0.024, 0.02, 0.013, 0.0068, 0.0028, 0.0013, 0.0012, 0.001, 0.0007, 0.0004, 0.0001]  # v = 0.008
    # delta_new_scale_0009 = []  # v = 0.007
    # delta_new_scale_0007 = []  # v = 0.0055
    # delta_new_scale_0005 = [0.57, 0.22, 0.029, 0.025, 0.021, 0.014, 0.0083, 0.004, 0.002, 0.0018, 0.0014, 0.001, 0.0006, 0.0002]  # v = 0.004
    # delta_new_scale_0003 = []  # v = 0.0023
    # delta_new_scale_0001 = [0.55, 0.21, 0.027, 0.023, 0.019, 0.012, 0.007, 0.003, 0.0017, 0.0015, 0.0012, 0.0009, 0.0005, 0.00017]  # v = 0.00078
    # plt.plot(v_vals, delta_new_scale_05, marker='o', label="new-v-0.52")
    # plt.plot(v_vals, delta_new_scale_03, marker='o', label="new-v-0.26")
    # plt.plot(v_vals, delta_new_scale_01, marker='o', label="new-v-0.08")
    # plt.plot(v_vals, delta_new_scale_005, marker='o', label="new-v-0.04")
    # plt.plot(v_vals, delta_new_scale_001, marker='o', label="new-v-0.008")
    # plt.plot(v_vals, delta_new_scale_0005, marker='o', label="new-v-0.004")
    # plt.plot(v_vals, delta_new_scale_0001, marker='o', label="new-v-0.00078")

    bench3_v_vals = [0.14, 0.08, 0.03, 0.025, 0.022, 0.017, 0.011, 0.0056, 0.0028, 0.0025, 0.002, 0.0014, 0.0008, 0.0003]
    bench3_delta_old_nex = [0.10, 0.065, 0.055, 0.054, 0.054, 0.054, 0.054, 0.053, 0.053, 0.053, 0.053, 0.053, 0.053,
                            0.053]

    bench3_delta_new_scale_001 = [0.13, 0.061, 0.017, 0.016, 0.014, 0.01, 0.0069, 0.0035, 0.0017, 0.0016, 0.0012, 0.0009,
                                  0.0005, 0.0002]

    singleP_v_vals = [0.21, 0.13, 0.04, 0.038, 0.033, 0.025, 0.017, 0.0084, 0.0042, 0.0038, 0.003, 0.0021, 0.0013, 0.00042]
    singleP_delta_old_nex = [0.12, 0.082, 0.073, 0.072, 0.072, 0.072, 0.072, 0.072, 0.072, 0.072, 0.072, 0.072, 0.072, 0.072]

    singleP_delta_new_scale_001 = [0.48, 0.20, 0.036, 0.031, 0.026, 0.018, 0.01, 0.0046, 0.0022, 0.0019, 0.0015,
                                  0.001, 0.0006, 0.0002]

    bench4_v_vals = [0.32, 0.18, 0.06, 0.055, 0.05, 0.047, 0.036, 0.011, 0.007, 0.006, 0.0035, 0.0027, 0.0024, 0.0006]
    bench4_delta_old_nex = [0.12, 0.06, 0.046, 0.047, 0.047, 0.046, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045,
                            0.045]

    colors = mcolors.CSS4_COLORS
    names = list(colors)
    print(names)

    for i, name in enumerate(names):
        print(i, colors[name], name)

    if benchmark is 'bench1':
        plt.plot(bench1_v_vals, bench1_delta_old_nex, marker='o', label="Mean delta")
        plt.plot(bench1_v_vals, bench1_delta_old_nex_lb, marker='o', label="10 quantile")
        plt.plot(bench1_v_vals, bench1_delta_old_nex_ub, marker='o', label="90 quantile")
        plt.fill_between(bench1_v_vals, bench1_delta_old_nex, bench1_delta_old_nex_lb, color=colors['lightskyblue'])
        plt.fill_between(bench1_v_vals, bench1_delta_old_nex, bench1_delta_old_nex_ub, color=colors['lightskyblue'])
    elif benchmark is 'bench2':
        plt.plot(bench2_v_vals, bench2_delta_old_nex, marker='o', label="Mean delta")
        plt.plot(bench2_v_vals, bench2_delta_old_nex_lb, marker='o', label="10 quantile")
        plt.plot(bench2_v_vals, bench2_delta_old_nex_ub, marker='o', label="90 quantile")
        plt.fill_between(bench2_v_vals, bench2_delta_old_nex, bench2_delta_old_nex_lb, color=colors['lightskyblue'])
        plt.fill_between(bench2_v_vals, bench2_delta_old_nex, bench2_delta_old_nex_ub, color=colors['lightskyblue'])
    else:
        plt.plot(bench1_v_vals, bench1_delta_old_nex, color='b', marker='o', linestyle=':', linewidth='2.0', label="NeuralExplorer - #1")
        plt.plot(bench2_v_vals, bench2_delta_old_nex, color='g', marker='o', linestyle=':', linewidth='2.0', label="NeuralExplorer - #2")
        plt.plot(bench3_v_vals, bench3_delta_old_nex, color='magenta', marker='o', linestyle=':', linewidth='2.0', label="NeuralExplorer - #3")
        plt.plot(singleP_v_vals, singleP_delta_old_nex, color='k', marker='o', linestyle=':', linewidth='2.0', label="NeuralExplorer - #4")
        plt.plot(bench1_v_vals, bench1_delta_new_scale_001, color='b', marker='o', linestyle='--', linewidth='2.0', label="NExG - #1")
        plt.plot(bench2_v_vals, bench2_delta_new_scale_001, color='g', marker='o', linestyle='--', linewidth='2.0', label="NExG - #2")
        plt.plot(bench3_v_vals, bench3_delta_new_scale_001, color='magenta', marker='o', linestyle='--', linewidth='2.0', label="NExG - #3")
        plt.plot(singleP_v_vals, singleP_delta_new_scale_001, color='k', linestyle='--', linewidth='2.0', marker='o', label="NExG - #4")
        # plt.plot(bench4_v_vals, bench4_delta_old_nex, marker='o', label="Benchmark 4")

    plt.xlabel(r'$r$', fontsize=24, fontweight="bold")
    plt.ylabel(r'$\epsilon_{abs}$', fontsize=24, fontweight="bold")

    plt.xticks(fontsize=20, fontweight="bold")
    plt.yticks(fontsize=20, fontweight="bold")
    plt.rcParams.update({'font.size': 20})
    plt.yscale("log")
    plt.xscale("log")
    # plt.ylim(0, 10)
    # plt.fill_between(x, y1, y4, color='yellow')
    plt.legend()
    plt.show()


def plotMSE(benchmark):

    if benchmark == 'bench1':
        v_vals = [0.18, 0.11, 0.037, 0.033, 0.029, 0.022, 0.015, 0.007, 0.0037, 0.0033, 0.0026, 0.0018, 0.0011, 0.00036]
        mse_old_nex = [0.078, 0.056, 0.047, 0.047, 0.047, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046]
        mse_new_scale_05 = [0.16, 0.31, 0.49, 0.50, 0.51, 0.53, 0.55, 0.57, 0.58, 0.58, 0.58, 0.58, 0.58, 0.59]  # v = 0.18
        mse_new_scale_03 = [0.22, 0.16, 0.27, 0.28, 0.28, 0.30, 0.32, 0.33, 0.34, 0.34, 0.34, 0.35, 0.35, 0.35]  # v = 0.11
        mse_new_scale_01 = [0.43, 0.27, 0.12, 0.11, 0.11, 0.11, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.13, 0.14]  # v = 0.037
        # mse_new_scale_009 = []  # v = 0.033
        # mse_new_scale_008 = []  # v = 0.029
        # mse_new_scale_006 = []  # v = 0.022
        mse_new_scale_005 = [0.35, 0.21, 0.1, 0.095, 0.094, 0.096, 0.1, 0.11, 0.12, 0.12, 0.12, 0.12, 0.12, 0.13]  # v = 0.018
        # mse_new_scale_004 = []  # v = 0.015
        # mse_new_scale_002 = []  # v = 0.007
        mse_new_scale_001 = [0.43, 0.30, 0.17, 0.17, 0.16, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.13, 0.13, 0.13]  # v = 0.0037
        # mse_new_scale_0009 = []  # v = 0.0033
        # mse_new_scale_0007 = []  # v = 0.0026
        mse_new_scale_0005 = [0.49, 0.35, 0.21, 0.21, 0.2, 0.19, 0.18, 0.17, 0.17, 0.16, 0.16, 0.16, 0.16, 0.16]  # v = 0.0018
        # mse_new_scale_0003 = []  # v = 0.0011
        mse_new_scale_0001 = [0.42, 0.27, 0.14, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.14, 0.14]  # v = 0.00036
        # plt.plot(v_vals, mse_old_nex, marker = 'o', label="old-neuralExp")
        plt.plot(v_vals, mse_new_scale_05, marker='o', label="new-v-0.18")
        plt.plot(v_vals, mse_new_scale_03, marker='o', label="new-v-0.11")
        plt.plot(v_vals, mse_new_scale_01, marker='o', label="new-v-0.037")
        plt.plot(v_vals, mse_new_scale_005, marker='o', label="new-v-0.018")
        plt.plot(v_vals, mse_new_scale_001, marker='o', label="new-v-0.0037")
        plt.plot(v_vals, mse_new_scale_0005, marker='o', label="new-v-0.0018")
        plt.plot(v_vals, mse_new_scale_0001, marker='o', label="new-v-0.00036")
    else:
        v_vals = [0.52, 0.26, 0.08, 0.071, 0.063, 0.047, 0.031, 0.016, 0.008, 0.007, 0.0055, 0.004, 0.0023, 0.00078]
        mse_old_nex = [0.22, 0.10, 0.061, 0.06, 0.059, 0.058, 0.057, 0.056, 0.056, 0.056, 0.056, 0.056, 0.056, 0.056]
        mse_new_scale_05 = [0.12, 0.43, 0.86, 0.88, 0.90, 0.94, 0.99, 1.03, 1.05, 1.06, 1.06, 1.06, 1.07, 1.07]  # v = 0.52
        mse_new_scale_03 = [0.39, 0.11, 0.54, 0.57, 0.6, 0.65, 0.7, 0.75, 0.78, 0.78, 0.79, 0.79, 0.8, 0.8]  # v = 0.26
        mse_new_scale_01 = [1.1, 0.65, 0.2, 0.19, 0.18, 0.19, 0.21, 0.25, 0.26, 0.27, 0.27, 0.27, 0.28, 0.28]  # v = 0.08
        # mse_new_scale_009 = []  # v = 0.071
        # mse_new_scale_008 = []  # v = 0.063
        # mse_new_scale_006 = []  # v = 0.047
        mse_new_scale_005 = [1.17, 0.71, 0.20, 0.17, 0.15, 0.13, 0.13, 0.16, 0.18, 0.18, 0.19, 0.19, 0.2, 0.2]  # v = 0.04
        # mse_new_scale_004 = []  # v = 0.031
        # mse_new_scale_002 = []  # v = 0.016
        mse_new_scale_001 = [1.2, 0.77, 0.29, 0.27, 0.25, 0.21, 0.17, 0.14, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13]  # v = 0.008
        # mse_new_scale_0009 = []  # v = 0.007
        # mse_new_scale_0007 = []  # v = 0.0055
        mse_new_scale_0005 = [1.14, 0.73, 0.29, 0.27, 0.26, 0.23, 0.21, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.21]  # v = 0.004
        # mse_new_scale_0003 = []  # v = 0.0023
        mse_new_scale_0001 = [1.09, 0.7, 0.27, 0.25, 0.23, 0.2, 0.18, 0.17, 0.17, 0.17, 0.17, 0.18, 0.18, 0.18]  # v = 0.00078
        # plt.plot(v_vals, delta_old_nex, marker='o', label="old-neuralExp")
        plt.plot(v_vals, mse_new_scale_05, marker='o', label="new-v-0.52")
        plt.plot(v_vals, mse_new_scale_03, marker='o', label="new-v-0.26")
        plt.plot(v_vals, mse_new_scale_01, marker='o', label="new-v-0.08")
        plt.plot(v_vals, mse_new_scale_005, marker='o', label="new-v-0.04")
        plt.plot(v_vals, mse_new_scale_001, marker='o', label="new-v-0.008")
        plt.plot(v_vals, mse_new_scale_0005, marker='o', label="new-v-0.004")
        plt.plot(v_vals, mse_new_scale_0001, marker='o', label="new-v-0.0008")

    # plt.ylim(0, 10)
    # plt.fill_between(x, y1, y4, color='yellow')
    plt.xlabel('v')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    plotDelta('All')
    # plotMSE('bench2')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
