from TrafficSensorAlg_V3 import *

"""
This script contains the main functions and modules for reproducing the results generated in the paper
'Traffic detection via energy efficient passive infrared sensing' submitted to IEEE Transactions on ITS in 2017.
"""

__author__ = 'Yanning Li'
__version__ = "3.0"
__email__ = 'yli171@illinois.edu, yanning.k.li@gmail.com'


def main():
    """
    The main function calls modules to
        - preprocess the collected data (preprocess_data)
        - run the vehicle detection and speed estimation algorithm (run_traffic_detection)
        - evaluate the detection accuracy (evaluate_accuracy)

    COMMENT/UNCOMMENT the corresponding function run each function one by one.
    :return:
    """

    # ===============================================================================================
    # Run preprocessing data
    #     - loading the raw data from txt file to pandas data frame structure
    #     - visualizing the raw PIR and ultrasonic sensor data
    #     - visualizing the evolution and distribution of the temperature data from each pixel
    #     - performing background subtraction and data normalization using KF and FSM
    #     - visualizing the normalized data
    # ===============================================================================================
    # preprocess_data()

    # ===============================================================================================
    # Run vehicle detection and speed estimation algorithm on the processed data.
    # This function runs on the processed data from process_data()
    # ===============================================================================================
    run_traffic_detection()


    # ===============================================================================================
    # Run post processing result, including
    #   - trimming detection results and video ground truth into interested period
    #   - DEBUG ONLY, quickly regenerate the result with new ultrasonic sensor parameters without
    #     reruning the traffic detection algorithm
    #   - correcting the time drift of the ground truth caused by the constant video drift assumption
    #   - combine two sensors detection result for two lanes
    # ===============================================================================================
    # post_process()

    # ===============================================================================================
    # Run the following function to evaluate the traffic detection accuracy.
    # ===============================================================================================
    # evaluate_accuracy()


    plt.show()

def preprocess_data():
    """
    This function is used to perform preprocessing and analysis of the data, including
        - loading the raw data from txt file to pandas data frame structure
        - visualizing the raw PIR and ultrasonic sensor data
        - visualizing the evolution and distribution of the temperature data from each pixel
        - performing background subtraction and data normalization using KF and FSM
        - visualizing the normalized data
    :return:
    """

    # ===============================================================================================
    # Specify the directory of the data to be processed
    # ===============================================================================================

    # --------------------------------------------------
    # Field experiment Test 1
    # Deployed one sensor on 6th street (one-lane one direction) near the parking lot with ~18 mph average speeds.
    # Collected data between 21:40 ~ 22:20 UTC on Jun 08, 2017
    # folder = 'Jun08_2017'                             # folder of the dataset
    # sensor = 's1'                                     # sensor id
    # data_dir = '../datasets/{0}/s1/'.format(folder)
    # dataset = '0806-201711_s1'                        # dataset name

    # --------------------------------------------------
    # Field experiment Test 2
    # Deployed one sensor on 6th street (one-lane one direction) near the parking lot with ~18 mph average speeds.
    # Collected data between 19:10~20:39 UTC Jun 09, 2017
    # folder = 'Jun09_2017'                             # folder of the dataset
    # data_dir = '../datasets/{0}/s1/'.format(folder)
    # dataset = '0906-190351_s1'                        # dataset name

    # --------------------------------------------------
    # Field experiment Test 3
    # Deployed two sensors, one on each side of Neil street in Savoy for freeflow traffic at ~45 mph average speeds
    # Collected data between 20:55 ~ 21:45 UTC on May 30th, 2017
    # SENSOR 1
    folder = 'May30_2017'                                # folder of the dataset
    sensor = 's1'                                       # sensor id
    data_dir = '../datasets/{0}/{1}/'.format(folder, sensor)
    dataset = '3005-205500_{0}'.format(sensor)          # dataset name

    # SENSOR 2
    # folder = 'May30_2017'
    # sensor = 's2'                                     # sensor id
    # data_dir = '../datasets/{0}/{1}/'.format(folder, sensor)
    # dataset = '3005-203625_{0}'.format(sensor)        # dataset name



    # ===============================================================================================
    # Load the raw data from txt files to pandas dataframe structure
    # ===============================================================================================
    save_dir = '../workspace/{0}/'.format(folder)
    if not exists(save_dir): os.mkdir(save_dir)

    data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
    periods = data.get_data_periods(data_dir, update=True, f_type='txt')

    print('Loading txt data...')
    t1 = datetime.now()

    if folder == 'May30_2017' and sensor == 's1':
        # the two sensor components on s1 was misconnected which flipped the upper 64 rows with the bottom 64 rows in
        # the time space representation in the dataframe. Set flip to be true to correct.
        flip = True
    else:
        flip = False

    df = data.load_txt_data(data_dir+'{0}.txt'.format(dataset), flip=flip)
    t2 = datetime.now()
    print('Loading data took {0} s'.format((t2-t1).total_seconds()))


    # ===============================================================================================
    # Plot the original raw data
    # ===============================================================================================
    if False:
        # Use periods or str2time to define the period to be plotted
        # t_start = periods[dataset][0] #+ timedelta(seconds=180)
        # t_end = periods[dataset][0] # + timedelta(seconds=780)
        t_start = str2time('2017-05-30 21:21:00.0')
        t_end = str2time('2017-05-30 21:21:10.0')
        data.plot_heatmap_in_period(df, t_start=t_start, t_end=t_end, cbar=(35,45), option='vec',
                                    nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False, save_df=False,
                                    figsize=(18,8))

        # plot ultrasonic sensor
        plt.figure(figsize=(15,8))
        plt.plot(df.index, df['ultra'])

        plt.draw()

    # ===============================================================================================
    # Plot the evolution of the collected data from a single pixel and its histogram
    # ===============================================================================================
    if False:
        t_start = periods[dataset][0] + timedelta(seconds=350)
        t_end = periods[dataset][0]  + timedelta(seconds=650)
        data.plot_noise_evolution(df, t_start=t_start, t_end=t_end, p_outlier=0.01, stop_thres=(0.01,0.1),
                                  pixel=(2,20), window_s=10, step_s=3)

        data.plot_histogram_for_pixel(df, pixels=[(2,20)], t_start=t_start, t_end=t_end, p_outlier=0.01,
                                      stop_thres=(0.01, 0.1))

    # ===============================================================================================
    # Use Kalman filter and finite state machine to perform background subtraction and data normalization;
    # The normalized data is then saved to run the vehicle detection and speed estimation algorithm later on.
    # ===============================================================================================
    if False:
        t_start = periods[dataset][0] # + timedelta(seconds=200)
        t_end = periods[dataset][1] #+ timedelta(seconds=400)
        t1 = datetime.now()
        kf_norm_df = data.subtract_background_KF(df, t_start=t_start, t_end=t_end, init_s=30,
                                           veh_pt_thres=3, noise_pt_thres=3, prob_int=0.95,
                                                 pixels=None, debug=False)
        t2 = datetime.now()
        print('\nNormalization sequential KF took {0} s'.format((t2-t1).total_seconds()))

        kf_norm_df.to_csv(save_dir + '{0}_2d_KF__{1}__{2}_prob95.csv'.format(sensor, time2str_file(t_start),
                                                                time2str_file(t_end)))


    # ===============================================================================================
    # Visualized the previously saved heatmap (i.e., pandas data frame), which can be the raw data or normalized data
    # ===============================================================================================
    if False:

        # ---------------------------------------------------------------------------------------
        # load norm_df
        t_start = periods[dataset][0]  # + timedelta(seconds=200)
        t_end = periods[dataset][1]  # + timedelta(seconds=400)
        norm_df = pd.read_csv(save_dir+'{0}_2d_KF__{1}__{2}_prob95.csv'.format(sensor, time2str_file(t_start),
                                                                               time2str_file(t_end)), index_col=0)
        norm_df.index = norm_df.index.to_datetime()
        _, _ = data.plot_heatmap_in_period(norm_df, t_start=t_start, t_end=t_end, cbar=(0,4),
                                              option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
                                              save_df=False, figsize=(18,8))

        plt.draw()


def run_traffic_detection():
    """
    This function runs the vehicle detection and speed estimation algorithm for each PIR sensor. The two sensors in the
    third test are processed separately for vehicle detection and speed estimation.

    The detection results from two sensors are combined in the evaluation function in the current version of code.
    The combination of two the detection results should be moved to the traffic detection class later. Putting the combination in the
    evaluation class is purely for ease of implementation and does not affect the accuracy results presented in paper.

    Set False to True to run the detection algorithm for each data set.

    :return:
    """

    # ===============================================================================================
    # Test 1: dataset Jun 08, 2017 on 6th street down near the parking lot.
    # collected data 21:40 ~ 22:20 UTC
    # ===============================================================================================
    if False:

        # Set up the directory
        folder = 'Jun08_2017'
        save_dir = '../workspace/{0}/'.format(folder)

        # Measured the traffic in one direction, hence negative. The current version of the algorithm does not support
        # stopped traffic yet, hence the range starts from 1 mph.
        speed_range = (-71,-1)  # mph

        # read in the normalized data from the preprocess_data() function.
        norm_df = pd.read_csv(save_dir +
                              's1_2d_KF__20170608_213900_001464__20170608_222037_738293_prob95.csv', index_col=0)
        norm_df.index = norm_df.index.to_datetime()

        # Set up the algorithm parameters.
        # The parameters pir_res and sampling frequency are inherent to the sensor.
        # The parameters r2_thres, dens_thres, min_dt determines the detection performance. See paper for more deteails.
        alg = TrafficSensorAlg(r2_thres=0.2, dens_thres=0.4, min_dt=0.2, pir_res=(4,32), sampling_freq=64)

        # The detection window is adaptive to capture the full trajectory of the vehicle. But parameters window_s and
        # step_s sets the initial time window for detection.
        alg.run_adaptive_window(norm_df, window_s=5.0, step_s=2.5, speed_range=speed_range, plot_final=False,
                                plot_debug=False, save_dir='../workspace/{0}/'.format(folder), sensor='s1',
                                t_start=str2time('2017-06-08 21:40:00.0'), t_end=str2time('2017-06-08 22:20:00.0'))


    # ===============================================================================================
    # Test 2: dataset Jun 09, 2017, 19:10~20:39 UTC
    # ===============================================================================================
    if False:

        # Set up the directory
        folder = 'Jun09_2017'
        save_dir = '../workspace/{0}/'.format(folder)

        speed_range = (-71,-1)  # mph

        # read in the normalized data from the process_data() function.
        norm_df = pd.read_csv(save_dir +
                              's1_2d_KF__20170609_190900_009011__20170609_203930_905936_prob95.csv', index_col=0)
        norm_df.index = norm_df.index.to_datetime()

        # Set up the algorithm parameters.
        alg = TrafficSensorAlg(r2_thres=0.2, dens_thres=0.4, min_dt=0.2, pir_res=(4,32), sampling_freq=64)

        # Run the algorithm
        alg.run_adaptive_window(norm_df, window_s=5.0, step_s=2.5, speed_range=speed_range, plot_final=True,
                                plot_debug=False, save_dir='../workspace/{0}/'.format(folder), sensor='s1',
                                t_start=None, t_end=None)

    # ===============================================================================================
    # Test 3, sensor 1:  data set 0530, S1: 20:55 ~ 21:45 UTC
    # ===============================================================================================
    if False:

        # Set up directory
        folder = 'May30_2017'
        save_dir = '../workspace/{0}/'.format(folder)

        speed_range = (-71,-1)  # mph

        # read in the data with background subtracted.
        norm_df = pd.read_csv(save_dir +
                              's1_2d_KF__20170530_205400_0__20170530_214500_0_prob95.csv', index_col=0)
        norm_df.index = norm_df.index.to_datetime()

        # Set up algorithm parameters
        alg = TrafficSensorAlg(r2_thres=0.2, dens_thres=0.4, min_dt=0.2, pir_res=(4,32), sampling_freq=64)

        # Run detection algorithm using adaptive window
        alg.run_adaptive_window(norm_df, window_s=5.0, step_s=2.5, speed_range=speed_range, plot_final=True,
                                plot_debug=False, save_dir='../workspace/{0}/'.format(folder), sensor='s1',
                                t_start=str2time('2017-05-30 20:55:0.0'),
                                t_end=str2time('2017-05-30 21:45:00.0'))

    # ===============================================================================================
    # Test 3, sensor 2: data set 0530, S2: 20:55 ~ 21:45 UTC
    # ===============================================================================================
    if False:

        # Set up directory
        folder = 'May30_2017'
        save_dir = '../workspace/{0}/'.format(folder)

        # This sensor is on the other side of the road, hence the speed direction measured in positive.
        speed_range = (1,71)

        # read in the data with background subtracted.
        norm_df = pd.read_csv(save_dir +
                              's2_2d_KF__20170530_205400_0__20170530_214500_0_prob95.csv', index_col=0)
        norm_df.index = norm_df.index.to_datetime()

        # Set up algorithm parameters
        alg = TrafficSensorAlg(r2_thres=0.2, dens_thres=0.4, min_dt=0.2, pir_res=(4,32), sampling_freq=64)

        # Run detection algorithm using adaptive windows.
        alg.run_adaptive_window(norm_df, window_s=5.0, step_s=2.5, speed_range=speed_range, plot_final=True,
                                plot_debug=False, save_dir='../workspace/{0}/'.format(folder), sensor='s2',
                                t_start=str2time('2017-05-30 20:55:00.0'),
                                t_end=str2time('2017-05-30 21:45:00.0'))

def post_process():
    """
    This function post processes the detection results, including
       - trimming detection results and video ground truth into interested period
       - DEBUG ONLY, quickly regenerate the result with new ultrasonic sensor parameters without
         rerunning the traffic detection algorithm
       - correcting the time drift of the ground truth caused by the constant video drift assumption
       - combine two sensors detection result for two lanes

    SET TRUE/FALSE to run the post processing for each dataset

    :return:
    """
    # ===============================================================================================
    # data set Jun08, 2017, S1: trim to 21:40~22:20 UTC
    # ===============================================================================================
    if False:
        folder = 'Jun08_2017'
        save_dir = '../workspace/{0}/'.format(folder)

        t_period_start = str2time('2017-06-08 21:40:00.0')
        t_period_end = str2time('2017-06-08 22:20:00.0')

        vehs_npy = save_dir + 's1_detected_vehs.npy'

        # Ground truth
        init_t = str2time('2017-06-08 21:32:18.252811')
        offset = -0.28
        drift_ratio = 1860.5/1864.0
        true_file = save_dir + 'CV_truth_Jun08.npy'

        # Trimming
        post = PostProcess()

        # trim the norm_df
        if False:
            norm_df_file = save_dir + 's1_2d_KF__20170608_213900_001464__20170608_222037_738293_prob95.csv'
            post.trim_norm_df(norm_df_file, t_period_start-timedelta(seconds=60), t_period_end)

        if True:

            post.postprocess_detection(vehs_npy, t_period_start, t_period_end,
                                 ultra_fp_lb=4.0, ultra_fp_ub=8.0, speed_range=(0.0, 30.0))

        if False:
            post.postprocess_true(true_file, init_t, offset, drift_ratio, t_period_start, t_period_end)


    # ===============================================================================================
    # data set Jun09, 2017, S1, trim to 19:10~20:39
    # ===============================================================================================
    if True:
        folder = 'Jun09_2017'
        save_dir = '../workspace/{0}/'.format(folder)

        t_period_start = str2time('2017-06-09 19:10:00.0')
        t_period_end = str2time('2017-06-09 20:39:00.0')

        vehs_npy = save_dir + 'detected_vehs.npy'

        # Ground truth
        init_t = str2time('2017-06-09 19:09:00.0')
        offset = -0.66
        drift_ratio = 1860.5/1864.0
        true_file = save_dir + 'CV_truth_Jun09.npy'

        # Trimming
        post = PostProcess()

        # trim the norm_df
        if False:
            norm_df_file = save_dir + 's1_2d_KF__20170609_190900_009011__20170609_203930_905936_prob95.csv'
            post.trim_norm_df(norm_df_file, t_period_start-timedelta(seconds=60), t_period_end)

        if True:

            post.postprocess_detection(vehs_npy, t_period_start, t_period_end,
                                 ultra_fp_lb=4.0, ultra_fp_ub=8.0, speed_range=(0.0, 30.0))

        if False:
            post.postprocess_true(true_file, init_t, offset, drift_ratio, t_period_start, t_period_end)


    # ===============================================================================================
    # data set 0530, 2017, S1: 2017-05-30 20:55:00.0 ~ 2017-05-30 21:45:00.0
    # ===============================================================================================
    if False:
        folder = 'May30_2017'
        t_period_start = str2time('2017-05-30 20:55:00.0')
        t_period_end = str2time('2017-05-30 21:45:00.0')

        # Configuration
        save_dir = '../workspace/{0}/'.format(folder)

        # Detections
        vehs_npy = save_dir + 's1_detected_vehs.npy'
        norm_df_file = save_dir + 's1_2d_KF__20170530_205400_0__20170530_214500_0_prob95.csv'

        # Ground truth
        # 30/05: 2017-05-30 20:55:00 ~ 2017-05-30 21:55:00
        init_t = str2time('2017-05-30 20:55:00.0')
        offset = -0.48
        drift_ratio = 1860.6/1864.0
        true_file = save_dir + 'CV_truth_v11.npy'

        # Trimming
        post = PostProcess()

        # trim the norm_df
        if False:
            post.trim_norm_df(norm_df_file, t_period_start-timedelta(seconds=60), t_period_end)

        if False:

            post.postprocess_detection(vehs_npy, t_period_start, t_period_end,
                                 ultra_fp_lb=None, ultra_fp_ub=8.0, speed_range=(0.0, 70.0))

        if True:
            post.postprocess_true(true_file, init_t, offset, drift_ratio, t_period_start, t_period_end)

    # ===============================================================================================
    # data set 0530, 2017, S2: 2017-05-30 20:55:00.0 ~ 2017-05-30 21:45:00.0
    # ===============================================================================================
    if False:
        folder = 'May30_2017'
        t_period_start = str2time('2017-05-30 20:55:00.0')
        t_period_end = str2time('2017-05-30 21:45:00.0')

        # Configuration
        save_dir = '../workspace/{0}/'.format(folder)

        # Detection
        vehs_npy = save_dir + 's2_detected_vehs.npy'
        norm_df_file = save_dir + 's2_2d_KF__20170530_205400_0__20170530_214500_0_prob95.csv'

        # Ground truth
        # 30/05: 2017-05-30 20:55:00 ~ 2017-05-30 21:55:00
        init_t = str2time('2017-05-30 20:55:00.0')
        offset = 2.102
        drift_ratio = 1860.6/1864.0
        true_file = save_dir + 'CV_truth_v21.npy'

        # Trimming
        post = PostProcess()

        # trim the norm_df
        if False:
            post.trim_norm_df(norm_df_file, t_period_start-timedelta(seconds=60), t_period_end)

        if False:

            post.postprocess_detection(vehs_npy, t_period_start, t_period_end,
                                 ultra_fp_lb=None, ultra_fp_ub=8.0, speed_range=(0.0, 70.0))

        if True:
            post.postprocess_true(true_file, init_t, offset, drift_ratio, t_period_start, t_period_end)


        # -----------------------------------------------------------------------------------------------
        # Combine detections from two lanes
        if False:
            folder = 'May30_2017'
            save_dir = '../workspace/{0}/'.format(folder)

            s1_vehs_npy = save_dir + 's1_detected_vehs_post.npy'
            s2_vehs_npy = save_dir + 's2_detected_vehs_post.npy'

            # combine detection
            s1_s2_shift = 0.33
            post.combine_two_lane_detections(s1_vehs_npy, s2_vehs_npy, s1_s2_shift=s1_s2_shift, dt=0.5, save_dir=save_dir,
                                           speed_range=(0.0,70.0))



def evaluate_accuracy():
    """
    This function plot the detection results. MAKE SURE the run trim_results first.
    :return:
    """
    # ===============================================================================================
    # data set 0530, 2017: 2017-05-30 20:55:00.0 ~ 2017-05-30 21:45:00.0
    # SET TRUE/FALSE TO
    #   - plot s1 detections vs truth
    #   - plot s2 detections vs truth
    #   - plot combined s1-s2 detections vs truth
    #   - compute error statistics
    #   - plot error histogram
    # ===============================================================================================

    # -----------------------------------------------------------------------------------------------
    # S1: plot s1 detection vs true
    if False:
        folder = 'May30_2017'

        # Configuration
        save_dir = '../workspace/{0}/'.format(folder)
        paras_file = save_dir + 's1_detector_paras.txt'
        norm_df_file = save_dir + 's1_2d_KF__20170530_205400_0__20170530_214500_0_prob95.csv'
        est_vehs_npy = save_dir + 's1_detected_vehs_post.npy'
        true_file = save_dir + 'CV_truth_v11_post.npy'

        # Plot the detection VS true
        ev = EvaluatePerformance()

        # load norm df data
        norm_df = pd.read_csv(norm_df_file, index_col=0)
        norm_df.index = norm_df.index.to_datetime()
        true_vehs = np.load(true_file)
        det_vehs = np.load(est_vehs_npy)

        ev.plot_single_lane_detection_vs_true(norm_df, paras_file, vehs=det_vehs, true_vehs=true_vehs,
                              t_start=str2time('2017-05-30 21:30:00.0'),
                              t_end=str2time('2017-05-30 21:35:00.0'))

    # -----------------------------------------------------------------------------------------------
    # S2: plot s2 detection vs true
    if False:
        folder = 'May30_2017'

        # Configuration
        save_dir = '../workspace/{0}/'.format(folder)
        paras_file = save_dir + 's2_detector_paras.txt'
        norm_df_file = save_dir + 's2_2d_KF__20170530_205500_0__20170530_214500_0_prob95.csv'
        est_vehs_npy = save_dir + 's2_detected_vehs_post.npy'
        true_file = save_dir + 'CV_truth_v21_post.npy'

        # Plot the detection VS true
        ev = EvaluatePerformance()

        # load norm df data
        norm_df = pd.read_csv(norm_df_file, index_col=0)
        norm_df.index = norm_df.index.to_datetime()
        true_vehs = np.load(true_file)
        true_vehs[:,2] = -true_vehs[:,2]
        det_vehs = np.load(est_vehs_npy)

        ev.plot_single_lane_detection_vs_true(norm_df,det_vehs, paras_file, true_vehs=true_vehs,
                              t_start=str2time('2017-05-30 21:35:00.0'),
                              t_end=str2time('2017-05-30 21:50:00.0'))

    # -----------------------------------------------------------------------------------------------
    # S1 and S2, plot combined detection vs true
    if False:
        folder = 'May30_2017'
        save_dir = '../workspace/{0}/'.format(folder)
        paras_file = save_dir + 's1_detector_paras.txt'

        s1_true = np.load(save_dir + 'CV_truth_v11_post.npy')
        s2_true = np.load(save_dir + 'CV_truth_v21_post.npy')
        s2_true[:,2] = -s2_true[:,2]

        s1_vehs_comb = np.load(save_dir + 's1_detected_vehs_post_combined.npy')
        s2_vehs_comb = np.load(save_dir + 's2_detected_vehs_post_combined.npy')

        debug_combined_vehs = np.load(save_dir + 'debug_combined_vehs.npy')

        ev = EvaluatePerformance()
        # Plot the detections vs truth
        if False:
            s1_norm_df = pd.read_csv(save_dir + 's1_2d_KF__20170530_205400_0__20170530_214500_0_prob95.csv', index_col=0)
            s1_norm_df.index = s1_norm_df.index.to_datetime()
            s2_norm_df = pd.read_csv(save_dir + 's2_2d_KF__20170530_205400_0__20170530_214500_0_prob95.csv', index_col=0)
            s2_norm_df.index = s2_norm_df.index.to_datetime()
            s1_s2_shift = 0.33
            ev.plot_two_lanes_detection_vs_true(s1_df=s1_norm_df, s1_vehs=s1_vehs_comb, s1_true=s1_true,
                                     s2_df=s2_norm_df, s2_vehs=s2_vehs_comb, s2_true=s2_true,
                                     s1s2_shift=s1_s2_shift, s1_paras_file=paras_file,
                                     t_start=str2time('2017-05-30 21:29:00.0'),
                                     t_end=str2time('2017-05-30 21:44:00.0'),
                                     debug_combined_vehs=debug_combined_vehs)

        # Match vehicle and compute statistics
        if True:
            matched_vehs = ev.match_two_lanes_detection_with_true(s1_vehs_comb, s2_vehs_comb,
                                                           s1_true, s2_true, dt=0.6)
            matched_vehs = np.asarray(matched_vehs)

            matched_vehs[:,2] = np.abs(matched_vehs[:,2])
            matched_vehs[:,3] = np.abs(matched_vehs[:,3])

            ev.compute_statistics(matched_vehs)

            # Plot per vehicle error distribution
            tp_idx = ~pd.isnull(matched_vehs[:,0]) & ~pd.isnull(matched_vehs[:,1])
            ev.plot_hist([matched_vehs[tp_idx,2]-matched_vehs[tp_idx,3]], labels=None,
                         title='Test 3 per vehicle speed error', xlabel='Speed (mph)',
                         fontsizes=(34, 30, 28), xlim=(-25, 25), ylim=(0, 0.13), text_loc=(10, 0.11))

            # Plot one-minute speed error distribution
            one_min_err = ev.compute_aggregated_speed_error(matched_vehs[tp_idx,:], agg_s=60)
            one_min_err = np.asarray(one_min_err)
            rmse = np.sqrt( np.sum(one_min_err**2)/len(one_min_err) )
            print('One minute rmse: {0}'.format(rmse))
            ev.plot_hist([one_min_err], labels=None,
                         title='Test 3 aggregated speed error', xlabel='Speed (mph)',
                         fontsizes=(34, 30, 28), xlim=(-8, 8), ylim=(0, 0.42), text_loc=(3, 0.35))

            # find out false positive and false negatives
            if False:
                for v in matched_vehs:
                    if pd.isnull(v[1]):
                        print('False positive at {0}'.format(v[4]))

                for v in matched_vehs:
                    if pd.isnull(v[0]):
                        print('False negative at {0}'.format(v[5]))

            # find out outliers
            if False:
                for v in matched_vehs[tp_idx,:]:
                    if abs(v[2]-v[3]) >= 10:
                        print('Outlier speed error: {0:.3f} - {1:.3f}= {2:.3f}, {3}'.format(v[2], v[3],
                                                                                            v[2]-v[3], v[4]))


    # -----------------------------------------------------------------------------------------------
    # S1 and S2, plot combined speed and distance distribution
    if False:
        folder = 'May30_2017'
        save_dir = '../workspace/{0}/'.format(folder)

        s1_true = np.load(save_dir + 'CV_truth_v11_post.npy')
        s2_true = np.load(save_dir + 'CV_truth_v21_post.npy')

        s1_vehs_npy_comb = save_dir + 's1_detected_vehs_post_combined.npy'
        s2_vehs_npy_comb = save_dir + 's2_detected_vehs_post_combined.npy'

        true_count_l1 = len(s1_true)
        true_count_l2 = len(s2_true)

        print('True vehicles: Lane 1,  Lane 2,  Total')
        print('               {0},     {1},     {2}'.format(true_count_l1, true_count_l2,
                                                            true_count_l1+true_count_l2))
        # vehicle during the true period
        s1_vehs_comb = np.load(s1_vehs_npy_comb)
        s2_vehs_comb = np.load(s2_vehs_npy_comb)

        tmp_ev = EvaluatePerformance()
        count_l1 = tmp_ev.count_closer_lane_vehs(s1_vehs_comb)
        count_l2 = tmp_ev.count_closer_lane_vehs(s2_vehs_comb)

        print('Estimated    : Lane 1,  Lane 2,  Total')
        print('               {0},     {1},     {2}'.format(count_l1, count_l2, count_l1 + count_l2))

        # plot the distribution
        tmp_ev.plot_two_lanes_hist(s1_vehs_comb, s2_vehs_comb, s1_true, s2_true)



    # ===============================================================================================
    # dataset Jun 08, 2017 on 6th street down near the parking lot.
    # ===============================================================================================
    # Saved data is 21:40 ~ 22:20 UTC
    if False:
        folder = 'Jun08_2017'

        # Configuration
        save_dir = '../workspace/{0}/'.format(folder)
        est_vehs_npy = save_dir + 's1_detected_vehs_post.npy'
        true_file = save_dir + 'CV_truth_Jun08_post.npy'

        true_vehs = np.load(true_file)
        det_vehs = np.load(est_vehs_npy)

        ev = EvaluatePerformance()

        # plot the distribution of speed
        if False:
            # Get the distances and speeds
            true_distance, true_speed = [], []
            for v in true_vehs:
                true_distance.append(v[3])
                true_speed.append(abs(v[2]))

            est_distance, est_speed = [], []
            for v in det_vehs:
                est_distance.append(v['distance'])
                est_speed.append(abs(v['speed']))

            print('Detection: true,  est')
            print('          {0},     {1}'.format(len(true_speed), len(est_speed)))

            ev.plot_hist([true_distance, est_distance], ['true', 'est'],
                         title='Distance distribution', xlabel='Distance (m)')
            ev.plot_hist([true_speed, est_speed], ['true', 'est'],
                         title='Speed distribution', xlabel='Speed (mph)')

        # Match with true labels, and compute statistics
        if True:
            matched_vehs = ev.match_single_lane_detection_with_true(det_vehs, true_vehs, dt=0.5)
            matched_vehs = np.asarray(matched_vehs)

            ev.compute_statistics(matched_vehs)

            # plot per vehicle error
            tp_idx = ~pd.isnull(matched_vehs[:,0]) & ~pd.isnull(matched_vehs[:,1])
            ev.plot_hist([matched_vehs[tp_idx,2]-matched_vehs[tp_idx,3]], None,
                         title='Test 1 per vehicle speed error', xlabel='Speed (mph)',
                         fontsizes=(34, 30, 28), xlim=(-7, 7), ylim=(0, 0.5), text_loc=(3, 0.42))

            # plot one minute error
            one_min_err = ev.compute_aggregated_speed_error(matched_vehs[tp_idx,:], agg_s=60)
            one_min_err = np.asarray(one_min_err)
            rmse = np.sqrt( np.sum(one_min_err**2)/len(one_min_err) )
            print('One minute rmse: {0}'.format(rmse))
            ev.plot_hist([one_min_err], labels=None,
                         title='Test 1 aggregated speed error', xlabel='Speed (mph)',
                         fontsizes=(34, 30, 28), xlim=(-2, 3.5), ylim=(0, 0.75), text_loc=(1.8, 0.6))

            # find out false positive and false negatives
            if False:
                for v in matched_vehs:
                    if pd.isnull(v[1]):
                        print('False positive at {0}'.format(v[4]))

                for v in matched_vehs:
                    if pd.isnull(v[0]):
                        print('False negative at {0}'.format(v[5]))

            # find outliers
            if False:
                for v in matched_vehs[tp_idx,:]:
                    if abs(v[2]-v[3]) >= 4:
                        print('Outlier speed error: {0:.3f}, {1}'.format(v[2]-v[3], v[4]))

            # plot det vs true
            if False:
                paras_file = save_dir + 's1_detector_paras.txt'
                norm_df_file = save_dir + 's1_2d_KF__20170608_213900_001464__20170608_222037_738293_prob95.csv'
                norm_df = pd.read_csv(norm_df_file, index_col=0)
                norm_df.index = norm_df.index.to_datetime()

                ev.plot_single_lane_detection_vs_true(norm_df,det_vehs, paras_file, true_vehs=true_vehs,
                                  t_start=str2time('2017-06-08 22:05:00.0'),
                                  t_end=str2time('2017-06-08 22:15:00.0'), matches=matched_vehs)


    # ===============================================================================================
    # dataset Jun 09, 2017, same setup as Jun 08, 19:10~20:39
    # ===============================================================================================
    if True:
        folder = 'Jun09_2017'

        # Configuration
        save_dir = '../workspace/{0}/'.format(folder)
        est_vehs_npy = save_dir + 's2_detected_vehs_post.npy'
        true_file = save_dir + 'CV_truth_Jun09_post.npy'

        true_vehs = np.load(true_file)
        det_vehs = np.load(est_vehs_npy)

        ev = EvaluatePerformance()
        # plot the distribution
        if False:
            # Get the distances and speeds
            true_distance, true_speed = [], []
            for v in true_vehs:
                true_distance.append(v[3])
                true_speed.append(abs(v[2]))

            est_distance, est_speed = [], []
            for v in det_vehs:
                est_distance.append(v['distance'])
                est_speed.append(abs(v['speed']))

            print('Detection: true,  est')
            print('          {0},     {1}'.format(len(true_speed), len(est_speed)))

            ev.plot_hist([true_distance, est_distance], ['true', 'est'],
                         title='Distance distribution', xlabel='Distance (m)')
            ev.plot_hist([true_speed, est_speed], ['true', 'est'],
                         title='Speed distribution', xlabel='Speed (mph)')

        # Match with true labels, and compute statistics
        if True:
            # matched_vehs: [idx_veh, idx_true_veh, est_speed, true_speed, t_veh, t_true_veh]
            matched_vehs = ev.match_single_lane_detection_with_true(det_vehs, true_vehs, dt=1.5)
            matched_vehs = np.array(matched_vehs)

            if True:
                ev.compute_statistics(matched_vehs)

                # plot speed error distribution
                tp_idx = ~pd.isnull(matched_vehs[:,0]) & ~pd.isnull(matched_vehs[:,1])
                ev.plot_hist([matched_vehs[tp_idx,2]-matched_vehs[tp_idx,3]], labels=None,
                         title='Test 2 per vehicle speed error', xlabel='Speed (mph)',
                         fontsizes=(34, 30, 28), xlim=(-7, 6), ylim=(0, 0.35), text_loc=(2, 0.3))

                # plot one-minute error distribution
                one_min_err = ev.compute_aggregated_speed_error(matched_vehs[tp_idx,:], agg_s=60)
                one_min_err = np.asarray(one_min_err)
                rmse = np.sqrt( np.sum(one_min_err**2)/len(one_min_err) )
                print('One minute rmse: {0}'.format(rmse))
                ev.plot_hist([one_min_err], labels=None,
                         title='Test 2 aggregated speed error', xlabel='Speed (mph)',
                         fontsizes=(34, 30, 28), xlim=(-3.5, 2), ylim=(0, 0.9), text_loc=(0, 0.7))

            # find out false positive and false negatives
            if False:
                for v in matched_vehs:
                    if pd.isnull(v[1]):
                        print('False positive at {0}'.format(v[4]))

                for v in matched_vehs:
                    if pd.isnull(v[0]):
                        print('False negative at {0}'.format(v[5]))

            # find out outliers
            if False:
                for v in matched_vehs[tp_idx,:]:
                    if abs(v[2]-v[3]) >= 5:
                        print('Outlier speed error: {0:.3f} - {1:.3f}= {2:.3f}, {3}'.format(v[2], v[3],
                                                                                            v[2]-v[3], v[4]))

            # plot det vs true
            if False:

                paras_file = save_dir + 's1_detector_paras.txt'
                norm_df_file = save_dir + 's1_2d_KF__20170609_190900_009011__20170609_203930_905936_prob95.csv'
                norm_df = pd.read_csv(norm_df_file, index_col=0)
                norm_df.index = norm_df.index.to_datetime()

                ev.plot_single_lane_detection_vs_true(norm_df,det_vehs, paras_file, true_vehs=true_vehs,
                                  t_start=str2time('2017-06-09 19:25:00.0'),
                                  t_end=str2time('2017-06-09 19:35:00.0'), matches=matched_vehs)








#
# # To udpate
# def generate_video_frames():
#     # ===============================================================================================
#     # generate a video
#     folder = '1013_2016'
#     data_dir = '../datasets/{0}/'.format(folder)
#     dataset = '1310-210300'     # freeflow part 1
#     # dataset = '1310-213154'   # freeflow part 2
#     # dataset ='1310-221543'       # stop and go
#
#     # ========================================================
#     # Load the raw PIR data
#     print('Loading raw data...')
#     t1 = datetime.now()
#     raw_df = pd.read_csv(data_dir+'{0}.csv'.format(dataset), index_col=0)
#     raw_df.index = raw_df.index.to_datetime()
#     t2 = datetime.now()
#     print('Loading raw data csv data took {0} s'.format((t2-t1).total_seconds()))
#
#     # ========================================================
#     # Load the normalized data file
#     print('Loading normalized data...')
#     norm_df = pd.read_csv('../workspace/1013/s1_2d_MAP__20161013_210305_738478__20161013_211516_183544_prob95.csv', index_col=0)
#     norm_df.index = norm_df.index.to_datetime()
#
#     # ========================================================
#     # Load the detected vehicles
#     print('Loading detected vehicles...')
#     det_vehs = np.load('../workspace/1013/s1_vehs__20161013_210305_738478__20161013_211516_183544_prob95.npy')
#
#     # ========================================================
#     # Construct algorithm instance
#     print('Generating video frames...')
#     # _dir = '../workspace/1013/Video_ff1/'
#     _dir = '/data_fast/Yanning_sensors/video_1013_ff1_rgb/'
#     ev = EvaluatePerformance()
#     ev.plot_video_frames(video_file='../datasets/1013_2016/1013_v1_1_ff_hq.mp4', video_fps=60.1134421399,
#                           video_start_time=str2time('2016-10-13 20:57:27.5'),
#                           raw_df=raw_df, raw_pir_clim=(20,40),
#                           ratio_tx=6.0, norm_df=norm_df, norm_df_win=5.0, det_vehs=det_vehs,
#                           save_dir=_dir)
#
#
# def update_fps(offset1, offset2, T, fps):
#     """
#     This function updates the fps based on the offsets
#     :param offset1: the offset in seconds regarding to the real time
#     :param offset2: the increased or decreased offset in seconds regarding to the real time after T
#     :param T: the duration for the offset in seconds
#     :param fps: old fps
#     :return: new fps
#     """
#     return fps*(T-offset1+offset2)/T
#



if __name__ == '__main__':
    main()


