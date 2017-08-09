from TrafficSensorAlg_V3 import *

def main():

    # data_analysis()

    test_alg()

    # trim_results()

    # evaluation()

    plt.show()

def data_analysis():
    # --------------------------------------------------
    # data set 0530, 2017
    # Two sensors on both sides of road at the same location.
    # Freeflow on Neil in Savoy
    # data set is good to use
    folder = '0530_2017'
    sensor = 's1'
    data_dir = '../datasets/{0}/{1}/'.format(folder, sensor)
    dataset = '3005-205500_{0}'.format(sensor)

    # folder = '0530_2017'
    # sensor = 's2'
    # data_dir = '../datasets/{0}/{1}/'.format(folder, sensor)
    # dataset = '3005-203625_{0}'.format(sensor)

    # --------------------------------------------------
    # dataset Jun 08, 2017 on 6th street down near the parking lot.
    # Saved data is 21:40 ~ 22:20 UTC
    # folder = 'Jun08_2017'
    # sensor = 's1'
    # data_dir = '../datasets/{0}/s1/'.format(folder)
    # dataset = '0806-201711_s1'

    # --------------------------------------------------
    # dataset Jun 09, 2017, same setup as Jun 08
    # folder = 'Jun09_2017'
    # data_dir = '../datasets/{0}/s1/'.format(folder)
    # dataset = '0906-190351_s1'


    # ===============================================================================================
    # Configuration
    save_dir = '../workspace/{0}/'.format(folder)
    if not exists(save_dir): os.mkdir(save_dir)

    data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
    periods = data.get_data_periods(data_dir, update=True, f_type='txt')

    # either load txt or csv data. csv is faster
    if True:
        print('Loading txt data...')
        t1 = datetime.now()

        if folder == '0530_2017' and sensor == 's1':
            flip = True
        else:
            flip = False

        df = data.load_txt_data(data_dir+'{0}.txt'.format(dataset), flip=flip)
        t2 = datetime.now()
        print('Loading data took {0} s'.format((t2-t1).total_seconds()))

        # df.to_csv(folder+'{0}.csv'.format(dataset))
        # print('saved df data.')

    if False:
        t1 = datetime.now()
        df = pd.read_csv(data_dir+'{0}.csv'.format(dataset), index_col=0)
        df.index = df.index.to_datetime()
        t2 = datetime.now()
        print('Loading df csv data took {0} s'.format((t2-t1).total_seconds()))

    # ===============================================================================================
    # plot the original df
    if False:
        # t_start = periods[dataset][0] #+ timedelta(seconds=180)
        # t_end = periods[dataset][1] # + timedelta(seconds=780)
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
    # plot the evolution of the noise (350~650)
    if True:
        t_start = periods[dataset][0] + timedelta(seconds=350)
        t_end = periods[dataset][0]  + timedelta(seconds=650)
        data.plot_noise_evolution(df, t_start=t_start, t_end=t_end, p_outlier=0.01, stop_thres=(0.01,0.1),
                                  pixel=(2,20), window_s=10, step_s=3)

        data.plot_histogram_for_pixel(df, pixels=[(2,20)], t_start=t_start, t_end=t_end, p_outlier=0.01,
                                      stop_thres=(0.01, 0.1))

    # ===============================================================================================
    # analyze and debug KF background subtraction
    if False:
        t_start = periods[dataset][0] # + timedelta(seconds=200)
        t_end = periods[dataset][1] #+ timedelta(seconds=400)
        t1 = datetime.now()
        kf_norm_df = data.subtract_background_KF(df, t_start=t_start, t_end=t_end, init_s=30,
                                           veh_pt_thres=3, noise_pt_thres=3, prob_int=0.95,
                                                 pixels=None, debug=False)
        # kf_norm_df = data.subtract_background_KF(df, t_start=t_start, t_end=t_end, init_s=30,
        #                                    veh_pt_thres=3, noise_pt_thres=3, prob_int=0.95,
        #                                          pixels=[(1, 5)], debug=True)
        # kf_norm_df = data.subtract_background_KF(df, t_start=t_start, t_end=t_end, init_s=30,
        #                                    veh_pt_thres=3, noise_pt_thres=3, prob_int=0.95,
        #                                          pixels=[(2, 5)], debug=True)
        # kf_norm_df = data.subtract_background_KF(df, t_start=t_start, t_end=t_end, init_s=30,
        #                                    veh_pt_thres=3, noise_pt_thres=3, prob_int=0.95,
        #                                          pixels=[(3, 5)], debug=True)
        t2 = datetime.now()
        print('\nNormalization sequential KF took {0} s'.format((t2-t1).total_seconds()))

        kf_norm_df.to_csv(save_dir + '{0}_2d_KF__{1}__{2}_prob95.csv'.format(sensor, time2str_file(t_start),
                                                                time2str_file(t_end)))


    # ===============================================================================================
    # Use batch normalization to subtract the background noise
    if False:
        t_start = periods[dataset][0]
        t_end = periods[dataset][0] + timedelta(seconds=30)
        t1 = datetime.now()
        dis_norm_df = data.batch_normalization(df, t_start=t_start, t_end=t_end, p_outlier=0.01,
                                           stop_thres=(0.001,0.0001), window_s=5, step_s=1)
        t2 = datetime.now()
        print('Normalization using iterative distribution took {0} s'.format((t2-t1).total_seconds()))

        # norm_df.to_csv(save_dir + 's3_2d_{0}.csv'.format(time2str_file(periods[dataset][0])))
        fig, ax = data.plot_heatmap_in_period(dis_norm_df, t_start=t_start, t_end=t_end, cbar=(0,4),
                                              option='vec', nan_thres_p=0.95, plot=True, save_dir=save_dir, save_img=False,
                                              save_df=False, figsize=(18,8))


    # ===============================================================================================
    # plot saved heatmap
    if False:
        # load norm_df
        t_start = periods[dataset][0]  # + timedelta(seconds=200)
        t_end = periods[dataset][1]  # + timedelta(seconds=400)
        norm_df = pd.read_csv(save_dir+'{0}_2d_KF__{1}__{2}_prob95.csv'.format(sensor, time2str_file(t_start),
                                                                               time2str_file(t_end)),
                              index_col=0)
        norm_df.index = norm_df.index.to_datetime()
        fig, ax = data.plot_heatmap_in_period(norm_df, t_start=t_start, t_end=t_end, cbar=(0,4),
                                              option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
                                              save_df=False, figsize=(18,8))

        # norm_df = pd.read_csv(save_dir+'s1_2d_KF__{0}__{1}_prob95_all_meas.csv'.format(time2str_file(t_start), time2str_file(t_end)),
        #                       index_col=0)
        # norm_df.index = norm_df.index.to_datetime()
        # fig, ax = data.plot_heatmap_in_period(norm_df, t_start=t_start, t_end=t_end, cbar=(0,4),
        #                                       option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
        #                                       save_df=False, figsize=(18,8))

        if False:
            # overlay the ground truth on the data heatmap

            # read the manual labeled ground truth
            idx = []
            times = []
            lane_idx = []
            with open('../datasets/{0}/v11/'.format(folder) + 'ground_truth_10min.txt','r') as f:
                for line in f:
                    if len(line) == 0 or line[0] == '#':
                        continue
                    items = line.strip().split(',')

                    ms = items[0].split(':')
                    sec = int(ms[0])*60+float(ms[1])
                    idx.append( int( sec * 64)-200 )
                    times.append( t_start + timedelta(seconds=sec ) )
                    lane_idx.append(int(items[1]))

            for t_idx, l_idx in zip(idx, lane_idx):
                # ax.fill_between(norm_df.index, 0, 127, where=((norm_df.index>=veh[0]) & (norm_df.index<=veh[1])), facecolor='green',alpha=0.5  )

                if l_idx == 1:
                    col = 'g'
                elif l_idx == 2:
                    col = 'r'
                rect = patches.Rectangle((t_idx, 0), 16, 128, linewidth=1, edgecolor=col,
                                         facecolor=(0,1,0,0.2))
                ax.add_patch(rect)


            # plot ultrasonic sensor
            fig, ax = plt.subplots(figsize=(18,5))
            t_ultra = df.index
            # mtimes = [mdates.date2num(t) for t in times]
            ultra = df['ultra']
            ax.plot(t_ultra, ultra)

            for t, l_idx in zip(times, lane_idx):
                if l_idx == 1:
                    col = 'g'
                    f_col = (0,1,0,0.2)
                elif l_idx == 2:
                    col = 'r'
                    f_col = (1,0,0,0.2)

                # rect = patches.Rectangle((t, 0.0), 0.2, 12,
                #                          linewidth=1, edgecolor=col, facecolor=(0,1,0,0.2))
                # ax.add_patch(rect)

                ax.axvspan(t, t+ pd.datetools.Second(1), facecolor=f_col, edgecolor=col)

            # assign date locator / formatter to the x-axis to get proper labels
            # locator = mdates.AutoDateLocator(minticks=3)
            # formatter = mdates.AutoDateFormatter(locator)
            # ax.xaxis.set_major_locator(locator)
            # ax.xaxis.set_major_formatter(formatter)

        plt.draw()


def test_alg():

    # ===============================================================================================
    # data set 0530, S1: 20:55 ~ 21:45;
    # Two sensors on both sides of road at the same location. Freeflow on Neil in Savoy
    # ===============================================================================================
    if False:
        folder = '0530_2017'
        speed_range = (-71,-1)
        save_dir = '../workspace/{0}/'.format(folder)

        # read in the data with background subtracted.
        norm_df = pd.read_csv(save_dir +
                              's1_2d_KF__20170530_205400_0__20170530_214500_0_prob95.csv', index_col=0)
        norm_df.index = norm_df.index.to_datetime()

        # Plot the norm_df
        # data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
        # fig, ax = data.plot_heatmap_in_period(norm_df, t_start=str2time('2017-05-30 21:21:0.0'),
        #                                       t_end=str2time('2017-05-30 21:21:10.0'),
        #                                       cbar=(0,4), option='vec', nan_thres_p=None, plot=True,
        #                                       save_dir=save_dir, save_img=False, save_df=False, figsize=(18,8))
        #
        alg = TrafficSensorAlg(r2_thres=0.2, dens_thres=0.4, min_dt=0.2, pir_res=(4,32), sampling_freq=64)

        # To plot overlapping cluster splitting
        # str2time('2017-05-30 21:21:02.0')    to   str2time('2017-05-30 21:21:10.0')
        # To plot different width and robust regression
        # str2time('2017-05-30 20:55:0.0')     to   str2time('2017-05-30 20:55:10.0')
        # alg.run_adaptive_window(norm_df, TH_det=600, window_s=5.0, step_s=2.5, speed_range=speed_range, plot_final=True,
        #                         plot_debug=True, save_dir='../workspace/{0}/figs/speed/s1/'.format(folder),
        #                         t_start=str2time('2017-05-30 21:21:02.0'),
        #                         t_end=str2time('2017-05-30 21:21:10.0'))
        alg.run_adaptive_window(norm_df, TH_det=600, window_s=5.0, step_s=2.5, speed_range=speed_range, plot_final=True,
                                plot_debug=False, save_dir='../workspace/{0}/figs/speed/s1/'.format(folder),
                                t_start=str2time('2017-05-30 20:55:0.0'),
                                t_end=str2time('2017-05-30 21:45:00.0'))

    # ===============================================================================================
    # data set 0530, S2: 20:55 ~ 21:45
    # ===============================================================================================
    if True:
        folder = '0530_2017'
        sensor = 's2'
        speed_range = (1,71)
        save_dir = '../workspace/{0}/'.format(folder)

        # read in the data with background subtracted.
        norm_df = pd.read_csv(save_dir +
                              's2_2d_KF__20170530_205400_0__20170530_214500_0_prob95.csv', index_col=0)
        norm_df.index = norm_df.index.to_datetime()

        # Plot the norm_df
        # data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
        # fig, ax = data.plot_heatmap_in_period(norm_df, t_start=None, t_end=None, cbar=(0,4),
        #                                           option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
        #                                           save_df=False, figsize=(18,8))

        alg = TrafficSensorAlg(r2_thres=0.2, dens_thres=0.4, min_dt=0.2, pir_res=(4,32), sampling_freq=64)

        alg.run_adaptive_window(norm_df, TH_det=600, window_s=5.0, step_s=2.5, speed_range=speed_range, plot_final=True,
                                plot_debug=False, save_dir='../workspace/{0}/figs/speed/s2/'.format(folder),
                                t_start=str2time('2017-05-30 20:55:00.0'),
                                t_end=str2time('2017-05-30 21:45:00.0'))

    # ===============================================================================================
    # dataset Jun 08, 2017 on 6th street down near the parking lot.
    # Saved data is 21:40 ~ 22:20 UTC
    # ===============================================================================================
    if False:
        folder = 'Jun08_2017'
        speed_range = (-71,-1)
        save_dir = '../workspace/{0}/'.format(folder)

        # read in the data with background subtracted.
        norm_df = pd.read_csv(save_dir +
                              's1_2d_KF__20170608_213900_001464__20170608_222037_738293_prob95.csv', index_col=0)
        norm_df.index = norm_df.index.to_datetime()

        # Plot the norm_df
        # data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
        # fig, ax = data.plot_heatmap_in_period(norm_df, t_start=None, t_end=None, cbar=(0,4),
        #                                           option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
        #                                           save_df=False, figsize=(18,8))

        alg = TrafficSensorAlg(r2_thres=0.2, dens_thres=0.4, min_dt=0.2, pir_res=(4,32), sampling_freq=64)

        alg.run_adaptive_window(norm_df, TH_det=600, window_s=5.0, step_s=2.5, speed_range=speed_range, plot_final=False,
                                plot_debug=False, save_dir='../workspace/{0}/figs/speed/'.format(folder),
                                t_start=str2time('2017-06-08 21:40:00.0'), t_end=str2time('2017-06-08 22:20:00.0'))


    # ===============================================================================================
    # dataset Jun 09, 2017, same setup as Jun 08: 19:10~20:39
    # ===============================================================================================
    if False:
        folder = 'Jun09_2017'
        speed_range = (-71,-1)

        save_dir = '../workspace/{0}/'.format(folder)

        # read in the data with background subtracted.
        norm_df = pd.read_csv(save_dir +
                              's1_2d_KF__20170609_190900_009011__20170609_203930_905936_prob95.csv', index_col=0)
        norm_df.index = norm_df.index.to_datetime()

        # Plot the norm_df
        # data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
        # fig, ax = data.plot_heatmap_in_period(norm_df, t_start=None, t_end=None, cbar=(0,4),
        #                                           option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
        #                                           save_df=False, figsize=(18,8))

        alg = TrafficSensorAlg(r2_thres=0.2, dens_thres=0.4, min_dt=0.2, pir_res=(4,32), sampling_freq=64)

        alg.run_adaptive_window(norm_df, TH_det=600, window_s=5.0, step_s=2.5, speed_range=speed_range, plot_final=True,
                                plot_debug=False, save_dir='../workspace/{0}/figs/speed/'.format(folder),
                                t_start=None, t_end=None)


def trim_results():
    """
    This function trims the same time period out from the datasets for evaluation.
    For detection result:
        - Add a column
    :return:
    """
    # ===============================================================================================
    # data set Jun08, 2017, S1: trim to 21:40~22:20
    # ===============================================================================================
    if False:
        folder = 'Jun08_2017'
        save_dir = '../workspace/{0}/'.format(folder)

        t_period_start = str2time('2017-06-08 21:40:00.0')
        t_period_end = str2time('2017-06-08 22:20:00.0')

        vehs_npy = save_dir + 'figs/speed/v3_final/detected_vehs.npy'

        # Ground trueth
        init_t = str2time('2017-06-08 21:32:18.252811')
        offset = -0.28
        drift_ratio = 1860.5/1864.0
        true_file = save_dir + 'labels_Jun08.npy'

        # Trimming
        eval = EvaluateResult()

        # trim the norm_df
        if False:
            norm_df_file = save_dir + 's1_2d_KF__20170608_213900_001464__20170608_222037_738293_prob95.csv'
            eval.post_trim_norm_df(norm_df_file, t_period_start-timedelta(seconds=60), t_period_end)

        if True:
            # paras_file = save_dir + 'figs/speed/v3_final/paras.txt'
            # norm_df_file = save_dir + 's1_2d_KF__20170608_213900_001464__20170608_222037_738293_prob95.csv'
            # eval.post_clean_ultra_norm_df(norm_df_file, paras_file)

            eval.post_trim_detection(vehs_npy, t_period_start, t_period_end,
                                 ultra_fp_lb=4.0, ultra_fp_ub=8.0, speed_range=(0.0, 30.0))

        if False:
            eval.post_trim_true_detection(true_file, init_t, offset, drift_ratio, t_period_start, t_period_end)


    # ===============================================================================================
    # data set Jun09, 2017, S1, trim to 19:10~20:39
    # ===============================================================================================
    if True:
        folder = 'Jun09_2017'
        save_dir = '../workspace/{0}/'.format(folder)

        t_period_start = str2time('2017-06-09 19:10:00.0')
        t_period_end = str2time('2017-06-09 20:39:00.0')

        vehs_npy = save_dir + 'figs/speed/detected_vehs.npy'

        # Ground truth
        init_t = str2time('2017-06-09 19:09:00.0')
        offset = -0.66
        drift_ratio = 1860.5/1864.0
        true_file = save_dir + 'labels_Jun09.npy'

        # Trimming
        eval = EvaluateResult()

        # trim the norm_df
        if False:
            norm_df_file = save_dir + 's1_2d_KF__20170609_190900_009011__20170609_203930_905936_prob95.csv'
            eval.post_trim_norm_df(norm_df_file, t_period_start-timedelta(seconds=60), t_period_end)

        if True:
            # paras_file = save_dir + 'figs/speed/paras.txt'
            # norm_df_file = save_dir + 's1_2d_KF__20170609_190900_009011__20170609_203930_905936_prob95.csv'
            # eval.post_clean_ultra_norm_df(norm_df_file, paras_file)

            eval.post_trim_detection(vehs_npy, t_period_start, t_period_end,
                                 ultra_fp_lb=4.0, ultra_fp_ub=8.0, speed_range=(0.0, 30.0))

        if False:
            eval.post_trim_true_detection(true_file, init_t, offset, drift_ratio, t_period_start, t_period_end)


    # ===============================================================================================
    # data set 0530, 2017, S1: 2017-05-30 20:55:00.0 ~ 2017-05-30 21:45:00.0
    # ===============================================================================================
    if False:
        folder = '0530_2017'
        t_period_start = str2time('2017-05-30 20:55:00.0')
        t_period_end = str2time('2017-05-30 21:45:00.0')

        # Configuration
        save_dir = '../workspace/{0}/'.format(folder)

        # Detections
        paras_file = save_dir + 'figs/speed/s1/paras.txt'
        norm_df_file = save_dir + 's1_2d_KF__20170530_205400_0__20170530_214500_0_prob95.csv'
        vehs_npy = save_dir + 'figs/speed/s1/detected_vehs.npy'

        # Ground truth
        # 30/05: 2017-05-30 20:55:00 ~ 2017-05-30 21:55:00
        init_t = str2time('2017-05-30 20:55:00.0')
        offset = -0.48
        drift_ratio = 1860.6/1864.0
        true_file = save_dir + 'labels_v11.npy'

        # Trimming
        eval = EvaluateResult()

        # trim the norm_df
        if False:
            eval.post_trim_norm_df(norm_df_file, t_period_start-timedelta(seconds=60), t_period_end)

        if False:
            # eval.post_clean_ultra_norm_df(norm_df_file, paras_file)

            eval.post_trim_detection(vehs_npy, t_period_start, t_period_end,
                                 ultra_fp_lb=None, ultra_fp_ub=8.0, speed_range=(0.0, 70.0))

        if True:
            eval.post_trim_true_detection(true_file, init_t, offset, drift_ratio, t_period_start, t_period_end)

    # ===============================================================================================
    # data set 0530, 2017, S2: 2017-05-30 20:55:00.0 ~ 2017-05-30 21:45:00.0
    # ===============================================================================================
    if False:
        folder = '0530_2017'
        t_period_start = str2time('2017-05-30 20:55:00.0')
        t_period_end = str2time('2017-05-30 21:45:00.0')

        # Configuration
        save_dir = '../workspace/{0}/'.format(folder)

        # Detection
        paras_file = save_dir + 'figs/speed/s2/paras.txt'
        norm_df_file = save_dir + 's2_2d_KF__20170530_205400_0__20170530_214500_0_prob95.csv'
        vehs_npy = save_dir + 'figs/speed/s2/detected_vehs.npy'

        # Ground truth
        # 30/05: 2017-05-30 20:55:00 ~ 2017-05-30 21:55:00
        init_t = str2time('2017-05-30 20:55:00.0')
        offset = 2.102
        drift_ratio = 1860.6/1864.0
        true_file = save_dir + 'labels_v21_post.npy'

        # Trimming
        eval = EvaluateResult()

        # trim the norm_df
        if False:
            eval.post_trim_norm_df(norm_df_file, t_period_start-timedelta(seconds=60), t_period_end)

        if False:
            # eval.post_clean_ultra_norm_df(norm_df_file, paras_file)
            eval.post_trim_detection(vehs_npy, t_period_start, t_period_end,
                                 ultra_fp_lb=None, ultra_fp_ub=8.0, speed_range=(0.0, 70.0))

        if True:
            eval.post_trim_true_detection(true_file, init_t, offset, drift_ratio, t_period_start, t_period_end)


def evaluation():
    """
    This function plot the detection results. MAKE SURE the run trim_results first.
    :return:
    """
    # ===============================================================================================
    # data set 0530, 2017: 2017-05-30 20:55:00.0 ~ 2017-05-30 21:45:00.0
    # S1 and S2
    # -----------------------------------------------------------------------------------------------
    # Combine detections
    if False:
        folder = '0530_2017'
        save_dir = '../workspace/{0}/'.format(folder)

        s1_vehs_npy = save_dir + 'figs/speed/s1/v2_3/detected_vehs_post.npy'
        s2_vehs_npy = save_dir + 'figs/speed/s2/v2_3/detected_vehs_post.npy'

        # combine detection
        s1_s2_shift = 0.33
        ev = EvaluateResult()
        # ev.combine_detections(s1_vehs_npy, s2_vehs_npy, s1_s2_shift=s1_s2_shift, dt=0.5)
        ev.combine_two_lane_detections(s1_vehs_npy, s2_vehs_npy, s1_s2_shift=s1_s2_shift, dt=0.5, save_dir=save_dir,
                                       speed_range=(0.0,70.0))

    # -----------------------------------------------------------------------------------------------
    # S1 and S2, plot combined detection vs true
    if False:
        folder = '0530_2017'
        save_dir = '../workspace/{0}/'.format(folder)
        paras_file = save_dir + 'figs/speed/s1/v2_3/paras.txt'

        s1_true = np.load(save_dir + 'labels_v11_post.npy')
        s2_true = np.load(save_dir + 'labels_v21_post.npy')
        s2_true[:,2] = -s2_true[:,2]

        s1_vehs_comb = np.load(save_dir + 'figs/speed/s1/v2_3/detected_vehs_post_comb_v2.npy')
        s2_vehs_comb = np.load(save_dir + 'figs/speed/s2/v2_3/detected_vehs_post_comb_v2.npy')

        unique_vehs = np.load(save_dir + 'unique_detected_vehs.npy')

        ev = EvaluateResult()
        if False:
            s1_norm_df = pd.read_csv(save_dir + 's1_2d_KF__20170530_205400_0__20170530_214500_0_prob95_clean_ultra.csv', index_col=0)
            s1_norm_df.index = s1_norm_df.index.to_datetime()
            s2_norm_df = pd.read_csv(save_dir + 's2_2d_KF__20170530_205400_0__20170530_214500_0_prob95_clean_ultra.csv', index_col=0)
            s2_norm_df.index = s2_norm_df.index.to_datetime()
            s1_s2_shift = 0.33
            ev.plot_two_lane_vs_true(s1_df=s1_norm_df, s1_vehs=s1_vehs_comb, s1_true=s1_true,
                                     s2_df=s2_norm_df, s2_vehs=s2_vehs_comb, s2_true=s2_true,
                                     s1s2_shift=s1_s2_shift, s1_paras_file=paras_file,
                                     t_start=str2time('2017-05-30 21:29:00.0'),
                                     t_end=str2time('2017-05-30 21:44:00.0'),
                                     unique_vehs=unique_vehs)

            # ev.plot_two_lane_vs_true(s1_df=s1_norm_df, s1_vehs=s1_vehs_comb, s1_true=s1_true,
            #                          s2_df=s2_norm_df, s2_vehs=s2_vehs_comb, s2_true=s2_true,
            #                          s1s2_shift=s1_s2_shift, s1_paras_file=paras_file,
            #                          t_start=str2time('2017-05-30 21:20:00.0'),
            #                          t_end=str2time('2017-05-30 21:25:00.0'),
            #                          unique_vehs=unique_vehs)
            #
            # ev.plot_two_lane_vs_true(s1_df=s1_norm_df, s1_vehs=s1_vehs_comb, s1_true=s1_true,
            #                          s2_df=s2_norm_df, s2_vehs=s2_vehs_comb, s2_true=s2_true,
            #                          s1s2_shift=s1_s2_shift, s1_paras_file=paras_file,
            #                          t_start=str2time('2017-05-30 21:40:00.0'),
            #                          t_end=str2time('2017-05-30 21:45:00.0'),
            #                          unique_vehs=unique_vehs)

        # Match vehicle and compute statistics
        if True:
            matched_vehs = ev.match_two_lane_det_with_true(s1_vehs_comb, s2_vehs_comb,
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
            one_min_err = ev.compute_aggregated_error(matched_vehs[tp_idx,:], agg_s=60)
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
        folder = '0530_2017'
        save_dir = '../workspace/{0}/'.format(folder)

        s1_true = np.load(save_dir + 'labels_v11_post.npy')
        s2_true = np.load(save_dir + 'labels_v21_post.npy')

        s1_vehs_npy_comb = save_dir + 'figs/speed/s1/detected_vehs_post_comb.npy'
        s2_vehs_npy_comb = save_dir + 'figs/speed/s2/detected_vehs_post_comb.npy'

        true_count_l1 = len(s1_true)
        true_count_l2 = len(s2_true)

        print('True vehicles: Lane 1,  Lane 2,  Total')
        print('               {0},     {1},     {2}'.format(true_count_l1, true_count_l2,
                                                            true_count_l1+true_count_l2))
        # vehicle during the true period
        s1_vehs_comb = np.load(s1_vehs_npy_comb)
        s2_vehs_comb = np.load(s2_vehs_npy_comb)
        count_l1 = count_vehs(s1_vehs_comb)
        count_l2 = count_vehs(s2_vehs_comb)

        print('Estimated    : Lane 1,  Lane 2,  Total')
        print('               {0},     {1},     {2}'.format(count_l1, count_l2, count_l1 + count_l2))

        # plot the distribution
        plot_hist(s1_vehs_comb, s2_vehs_comb, s1_true, s2_true)

    # -----------------------------------------------------------------------------------------------
    # S1: plot det vs true
    if False:
        folder = '0530_2017'

        # Configuration
        save_dir = '../workspace/{0}/'.format(folder)
        paras_file = save_dir + 'figs/speed/s1/v2_3/paras.txt'
        norm_df_file = save_dir + 's1_2d_KF__20170530_205400_0__20170530_214500_0_prob95_clean_ultra.csv'
        est_vehs_npy = save_dir + 'figs/speed/s1/v2_3/detected_vehs_post.npy'
        true_file = save_dir + 'labels_v11_post.npy'

        # Plot the detection VS true
        ev = EvaluateResult()

        # load norm df data
        norm_df = pd.read_csv(norm_df_file, index_col=0)
        norm_df.index = norm_df.index.to_datetime()
        true_vehs = np.load(true_file)
        det_vehs = np.load(est_vehs_npy)

        # ev.plot_det_vs_true(norm_df,det_vehs, paras_file, true_vehs=true_vehs,
        #                       t_start=str2time('2017-05-30 20:55:00.0'),
        #                       t_end=str2time('2017-05-30 21:05:00.0'))
        ev.plot_det_vs_true(norm_df, paras_file, vehs=None, true_vehs=None,
                              t_start=str2time('2017-05-30 21:30:00.0'),
                              t_end=str2time('2017-05-30 21:35:00.0'))

    # -----------------------------------------------------------------------------------------------
    # S2: plot det vs true
    if False:
        folder = '0530_2017'

        # Configuration
        save_dir = '../workspace/{0}/'.format(folder)
        paras_file = save_dir + 'figs/speed/s2/paras.txt'
        norm_df_file = save_dir + 's2_2d_KF__20170530_205500_0__20170530_214500_0_prob95.csv'
        est_vehs_npy = save_dir + 'figs/speed/s2/detected_vehs_post.npy'
        true_file = save_dir + 'labels_v21_post.npy'

        # Plot the detection VS true
        ev = EvaluateResult()

        # load norm df data
        norm_df = pd.read_csv(norm_df_file, index_col=0)
        norm_df.index = norm_df.index.to_datetime()
        true_vehs = np.load(true_file)
        true_vehs[:,2] = -true_vehs[:,2]
        det_vehs = np.load(est_vehs_npy)

        ev.plot_det_vs_true(norm_df,det_vehs, paras_file, true_vehs=true_vehs,
                              t_start=str2time('2017-05-30 21:35:00.0'),
                              t_end=str2time('2017-05-30 21:50:00.0'))


    # ===============================================================================================
    # dataset Jun 08, 2017 on 6th street down near the parking lot.
    # Saved data is 21:40 ~ 22:20 UTC
    if False:
        folder = 'Jun08_2017'

        # Configuration
        save_dir = '../workspace/{0}/'.format(folder)
        est_vehs_npy = save_dir + 'figs/speed/v3_1/detected_vehs_post.npy'
        true_file = save_dir + 'labels_Jun08_post.npy'

        true_vehs = np.load(true_file)
        det_vehs = np.load(est_vehs_npy)

        # find the outliers
        # for v in det_vehs:
        #     if abs(v['distance']) <= 4:
        #         print('outlier vehicles {0:.2f} m: {1}, {2}'.format(v['distance'], v['t_in'], v['valid']))

        ev = EvaluateResult()
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
            matched_vehs = ev.match_one_lane_det_with_true(det_vehs, true_vehs, dt=0.5)
            matched_vehs = np.asarray(matched_vehs)

            ev.compute_statistics(matched_vehs)

            # plot per vehicle error
            tp_idx = ~pd.isnull(matched_vehs[:,0]) & ~pd.isnull(matched_vehs[:,1])
            ev.plot_hist([matched_vehs[tp_idx,2]-matched_vehs[tp_idx,3]], None,
                         title='Test 1 per vehicle speed error', xlabel='Speed (mph)',
                         fontsizes=(34, 30, 28), xlim=(-7, 7), ylim=(0, 0.5), text_loc=(3, 0.42))

            # plot one minute error
            one_min_err = ev.compute_aggregated_error(matched_vehs[tp_idx,:], agg_s=60)
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
            paras_file = save_dir + 'figs/speed/v3_final/paras.txt'
            norm_df_file = save_dir + 's1_2d_KF__20170608_213900_001464__20170608_222037_738293_prob95_clean_ultra.csv'
            norm_df = pd.read_csv(norm_df_file, index_col=0)
            norm_df.index = norm_df.index.to_datetime()

            # str2time('2017-06-08 21:40:00.0')
            # str2time('2017-06-08 22:20:00.0')
            ev.plot_det_vs_true(norm_df,det_vehs, paras_file, true_vehs=true_vehs,
                              t_start=str2time('2017-06-08 22:05:00.0'),
                              t_end=str2time('2017-06-08 22:15:00.0'), matches=matched_vehs)


    # ===============================================================================================
    # dataset Jun 09, 2017, same setup as Jun 08, 19:10~20:39
    if True:
        folder = 'Jun09_2017'

        # Configuration
        save_dir = '../workspace/{0}/'.format(folder)
        est_vehs_npy = save_dir + 'figs/speed/v3_2/detected_vehs_post.npy'
        true_file = save_dir + 'labels_Jun09_post.npy'

        true_vehs = np.load(true_file)
        det_vehs = np.load(est_vehs_npy)

        ev = EvaluateResult()
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
            matched_vehs = ev.match_one_lane_det_with_true(det_vehs, true_vehs, dt=1.5)
            matched_vehs = np.array(matched_vehs)

            if True:
                ev.compute_statistics(matched_vehs)

                # plot speed error distribution
                tp_idx = ~pd.isnull(matched_vehs[:,0]) & ~pd.isnull(matched_vehs[:,1])
                ev.plot_hist([matched_vehs[tp_idx,2]-matched_vehs[tp_idx,3]], labels=None,
                         title='Test 2 per vehicle speed error', xlabel='Speed (mph)',
                         fontsizes=(34, 30, 28), xlim=(-7, 6), ylim=(0, 0.35), text_loc=(2, 0.3))

                # plot one-minute error distribution
                one_min_err = ev.compute_aggregated_error(matched_vehs[tp_idx,:], agg_s=60)
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
            # str2time('2017-06-09 19:10:00.0') ~ str2time('2017-06-09 20:39:00.0')

            paras_file = save_dir + 'figs/speed/v3_1/paras.txt'
            norm_df_file = save_dir + 's1_2d_KF__20170609_190900_009011__20170609_203930_905936_prob95_clean_ultra.csv'
            norm_df = pd.read_csv(norm_df_file, index_col=0)
            norm_df.index = norm_df.index.to_datetime()

            ev.plot_det_vs_true(norm_df,det_vehs, paras_file, true_vehs=true_vehs,
                              t_start=str2time('2017-06-09 19:25:00.0'),
                              t_end=str2time('2017-06-09 19:35:00.0'), matches=matched_vehs)


def plot_hist(s1_det, s2_det, true_s1, true_s2):
    """
    This function plots the speed histogram of detections
    :param s1_det: s1 detection, a list of vehicle dict
    :param s2_det: s2 detection
    :param true_s1: true for s1: t_start (dt), t_end (dt), speed (mph), distance (m), ...
    :param true_s2: true for s2
    :return:
    """

    # s1 distance and speed
    s1_distances = []
    s1_speeds = []
    for v in s1_det:
        if v['closer_lane'] is True:
            s1_distances.append(v['distance'])
            s1_speeds.append(abs(v['speed']))

    # s2 distance and speeds
    s2_distances = []
    s2_speeds = []
    for v in s2_det:
        if v['closer_lane'] is True:
            s2_distances.append(v['distance'])
            s2_speeds.append(abs(v['speed']))

    # true s1 speed and distance
    s1_true_distances = []
    s1_true_speeds = []
    for v in true_s1:
        if not np.isnan(v[2]):
            s1_true_distances.append(v[3])
            s1_true_speeds.append(abs(v[2]))


    # true s2 speed and distance
    s2_true_distances = []
    s2_true_speeds = []
    for v in true_s2:
        if not np.isnan(v[2]):
            s2_true_distances.append(v[3])
            s2_true_speeds.append(abs(v[2]))

    # ------------------------------------------------------------------------------
    # speed distribution
    plt.figure(figsize=(10,10))
    speeds = [s1_speeds, s2_speeds, s1_true_speeds, s2_true_speeds]
    labels = ['s1', 's2', 's1_true', 's2_true']
    colors = itertools.cycle( ['b', 'g', 'm', 'c', 'purple', 'r'])
    text = []
    for i, s in enumerate(speeds):
        c = next(colors)
        mu, std = np.mean(s), np.std(s)
        print('Speed {0}: {1}, {2}'.format(labels[i], mu, std))
        n, bins, patches = plt.hist(s, 50, normed=1, facecolor=c, alpha=0.75, label=labels[i])
        plt.plot(bins, mlab.normpdf(bins, mu, std), color=c, linestyle='--',
                 linewidth=2)

        text.append('{0} mean: {1:.2f}, std: {2:.2f}'.format(labels[i], mu, std))
        # text.append('Mean: {0:.2f}\nStandard deviation: {1:.2f}'.format(mu, std))

    text_str = '\n'.join(text)
    plt.text(5, 0.10, text_str, fontsize=14)
    # plt.text(6, 0.16, text_str, fontsize=16)
    plt.ylim(0, np.max(n)*1.2)

    plt.legend()
    plt.xlabel('Speed (mph)', fontsize=18)
    plt.ylabel('Distribution', fontsize=18)
    plt.title('Two lane speed distribution', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.draw()

    # ------------------------------------------------------------------------------
    # distance distribution
    plt.figure(figsize=(10,10))
    distances = [s1_distances, s2_distances, s1_true_distances, s2_true_distances]
    labels = ['s1', 's2', 's1_true', 's2_true']
    colors = itertools.cycle( ['b', 'g', 'm', 'c', 'purple', 'r'])
    text = []
    for i, d in enumerate(distances):
        c = next(colors)
        mu, std = np.mean(d), np.std(d)
        print('Distance {0}: {1}, {2}'.format(labels[i], mu, std))
        n, bins, patches = plt.hist(d, 50, normed=1, facecolor=c, alpha=0.75, label=labels[i])
        plt.plot(bins, mlab.normpdf(bins, mu, std), color=c, linestyle='--',
                 linewidth=2)

        text.append('{0} mean: {1:.2f}, std: {2:.2f}'.format(labels[i], mu, std))
        # text.append('Mean: {0:.2f}\nStandard deviation: {1:.2f}'.format(mu, std))

    text_str = '\n'.join(text)
    plt.text(1, 1.5, text_str, fontsize=14)
    # plt.text(6, 0.16, text_str, fontsize=16)
    plt.ylim(0, np.max(n)*1.2)
    plt.xlim([1,8])

    plt.legend()
    plt.xlabel('Distance (m)', fontsize=18)
    plt.ylabel('Distribution', fontsize=18)
    plt.title('Two lane Distance distribution', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.draw()


def count_vehs(vehs):
    counter = 0
    for v in vehs:
        if v['closer_lane'] is True:
            counter +=1

    return counter



def generate_video_frames():
    # ===============================================================================================
    # generate a video
    folder = '1013_2016'
    data_dir = '../datasets/{0}/'.format(folder)
    dataset = '1310-210300'     # freeflow part 1
    # dataset = '1310-213154'   # freeflow part 2
    # dataset ='1310-221543'       # stop and go

    # ========================================================
    # Load the raw PIR data
    print('Loading raw data...')
    t1 = datetime.now()
    raw_df = pd.read_csv(data_dir+'{0}.csv'.format(dataset), index_col=0)
    raw_df.index = raw_df.index.to_datetime()
    t2 = datetime.now()
    print('Loading raw data csv data took {0} s'.format((t2-t1).total_seconds()))

    # ========================================================
    # Load the normalized data file
    print('Loading normalized data...')
    norm_df = pd.read_csv('../workspace/1013/s1_2d_MAP__20161013_210305_738478__20161013_211516_183544_prob95.csv', index_col=0)
    norm_df.index = norm_df.index.to_datetime()

    # ========================================================
    # Load the detected vehicles
    print('Loading detected vehicles...')
    det_vehs = np.load('../workspace/1013/s1_vehs__20161013_210305_738478__20161013_211516_183544_prob95.npy')

    # ========================================================
    # Construct algorithm instance
    print('Generating video frames...')
    # _dir = '../workspace/1013/Video_ff1/'
    _dir = '/data_fast/Yanning_sensors/video_1013_ff1_rgb/'
    alg = TrafficSensorAlg(pir_res=(4,32))
    alg.plot_video_frames(video_file='../datasets/1013_2016/1013_v1_1_ff_hq.mp4', video_fps=60.1134421399,
                          video_start_time=str2time('2016-10-13 20:57:27.5'),
                          raw_df=raw_df, raw_pir_clim=(20,40),
                          ratio_tx=6.0, norm_df=norm_df, norm_df_win=5.0, det_vehs=det_vehs,
                          save_dir=_dir)


def update_fps(offset1, offset2, T, fps):
    """
    This function updates the fps based on the offsets
    :param offset1: the offset in seconds regarding to the real time
    :param offset2: the increased or decreased offset in seconds regarding to the real time after T
    :param T: the duration for the offset in seconds
    :param fps: old fps
    :return: new fps
    """
    return fps*(T-offset1+offset2)/T

if __name__ == '__main__':
    main()
