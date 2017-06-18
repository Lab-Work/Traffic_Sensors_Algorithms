from TrafficSensorAlg_V2 import *

def main():

    # data_analysis()
    # generate_1d_pir_signal()
    # sync_data()

    # trim_video()
    # video_analysis()
    # generate_video_frames()
    # trim_out_dense_traffic_videos()

    # veh_det()
    # test_alg()

    evaluate_results()

    plt.show()

def data_analysis():
    # --------------------------------------------------
    # data set 0530, 2017
    # Two sensors on both sides of road at the same location.
    # Freeflow on Neil in Savoy
    # data set is good to use
    folder = '0530_2017'
    data_dir = '../datasets/{0}/s1/'.format(folder)
    dataset = '3005-205500_s1'
    # data_dir = '../datasets/{0}/s2/'.format(folder)
    # dataset = '3005-203625_s2'

    # --------------------------------------------------
    # dataset Jun 08, 2017 on 6th street down near the parking lot.
    # Saved data is 21:40 ~ 22:20 UTC
    # folder = 'Jun08_2017'
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
        df = data.load_txt_data(data_dir+'{0}.txt'.format(dataset), flip=False)
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
        t_start = periods[dataset][0] #+ timedelta(seconds=180)
        t_end = periods[dataset][1] # + timedelta(seconds=780)
        data.plot_heatmap_in_period(df, t_start=t_start, t_end=t_end, cbar=(40,60), option='vec',
                                    nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False, save_df=False,
                                    figsize=(18,8))

        # plot ultrasonic sensor
        plt.figure(figsize=(15,8))
        plt.plot(df.index, df['ultra'])

        plt.draw()

    # ===============================================================================================
    # plot the evolution of the noise
    if False:
        t_start = periods[dataset][0] + timedelta(seconds=1770)
        t_end = periods[dataset][0]  + timedelta(seconds=2070)
        data.plot_noise_evolution(df, t_start=t_start, t_end=t_end, p_outlier=0.01, stop_thres=(0.01,0.1),
                                  pixel=(2,20), window_s=10, step_s=3)

        data.plot_histogram_for_pixel(df, pixels=[(2,20)], t_start=t_start, t_end=t_end, p_outlier=0.01,
                                      stop_thres=(0.01, 0.1))

    # ===============================================================================================
    # analyze and debug KF background subtraction
    if True:
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

        kf_norm_df.to_csv(save_dir + 's1_2d_KF__{0}__{1}_prob95.csv'.format(time2str_file(t_start),
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
        t_end = periods[dataset][0]  # + timedelta(seconds=400)
        norm_df = pd.read_csv(save_dir+'s2_2d_KF__{0}__{1}_prob95.csv'.format(time2str_file(t_start), time2str_file(t_end)),
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

    # --------------------------------------------------
    # data set 0530, 2017
    # Two sensors on both sides of road at the same location.
    # Freeflow on Neil in Savoy
    # data set is good to use
    # folder = '0530_2017'
    # data_dir = '../datasets/{0}/s1/'.format(folder)
    # dataset = '3005-205500_s1'
    # data_dir = '../datasets/{0}/s2/'.format(folder)
    # dataset = '3005-203625_s2'

    # --------------------------------------------------
    # dataset Jun 08, 2017 on 6th street down near the parking lot.
    # Saved data is 21:40 ~ 22:20 UTC
    folder = 'Jun08_2017'
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

    # read in the data with background subtracted.
    norm_df = pd.read_csv(save_dir + 's1_2d_KF__20170608_213900_001464__20170608_222037_738293_prob95.csv', index_col=0)
    norm_df.index = norm_df.index.to_datetime()

    # plot the data.
    # data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
    # fig, ax = data.plot_heatmap_in_period(norm_df, t_start=None, t_end=None, cbar=(0,4),
    #                                           option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
    #                                           save_df=False, figsize=(18,8))


    # ===============================================================================================
    # run algorithm in batch mode
    alg = TrafficSensorAlg(r2_thres=0.2, dens_thres=0.4, min_dt=0.2, pir_res=(4,32), sampling_freq=64)

    # Define start time and end time to only run the algorithm for this period
    # _ct_s = str2time_file('20161013_221809_310524')
    # _ct_e = str2time_file('20161013_221821_805241')
    # _idx = (norm_df.index >= _ct_s) & (norm_df.index <= _ct_e)
    # _norm_df = norm_df.ix[_idx, :]

    alg.run(norm_df, TH_det=600, window_s=5.0, step_s=2.5, speed_range=(1,50), plot_final=True, plot_debug=False,
            save_dir='../workspace/{0}/figs/speed/'.format(folder))

    # alg.vehs = np.load('../workspace/{0}/figs/speed/detected_vehs.npy'.format(folder))
    # print('Total {0} vehs'.format(len(alg.vehs)))
    # alg.plot_detected_vehs(norm_df, ratio_tx=6.0,
    #                        t_start=str2time('2017-06-08 21:39:00.001464'),
    #                        t_end=str2time('2017-06-08 21:44:00.001464'))

    plt.draw()


def evaluate_results():
    """
    This function visualize the detection results compared to ground truth
    :return:
    """
    # --------------------------------------------------
    # data set 0530, 2017
    # Two sensors on both sides of road at the same location.
    # Freeflow on Neil in Savoy
    # data set is good to use
    # folder = '0530_2017'
    # data_dir = '../datasets/{0}/s1/'.format(folder)
    # dataset = '3005-205500_s1'
    # data_dir = '../datasets/{0}/s2/'.format(folder)
    # dataset = '3005-203625_s2'

    # --------------------------------------------------
    # dataset Jun 08, 2017 on 6th street down near the parking lot.
    # Saved data is 21:40 ~ 22:20 UTC
    if False:
        folder = 'Jun08_2017'

        # Configuration
        save_dir = '../workspace/{0}/'.format(folder)
        norm_df_file = save_dir + 's1_2d_KF__20170608_213900_001464__20170608_222037_738293_prob95.csv'
        t_start = str2time('2017-06-08 21:39:00.001464')
        t_end = str2time('2017-06-08 22:20:37.738293')

        # true vehicles
        init_t = str2time('2017-06-08 21:32:18.252811')
        offset = -0.28
        drift_ratio = 1860.5/1864.0
        true_file = save_dir + 'data_convexlog_v2_cleaned.npy'

        # estimated vehicles
        est_vehs_npy = save_dir + 'figs/speed/v2_5/detected_vehs.npy'
        est_vehs_txt = save_dir + 'figs/speed/v2_5/detected_vehs.txt'
        paras_file = save_dir + 'figs/speed/v2_5/paras.txt'

    # --------------------------------------------------
    # dataset Jun 09, 2017, same setup as Jun 08
    if True:
        folder = 'Jun09_2017'

        # Configuration
        save_dir = '../workspace/{0}/'.format(folder)
        norm_df_file = save_dir + 's1_2d_KF__20170609_190900_009011__20170609_203930_905936_prob95.csv'
        t_start = str2time('2017-06-09 19:09:00.009011')
        t_end = str2time('2017-06-09 20:39:30.905936')

        # true vehicles
        init_t = str2time('2017-06-09 19:09:00.0')
        offset = -0.66
        drift_ratio = 1860.5/1864.0
        true_file = save_dir + 'data_convexlog_v2.npy'

        # estimated vehicles
        est_vehs_npy = save_dir + 'figs/speed/v2_1/detected_vehs.npy'
        est_vehs_txt = save_dir + 'figs/speed/v2_1/detected_vehs.txt'
        paras_file = save_dir + 'figs/speed/v2_1/paras.txt'

    # ===============================================================================================


    # ===============================================================================================
    eval = EvaluateResult()
    eval.load_paras(paras_file)

    # load norm df data
    norm_df = pd.read_csv(norm_df_file, index_col=0)
    norm_df.index = norm_df.index.to_datetime()

    # load true vehicles
    true_vehs = eval.load_true_results(true_file, init_t, offset, drift_ratio)

    # load estimated
    est_vehs = np.load(est_vehs_npy)

    # ===============================================================================================
    # Visualize the matching
    if True:
        eval.visualize_results_all(norm_df, est_vehs, true_vehs=true_vehs, t_start= t_start,t_end=t_end)

        # eval.visualize_results(norm_df, est_vehs, true_vehs=true_vehs,
        #                    t_start=str2time('2017-06-09 19:09:00.009011'),
        #                    t_end=str2time('2017-06-09 19:19:00.009011'))

        plt.draw()

    # ===============================================================================================
    # Compute the statistics




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


def veh_det():

    # --------------------------------------------------
    # dataset 1013:
    # - Neil street. Freeflow and stop-and-to in total 33 min
    # - One PIR sensor array 4x32, at 64 Hz
    # --------------------------------------------------
    # folder = '1013_2016'
    # data_dir = '../datasets/{0}/'.format(folder)
    # dataset = '1310-210300'     # freeflow part 1
    # dataset = '1310-213154'   # freeflow part 2
    # dataset ='1310-221543'       # stop and go


    # --------------------------------------------------
    # dataset 1116:
    # - Neil street. Three PIR sensor arraies
    # - One PIR sensor array 4x32, at 64 Hz
    # --------------------------------------------------
    # folder = '../datasets/1116_2016/s1/'
    # dataset = '1611-061706'

    # folder = '../datasets/1116_2016/s2/'
    # dataset = '0711-211706'

    # folder = '../datasets/1116_2016/s3/'
    # dataset = '1611-151742'

    # --------------------------------------------------
    # dataset 1118:
    # - Neil street. Three PIR sensor arrays, for stop and go data.
    # - One PIR sensor array 4x32, at 64 Hz
    # --------------------------------------------------
    # folder = '../datasets/1118_2016/s1/'
    # dataset = '1611-171706'

    # folder = '../datasets/1118_2016/s2/'
    # dataset = '1811-144926'

    # folder = '../datasets/1118_2016/s3/'
    # dataset = '1611-171707'

    # --------------------------------------------------
    # dataset 0509, 2017
    # one sensor + two validation cameras
    # freeflow on Neil in Savoy
    folder = '0509_2017'
    data_dir = '../datasets/{0}/s1/'.format(folder)
    dataset = '0404-171706'

    # ===============================================================================================
    # Configuration
    save_dir = '../workspace/{0}/'.format(folder)

    norm_df = pd.read_csv(save_dir + 's1_2d_KF__20170509_194724_799628__20170509_220519_672645_prob95.csv', index_col=0)
    norm_df.index = norm_df.index.to_datetime()

    data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
    fig, ax =data.plot_heatmap_in_period(norm_df, t_start=None, t_end=None, cbar=(0,4),
                                              option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
                                              save_df=False, figsize=(18,8))

    # ===============================================================================================
    # vehicle detection
    det = VehDet()
    vehs = det.batch_detect_veh(norm_df, energy_thres=600, window_s=2.0, step_s=1.0)

    # plot detected vehicles
    for veh in vehs:
        # t_start and t_end may not be in the index, hence replace by _t_start >= t_start and _t_end <= t_end
        _t_start_idx = np.where(norm_df.index>=veh[0])[0][0]
        _t_end_idx = np.where(norm_df.index<=veh[1])[0][-1]
        # ax.fill_between(norm_df.index, 0, 127, where=((norm_df.index>=veh[0]) & (norm_df.index<=veh[1])), facecolor='green',alpha=0.5  )
        rect = patches.Rectangle((_t_start_idx, 0), _t_end_idx-_t_start_idx, 128, linewidth=1, edgecolor='g',
                                 facecolor=(0,1,0,0.2))
        ax.add_patch(rect)


    # ===============================================================================================
    # speed estimation
    # save_dir = '../workspace/{0}/figs/'.format(folder)
    # # print('Estimating speed for {0} vehs'.format(len(vehs)))
    # for veh in vehs[200:]:
    #     est = SpeedEst(norm_df.ix[veh[0]:veh[1]], pir_res=(4,32), plot=False, save_dir=save_dir)
    #     est.estimate_speed(stop_tol=(0.002, 0.01), dist=3.5, r2_thres=0.8, min_num_pts=200)
    #
    plt.draw()


def video_analysis():

    # ============================================================================================
    # play videos
    # ============================================================================================
    # vid = VideoData()
    # input_video = '/data/chris/datasets/1118/v3_2/out.mp4'
    # # input_video = '../datasets/1013_2016/1013_2016_freeflow_cam2.mp4'
    # vid.play_video(input_video, 5.0, x_coord=None, y_coord=(0, 220))


    # ============================================================================================
    # crop videos
    # ============================================================================================
    # vid = VideoData()
    # input_video = '/data/chris/datasets/1116/v1_1/out.mp4'
    # output_video = '/data/chris/datasets/1116/v1_1/v1_1_cropped.mp4'
    # #
    # vid.crop_video(input_video=input_video, output_video=output_video, x_coord=None, y_coord=(60,125),
    #                frame_lim=None)


    # ============================================================================================
    # generate the heatmap
    # ============================================================================================
    # vid = VideoData()
    # input_video = '/data/chris/datasets/1116/v1_1/v1_1_cropped.mp4'
    # heatmap = vid.generate_heatmap(input_video, save_npy=None)

    # save the strength signal
    # cam_series = np.sum(heatmap, 0)
    # np.save('/data/chris/datasets/1116/v1_1/v1_1_signal.npy',cam_series)

    # ============================================================================================
    # Summarize the foreground in each frame (the lane area) to create a 1d signal which shows when
    # a vehicle is in presence
    # ============================================================================================
    vid = VideoData()
    videos = {'v1_1':(-1.0, (0, 300)),
              'v1_2':(None, (0, 170)),
              'v2_1':(None, (0, 270)),
              'v2_2':(-2.0, (0, 270)),
              'v3_1':(-2.0, (0, 320)),
              'v3_2':(5.0, (0, 220))}

    v = 'v3_2'
    input_video = '/data/chris/datasets/1118/{0}/out.mp4'.format(v)
    t1 = datetime.now()
    cam_series = vid.generate_1d_signal(input_video, rotation=videos[v][0],
                                        x_coord=None, y_coord=videos[v][1])
    np.save('/data/chris/datasets/1118/{0}/{0}_signal.npy'.format(v),cam_series)
    t2 = datetime.now()
    print('Finished extracting 1d signal {0} s'.format((t2-t1).total_seconds()))

    # ============================================================================================
    # plot the heatmap
    # ============================================================================================
    # vid = VideoData(save_dir='', fps=60, offset=0)
    # t1 = datetime.now()
    # heatmap = np.load('../datasets/1013_2016/1013_2016_freeflow_cam2_heatmap.npy')
    # t2 = datetime.now()
    # print('Loaded heatmap {0} s'.format((t2-t1).total_seconds()))
    # vid.plot_heatmap(heatmap,figsize=(18,8), plot=True,
    #                  save_img='../datasets/1013_2016/1013_2016_freeflow_cam2_heatmap.png',
    #                  title='2016 Oct 13 freeflow cam 2')
    # t3 = datetime.now()
    # print('Plotted heatmap {0} s'.format((t3-t2).total_seconds()))
    #
    # plt.show()


def trim_video():

    folder = 'Jun08_2017'

    input_video = '../datasets/{0}/v11/Jun08.mp4'.format(folder)
    output_video = '../datasets/{0}/v11/20170608_213900_001464__20170608_222037_738293.mp4'.format(folder)
    video_starttime = str2time('2017-06-08 21:32:18.252811')
    intvl=(str2time('2017-06-08 21:39:00.001464'), str2time('2017-06-08 22:20:37.738293'))
    # intvl=(str2time('2017-06-08 21:39:00.001464'), str2time('2017-06-08 21:44:00.001464'))

    vid = VideoData()
    vid.trim_video(input_video=input_video, output_video=output_video,
                   video_starttime=video_starttime, trim_period=intvl)


def trim_out_dense_traffic_videos():

    input_video = '../workspace/1013/Video_ff1/video_1013_ff1.mp4'
    output_video = '../workspace/1013/Video_ff1/video_1013_ff1_trimmed.mp4'
    video_starttime = str2time('2016-10-13 21:03:11.545070')
    trim_starttime = str2time('2016-10-13 21:03:18.60')

    det_vehs = np.load('../workspace/1013/s1_vehs__20161013_210305_738478__20161013_211516_183544_prob95.npy')
    dt = 2.0

    vid = VideoData()
    vid.trim_out_dense_traffic(input_video,video_starttime, output_video, trim_starttime, det_vehs, dt)


def sync_data():

    plot_video = False
    plot_pir = True
    plot_ultra = True
    folder = '0509_2017'
    save_dir = '../workspace/{0}/'.format(folder)

    # ============================================================================================
    # Load PIR and ultrasonic sensor data
    if plot_pir:
        pir1_df = pd.read_csv(save_dir+'s1_1d__20170509_194724_799628__20170509_220519_672645_prob95.csv', index_col=0)
        pir1_df.index = pir1_df.index.to_datetime()
        pir1_t = pir1_df.index
        pir1 = pir1_df['pir']
        ultra1 = pir1_df['ultra']

        # pir2_df = pd.read_csv(save_dir+'s1_1d__20161013_213203_763628__20161013_214035_278451_prob95.csv', index_col=0)
        # pir2_df.index = pir2_df.index.to_datetime()
        # pir2_t = pir2_df.index
        # pir2 = pir2_df['pir']
        # ultra2 = pir2_df['ultra']

    # ============================================================================================
    # Load the video signal
    if plot_video:
        vid = VideoData()

        cam1 = np.load(save_dir+'cam1_1d_ff.npy')*400
        cam2 = np.load(save_dir+'cam2_1d_ff.npy')*400

        # get the timestamps
        fps_1 = 60.1134421399
        fps_2 = 60.1134421399
        inc_1 = 1.0/fps_1
        inc_2 = 1.0/fps_2

        ####
        cam1_t_init = str2time('2016-10-13 20:57:27.5')
        cam2_t_init = str2time('2016-10-13 20:57:27.7')

        ####
        cam1_t = pd.date_range(cam1_t_init, periods=len(cam1),freq=str(int(inc_1*1e9))+'N')
        cam2_t = pd.date_range(cam2_t_init, periods=len(cam2),freq=str(int(inc_2*1e9))+'N')

        print('v1_1 timestamp: {0}, num_frames:{1}'.format(time2str(cam1_t[0]), len(cam1)))
        print('v1_2 timestamp: {0}, num_frames:{1}'.format(time2str(cam2_t[0]), len(cam2)))

        print('Loaded video data.')

    # ============================================================================================
    fig, ax = plt.subplots(figsize=(18,5))

    if plot_pir:
        # plot all signals
        ax.plot(pir1_t, pir1, label='pir', color='b')
        # ax.plot(pir2_t, pir2, label='pir', color='b')
        # ax.set_title('pir')

    if plot_ultra:
        # fig, ax = plt.subplots(figsize=(18,5))
        ax.plot(pir1_t, ultra1, label = 'ultra', color='g')
        # ax.plot(pir2_t, ultra2, label = 'ultra', color='g')
        # ax.set_title('ultra')

    if plot_video:
        # fig, ax = plt.subplots(figsize=(18,5))
        ax.plot(cam1_t, cam1, label='cam1', color='r')
        ax.plot(cam2_t, cam2, label='cam2', color='g')
        # ax.set_title('cam2')

    plt.legend()
    plt.draw()


def generate_1d_video_signal():

    # ============================================================================================
    # Load the video signal
    # video_file = '/data/chris/datasets/1013/valpi1/freeflow/1013_v1_1_ff.mp4'
    # y_coord = (335, 395)

    video_file = '/data/chris/datasets/1013/valpi2/freeflow/1013_v1_2_ff.mp4'
    y_coord = (305, 350)

    vid = VideoData()
    cam = vid.generate_1d_signal(video_file, rotation=None, x_coord=None, y_coord=y_coord)
    np.save('/data/chris/datasets/1013/valpi2/freeflow/v1_2_1d_ff.npy', cam)

    print('Generated 1d signal.')


def generate_1d_pir_signal():

    folder = '0404_2017'
    save_dir = '../workspace/{0}/'.format(folder)
    data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
    norm_df = pd.read_csv(save_dir+'s1_2d_KF__20170404_162238_073133__20170404_172021_074926_prob95.csv', index_col=0)
    norm_df.index = norm_df.index.to_datetime()

    columns = [i for i in norm_df.columns if 'pir' in i]
    pir_heatmap = norm_df.ix[:, columns].values.T

    # remove noise data
    # nan_thres_p = 0.9
    # v_thres = stats.norm.ppf(1-(1-nan_thres_p)/2.0)
    # pir_heatmap[ (pir_heatmap>=-v_thres) & (pir_heatmap<=v_thres) ] = 0
    pir_heatmap[ np.isnan(pir_heatmap)] = 0.0

    pir = np.sum(pir_heatmap, 0)
    pir_t = deepcopy(norm_df.index).tolist()

    print('Loaded PIR data.')

    # ============================================================================================
    # Load the ultrasonic sensor data
    ultra = (norm_df['ultra'].values)*10.0 + 200.0
    print('Loaded ultrasonic data')

    # save the 2d data into 1d time series
    df_1d = pd.DataFrame(zip(pir, ultra), index=norm_df.index, columns=['pir', 'ultra'])
    df_1d.to_csv(save_dir+'s1_1d__20170404_162238_073133__20170404_172021_074926_prob95.csv')
    print('Saved 1d pir+ultra data')


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
