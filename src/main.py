from TrafficSensorAlg import *

def main():

    data_analysis()
    # trim_video()
    # video_analysis()
    # sync_data()
    # veh_det()
    # generate_1d_pir_signal()

    # print update_fps(0.0, -0.526088, 37*60.0+4.838355, 60.12766)

    # test_alg()
    # generate_video_frames()
    # trim_out_dense_traffic_videos()

def data_analysis():
    # --------------------------------------------------
    # dataset 1013:
    # - Neil street. Freeflow and stop-and-to in total 33 min
    # - One PIR sensor array 4x32, at 64 Hz
    # --------------------------------------------------
    folder = '../datasets/1013_2016/'
    # dataset = '1310-210300'     # freeflow part 1
    # dataset = '1310-213154'   # freeflow part 2
    dataset ='1310-221543'       # stop and go


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
    # dataset 0316, 2017:
    # on Neil street before stop
    folder = '../datasets/0316_2017/s1/'
    dataset = '1603-175649'

    # ===============================================================================================
    # Configuration
    save_dir = '../workspace/0316_2017/'
    data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
    periods = data.get_data_periods(folder, update=True, f_type='txt')

    # either load txt or csv data. csv is faster
    if True:
        print('Loading txt data...')
        t1 = datetime.now()
        df = data.load_txt_data(folder+'{0}.txt'.format(dataset))
        t2 = datetime.now()
        print('Loading data took {0} s'.format((t2-t1).total_seconds()))

        # df.to_csv(folder+'{0}.csv'.format(dataset))
        # print('saved df data.')

    if False:
        t1 = datetime.now()
        df = pd.read_csv(folder+'{0}.csv'.format(dataset), index_col=0)
        df.index = df.index.to_datetime()
        t2 = datetime.now()
        print('Loading df csv data took {0} s'.format((t2-t1).total_seconds()))

    # ===============================================================================================
    # plot the original df
    if False:
        t_start = periods[dataset][0]
        t_end = periods[dataset][1]
        data.plot_heatmap_in_period(df, t_start=t_start, t_end=t_end, cbar=(0,40), option='vec',
                                    nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False, save_df=False,
                                    figsize=(18,8))

    # ===============================================================================================
    # plot the evolution of the nosie
    if False:
        t_start = periods[dataset][0]
        t_end = periods[dataset][1]
        data.plot_noise_evolution(df, t_start=t_start, t_end=t_end, p_outlier=0.01, stop_thres=(0.01,0.1),
                                  pixel=(1,10), window_s=10, step_s=3)
        data.plot_noise_evolution(df, t_start=t_start, t_end=t_end, p_outlier=0.01, stop_thres=(0.01,0.1),
                                  pixel=(1,25), window_s=10, step_s=3)

    # ===============================================================================================
    # analyze and debug MAP background subtraction
    if False:
        t_start = periods[dataset][0]
        t_end = periods[dataset][0] + timedelta(seconds=1000)
        t1 = datetime.now()
        map_norm_df = data.subtract_background_v2(df, t_start=t_start, t_end=t_end, init_s=4,
                                           veh_pt_thres=3, noise_pt_thres=3, prob_int=0.95, pixels=[(1,10)])
        t2 = datetime.now()
        print('\nNormalization sequential KF took {0} s'.format((t2-t1).total_seconds()))

    # ===============================================================================================
    # analyze and debug KF background subtraction
    if True:
        t_start = periods[dataset][0]
        t_end = periods[dataset][1]
        t1 = datetime.now()
        kf_norm_df = data.subtract_background_KF(df, t_start=t_start, t_end=t_end, init_s=4,
                                           veh_pt_thres=3, noise_pt_thres=3, prob_int=0.95, pixels=[(1,6)])
        t2 = datetime.now()
        print('\nNormalization sequential KF took {0} s'.format((t2-t1).total_seconds()))
        #
        # kf_norm_df.to_csv(save_dir + 's1_2d_KF__{0}__{1}_prob95.csv'.format(time2str_file(t_start),
        #                                                         time2str_file(t_end)))

        # kf_norm_df = pd.read_csv(save_dir+'/Q002_128/s1_2d_KF__20170316_184021_173046__20170316_185701_173046_prob95.csv', index_col=0)
        # kf_norm_df.index = kf_norm_df.index.to_datetime()
        # fig, ax = data.plot_heatmap_in_period(kf_norm_df, t_start=t_start, t_end=t_end, cbar=(0,4),
        #                                       option='vec', nan_thres_p=0.95, plot=True, save_dir=save_dir,
        #                                       save_img=False, save_df=False, figsize=(18,8))

    # ===============================================================================================
    # plot and normalize
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
    # Version 1 of MAP and FSM
    if False:
        t_start = periods[dataset][0]
        t_end = periods[dataset][0] + timedelta(seconds=30)
        t1 = datetime.now()
        map_norm_df = data.subtract_background(df, t_start=t_start, t_end=t_end, init_s=4,
                                           veh_pt_thres=3, noise_pt_thres=3, prob_int=0.8, pixels=None)
        t2 = datetime.now()
        print('Background subtraction took {0} s'.format((t2-t1).total_seconds()))

        # map_norm_df.to_csv(save_dir + 's1_2d_MAP__{0}__{1}.csv'.format(time2str_file(t_start),
        #                                                         time2str_file(t_end)))

        fig, ax = data.plot_heatmap_in_period(map_norm_df, t_start=t_start, t_end=t_end, cbar=(20,40),
                                              option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
                                              save_df=False, figsize=(18,8))
        fig, ax = data.plot_heatmap_in_period(map_norm_df, t_start=t_start, t_end=t_end, cbar=(0,4),
                                              option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
                                              save_df=False, figsize=(18,8))
    # version 2 of MAP and FSM
    if False:
        t_start = periods[dataset][0]
        t_end = periods[dataset][1]
        t1 = datetime.now()
        map_norm_df = data.subtract_background_v2(df, t_start=t_start, t_end=t_end, init_s=4,
                                           veh_pt_thres=3, noise_pt_thres=3, prob_int=0.95, pixels=None)
        t2 = datetime.now()
        print('Background subtraction v2 took {0} s'.format((t2-t1).total_seconds()))

        map_norm_df.to_csv(save_dir + 's1_2d_MAP__{0}__{1}_prob95.csv'.format(time2str_file(t_start),
                                                                time2str_file(t_end)))

        # fig, ax = data.plot_heatmap_in_period(map_norm_df, t_start=t_start, t_end=t_end, cbar=(20,40),
        #                                       option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
        #                                       save_df=False, figsize=(18,8))
        fig, ax = data.plot_heatmap_in_period(map_norm_df, t_start=t_start, t_end=t_end, cbar=(0,25),
                                              option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
                                              save_df=False, figsize=(18,8))

    # ===============================================================================================
    # plot saved heatmap
    if False:
        # load norm_df
        t_start = periods[dataset][0]
        t_end = periods[dataset][1]
        norm_df = pd.read_csv(save_dir+'s1_2d_MAP__20161013_221549_298439__20161013_222827_713595_prob80.csv', index_col=0)
        norm_df.index = norm_df.index.to_datetime()
        fig, ax = data.plot_heatmap_in_period(norm_df, t_start=t_start, t_end=t_end, cbar=(20,40),
                                              option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
                                              save_df=False, figsize=(18,8))
        # t_start = periods[dataset][0]
        # t_end = periods[dataset][1]
        # norm_df = pd.read_csv(save_dir+'s1_2d_MAP__20161013_210305_738478__20161013_211516_183544_prob90.csv', index_col=0)
        # norm_df.index = norm_df.index.to_datetime()
        # fig, ax = data.plot_heatmap_in_period(norm_df, t_start=t_start, t_end=t_end, cbar=(20,40),
        #                                       option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
        #                                       save_df=False, figsize=(18,8))
        # t_start = periods[dataset][0]
        # t_end = periods[dataset][1]
        # norm_df = pd.read_csv(save_dir+'s1_2d_MAP__20161013_210305_738478__20161013_211516_183544_prob95.csv', index_col=0)
        # norm_df.index = norm_df.index.to_datetime()
        # fig, ax = data.plot_heatmap_in_period(norm_df, t_start=t_start, t_end=t_end, cbar=(20,40),
        #                                       option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
        #                                       save_df=False, figsize=(18,8))



    # ===============================================================================================
    # plot the noise evolution
    # data.plot_noise_evolution(df, p_outlier=0.01, t_start=periods[dataset][0],
    #                           t_end=periods[dataset][1],
    #                           stop_thres=(0.01, 0.1), pixel=(1, 8), window_s=300, step_s=1)
    # data.plot_noise_evolution(df, p_outlier=0.01, t_start=periods[dataset][0],
    #                           t_end=periods[dataset][1],
    #                           stop_thres=(0.01, 0.1), pixel=(2, 8), window_s=300, step_s=1)
    # data.plot_noise_evolution(df, p_outlier=0.01, t_start=periods[dataset][0],
    #                           t_end=periods[dataset][1],
    #                           stop_thres=(0.01, 0.1), pixel=(1, 24), window_s=300, step_s=1)
    # data.plot_noise_evolution(df, p_outlier=0.01, t_start=periods[dataset][0],
    #                           t_end=periods[dataset][1],
    #                           stop_thres=(0.01, 0.1), pixel=(2, 24), window_s=300, step_s=1)

    plt.show()
#

def test_alg():
    # --------------------------------------------------
    # dataset 1013:
    # - Neil street. Freeflow and stop-and-to in total 33 min
    # - One PIR sensor array 4x32, at 64 Hz
    # --------------------------------------------------
    folder = '../datasets/1013_2016/'
    # dataset = '1310-210300'     # freeflow part 1
    # dataset = '1310-213154'   # freeflow part 2
    dataset ='1310-221543'       # stop and go

    # ===============================================================================================
    # Configuration
    save_dir = '../workspace/1013/'

    # norm_df = pd.read_csv(save_dir + 's1_2d_MAP__20161013_210305_738478__20161013_211516_183544_prob95.csv', index_col=0)
    # norm_df = pd.read_csv(save_dir + 's1_2d_MAP__20161013_213203_763628__20161013_214035_278451_prob95.csv', index_col=0)
    norm_df = pd.read_csv(save_dir + 's1_2d_MAP__20161013_221549_298439__20161013_222827_713595_prob95.csv', index_col=0)
    norm_df.index = norm_df.index.to_datetime()

    # data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
    # fig, ax = data.plot_heatmap_in_period(norm_df, t_start=None, t_end=None, cbar=(0,4),
    #                                           option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
    #                                           save_df=False, figsize=(18,8))


    # ===============================================================================================
    # run algorithm in batch mode
    alg = TrafficSensorAlg(pir_res=(4,32))

    # only run those congested traffic
    _ct_s = str2time_file('20161013_221809_310524')
    _ct_e = str2time_file('20161013_221821_805241')
    _idx = (norm_df.index >= _ct_s) & (norm_df.index <= _ct_e)
    _norm_df = norm_df.ix[_idx, :]

    alg.batch_run(_norm_df, det_thres=600, window_s=5.0, step_s=2.5,
                  save_dir='../workspace/1013/figs/speed_est_95_cg_batch/')
    # alg.vehs = np.load('../workspace/1013/figs/speed_est_95_ff2_batch/detected_vehs.npy')
    print('Total {0} vehs'.format(len(alg.vehs)))
    alg.plot_detected_vehs(norm_df, ratio_tx=6.0)

    plt.show()


def generate_video_frames():
    # ===============================================================================================
    # generate a video
    folder = '../datasets/1013_2016/'
    dataset = '1310-210300'     # freeflow part 1
    # dataset = '1310-213154'   # freeflow part 2
    # dataset ='1310-221543'       # stop and go

    # ========================================================
    # Load the raw PIR data
    print('Loading raw data...')
    t1 = datetime.now()
    raw_df = pd.read_csv(folder+'{0}.csv'.format(dataset), index_col=0)
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
    folder = '../datasets/1013_2016/'
    dataset = '1310-210300'     # freeflow part 1
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

    # ===============================================================================================
    # Configuration
    save_dir = '../workspace/1013/'

    norm_df = pd.read_csv(save_dir + 's1_2d_MAP__20161013_210305_738478__20161013_211516_183544_prob95.csv', index_col=0)
    # norm_df = pd.read_csv(save_dir + 's1_2d_MAP__20161013_213203_763628__20161013_214035_278451_prob95.csv', index_col=0)
    norm_df.index = norm_df.index.to_datetime()

    data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
    fig, ax =data.plot_heatmap_in_period(norm_df, t_start=None, t_end=None, cbar=(0,4),
                                              option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
                                              save_df=False, figsize=(18,8))

    # ===============================================================================================
    # vehicle detection
    det = VehDet()
    vehs = det.detect_veh(norm_df, energy_thres=600, window_s=2.0, step_s=1.0)

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
    save_dir = '../workspace/1013/figs/speed_est_95_ff1/'
    # print('Estimating speed for {0} vehs'.format(len(vehs)))
    for veh in vehs[200:]:
        est = SpeedEst(norm_df.ix[veh[0]:veh[1]], pir_res=(4,32), plot=False, save_dir=save_dir)
        est.estimate_speed(stop_tol=(0.002, 0.01), dist=3.5, r2_thres=0.8, min_num_pts=200)

    plt.show()


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

    input_video = '../datasets/1013_2016/1013_v1_2_ff_hq.mp4'
    output_video = '../datasets/1013_2016/1013_v1_2_ff_period1.mp4'
    video_starttime = str2time('2016-10-13 20:57:27.7')
    intvl=(str2time('2016-10-13 21:03:05.0'), str2time('2016-10-13 21:15:16.0'))

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
    data_dir = '../datasets/1013_2016/'
    save_dir = '../workspace/0316_2017/'

    # ============================================================================================
    # Load PIR and ultrasonic sensor data
    if plot_pir:
        pir1_df = pd.read_csv(save_dir+'s1_1d__20161013_213203_763628__20161013_214035_278451_prob95.csv', index_col=0)
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
    plt.show()


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

    save_dir = '../workspace/0316_2017/'
    data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
    # norm_df = pd.read_csv(save_dir+'s1_2d_MAP__20161013_210305_738478__20161013_211516_183544_prob95.csv', index_col=0)
    norm_df = pd.read_csv(save_dir+'s1_2d_KF__20170316_184021_173046__20170316_191239_656578_prob95.csv', index_col=0)
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
    df_1d.to_csv(save_dir+'s1_1d__20161013_213203_763628__20161013_214035_278451_prob95.csv')
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
