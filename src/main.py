from TrafficSensorAlg import *

def main():

    # data_analysis()
    # video_analysis()
    # sync_data()
    # veh_det()
    test_alg()

def data_analysis():
    # --------------------------------------------------
    # dataset 1013:
    # - Neil street. Freeflow and stop-and-to in total 33 min
    # - One PIR sensor array 4x32, at 64 Hz
    # --------------------------------------------------
    folder = '../datasets/1013_2016/'
    # dataset = '1310-210300'     # freeflow part 1
    dataset = '1310-213154'   # freeflow part 2
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
    data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
    periods = data.get_data_periods(folder, update=False, f_type='txt')

    # either load txt or csv data. csv is faster
    if False:
        print('Loading txt data...')
        t1 = datetime.now()
        df = data.load_txt_data(folder+'{0}.txt'.format(dataset))
        t2 = datetime.now()
        print('Loading data took {0} s'.format((t2-t1).total_seconds()))

        df.to_csv(folder+'{0}.csv'.format(dataset))
        print('saved df data.')

    if True:
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
        data.plot_heatmap_in_period(df, t_start=t_start, t_end=t_end, cbar=(20,40), option='vec',
                                    nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False, save_df=False,
                                    figsize=(18,8))

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
                                              option='vec', nan_thres_p=0.9, plot=True, save_dir=save_dir, save_img=False,
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
    if True:
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
        fig, ax = data.plot_heatmap_in_period(map_norm_df, t_start=t_start, t_end=t_end, cbar=(20,40),
                                              option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
                                              save_df=False, figsize=(18,8))

    # ===============================================================================================
    # plot saved heatmap
    # load norm_df
    t_start = periods[dataset][0]
    t_end = periods[dataset][1]
    norm_df = pd.read_csv(save_dir+'s1_2d_MAP__20161013_213203_763628__20161013_214035_278451_prob80.csv', index_col=0)
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
    dataset = '1310-210300'     # freeflow part 1
    # dataset = '1310-213154'   # freeflow part 2
    # dataset ='1310-221543'       # stop and go

    # ===============================================================================================
    # Configuration
    save_dir = '../workspace/1013/'

    norm_df = pd.read_csv(save_dir + 's1_2d_MAP__20161013_210305_738478__20161013_211516_183544_prob95.csv', index_col=0)
    # norm_df = pd.read_csv(save_dir + 's1_2d_MAP__20161013_213203_763628__20161013_214035_278451_prob95.csv', index_col=0)
    norm_df.index = norm_df.index.to_datetime()

    # data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
    # fig, ax =data.plot_heatmap_in_period(norm_df, t_start=None, t_end=None, cbar=(0,4),
    #                                           option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
    #                                           save_df=False, figsize=(18,8))


    # ===============================================================================================
    # run algorithm in batch mode
    alg = TrafficSensorAlg(pir_res=(4,32))
    # alg.batch_run(norm_df, det_thres=600, window_s=2.0, step_s=1.0,
    #               save_dir='../workspace/1013/figs/speed_est_95_ff1_batch/')
    alg.vehs = np.load('../workspace/1013/figs/speed_est_95_ff1_batch/detected_vehs.npy')
    print('Total {0} vehs'.format(len(alg.vehs)))
    alg.plot_detected_vehs(norm_df, ratio_tx=6.0)

    plt.show()


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


def sync_data():

    plot_video = True
    plot_pir = True
    plot_ultra = True
    data_dir = '../datasets/1118_2016/s1/'
    save_dir = '../workspace/1118/'
    # ============================================================================================
    if False:
        # Load the PIR signal
        if plot_pir:
            data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
            norm_df = pd.read_csv(save_dir+'s1_2d_20161118_145509_610213.csv', index_col=0)
            norm_df.index = norm_df.index.to_datetime()

            columns = [i for i in norm_df.columns if 'pir' in i]
            pir_heatmap = norm_df.ix[:, columns].values.T

            # remove noise data
            nan_thres_p = 0.9
            v_thres = stats.norm.ppf(1-(1-nan_thres_p)/2.0)
            pir_heatmap[ (pir_heatmap>=-v_thres) & (pir_heatmap<=v_thres) ] = 0

            pir = np.sum(pir_heatmap, 0)
            pir_t = deepcopy(norm_df.index).tolist()

            print('Loaded PIR data.')

        # ============================================================================================
        # Load the ultrasonic sensor data
        if plot_ultra:
            ultra = (norm_df['ultra'].values)*10.0 + 200.0
            print('Loaded ultrasonic data')

        # save the 2d data into 1d time series
        df_1d = pd.DataFrame(zip(pir, ultra), index=norm_df.index, columns=['pir', 'ultra'])
        df_1d.to_csv(save_dir+'s1_1d_20161118_145509_610213.csv')
        print('Saved 1d pir+ultra data')
    else:
        df_1d = pd.read_csv(save_dir+'s1_1d_20161118_145509_610213.csv', index_col=0)
        df_1d.index = df_1d.index.to_datetime()

        pir_t = df_1d.index
        pir = df_1d['pir']
        ultra = df_1d['ultra']

    # ============================================================================================
    # Load the video signal
    if plot_video:
        vid = VideoData()
        # t1 = datetime.now()
        # cam1_heatmap = np.load(save_dir+'1013_2016_freeflow_cam1_heatmap.npy')
        # cam2_heatmap = np.load(save_dir+'1013_2016_freeflow_cam2_heatmap.npy')
        # t2 = datetime.now()
        # print('Loaded video heatmap {0} s'.format((t2-t1).total_seconds()))

        # vid.plot_heatmap(cam1_heatmap,figsize=(18,8), plot=True,
        #                  save_img='../datasets/1013_2016/1013_2016_freeflow_cam1_heatmap.png',
        #                  title='2016 Oct 13 freeflow cam 1')
        # vid.plot_heatmap(cam2_heatmap,figsize=(18,8), plot=True,
        #                  save_img='../datasets/1013_2016/1013_2016_freeflow_cam2_heatmap.png',
        #                  title='2016 Oct 13 freeflow cam 2')
        # t3 = datetime.now()
        # print('Plotted heatmap {0} s'.format((t3-t2).total_seconds()))

        # summarize each column to get 1d signal
        # cam1 = np.sum(cam1_heatmap, 0)
        # cam2 = np.sum(cam2_heatmap, 0)

        cam1 = np.load(data_dir + 'v1_1_signal.npy')*400.0
        cam2 = np.load(data_dir + 'v1_2_signal.npy')*400.0

        # get the timestamps
        fps_1 = 90.4
        fps_2 = 90.0
        inc_1 = 1.0/fps_1
        inc_2 = 1.0/fps_2

        ####
        offset_cam1 = timedelta(seconds=84.0)
        offset_cam2 = timedelta(seconds=0.0)
        t_init = datetime(year=2016, month=11, day=18, hour=0, minute=0, second=0)
        ####
        cam1_t = pd.date_range(t_init+offset_cam1,periods=len(cam1),freq=str(int(inc_1*1e9))+'N')
        cam2_t = pd.date_range(t_init+offset_cam2,periods=len(cam2),freq=str(int(inc_2*1e9))+'N')

        print('v1_1 timestamp: {0}, num_frames:{1}'.format(time2str(cam1_t[0]), len(cam1)))
        print('v1_2 timestamp: {0}, num_frames:{1}'.format(time2str(cam2_t[0]), len(cam2)))

        print('Loaded video data.')

    # ============================================================================================
    # Use pandas correlation to find the shift
    # video_df = pd.DataFrame(video, index=video_t)
    # pir_df = pd.DataFrame(pir, index=pir_t)
    #
    # y = pd.rolling_corr(video_df, pir_df, 300)


    # ============================================================================================
    fig, ax = plt.subplots(figsize=(18,5))

    if plot_pir:
        # plot all signals
        pir_t_s = [(i-pir_t[0]).total_seconds() for i in pir_t]
        ax.plot(pir_t, pir, label='pir', color='b')
        # ax.set_title('pir')

    if plot_ultra:
        # fig, ax = plt.subplots(figsize=(18,5))
        pir_t_s = [(i-pir_t[0]).total_seconds() for i in pir_t]
        ax.plot(pir_t, ultra, label = 'ultra', color='g')
        # ax.set_title('ultra')

    if plot_video:
        fig, ax = plt.subplots(figsize=(18,5))
        ax.plot(cam1_t, cam1, label='cam1', color='r')
        ax.plot(cam2_t, cam2, label='cam2', color='g')
        # ax.set_title('cam2')

    plt.legend()
    plt.draw()
    plt.show()




if __name__ == '__main__':
    main()
