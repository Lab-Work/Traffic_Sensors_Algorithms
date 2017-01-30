from TrafficSensorAlg import *

def main():

    data_analysis()
    # video_analysis()

def data_analysis():
    # --------------------------------------------------
    # dataset 1003:
    # - Neil street. Freeflow and stop-and-to in total 33 min
    # - One PIR sensor array 4x32, at 64 Hz
    # --------------------------------------------------
    folder = '../datasets/1013_2016/'
    # dataset = '1310-205905'   # initial 10 second
    dataset = '1310-213154'   # freeflow part 1
    # dataset='1310-210300'     # freeflow part 2
    # dataset='1310-221543'       # stop and go


    # --------------------------------------------------
    # dataset 1027:
    # - Neil street. Two PIR sensors
    # - One PIR sensor array 4x32, at 64 Hz
    # --------------------------------------------------
    # folder = '../datasets/1027_2016/senspi1/'
    # dataset = '2710-192614'   # 7 seconds
    # dataset = '2710-205846'     # 45 min

    # folder = '../datasets/1027_2016/senspi2/'
    # dataset = '2710-210055'   # 45 min

    # --------------------------------------------------
    # dataset 1103:
    # - Neil street. Three PIR sensor arraies
    # - One PIR sensor array 4x32, at 64 Hz
    # --------------------------------------------------
    # folder = '../datasets/1103_2016/s1/'
    # dataset = '0311-191717'

    # folder = '../datasets/1103_2016/s2/'
    # dataset = '0311-192414'

    # folder = '../datasets/1103_2016/s3/'
    # dataset = '0311-193053'

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
    save_dir = '../workspace/'
    data = SensorData(pir_res=(4,32), save_dir=save_dir, plot=False)
    periods = data.get_data_periods(folder, update=True, f_type='txt')
    df = data.load_txt_data(folder+'{0}.txt'.format(dataset))

    # ===============================================================================================
    # plot and normalize
    data.plot_heatmap_in_period(df, t_start=periods[dataset][0], t_end=periods[dataset][1], cbar=(20,40), option='vec',
                                nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False, save_df=False,
                                figsize=(18,8))
    # norm_df = data.batch_normalization(df, t_start=None, t_end=None, p_outlier=0.01, stop_thres=(0.1,0.01), window_s=30,
    #                          step_s=1)
    # norm_df.to_csv(save_dir+'norm_df.csv')

    # ===============================================================================================
    # plot saved heatmap
    # load norm_df
    norm_df = pd.read_csv(save_dir+'norm_df.csv', index_col=0)
    norm_df.index = norm_df.index.to_datetime()
    fig, ax = data.plot_heatmap_in_period(norm_df, t_start=periods[dataset][0], t_end=periods[dataset][1], cbar=(0,4),
                                          option='vec', nan_thres_p=0.9, plot=True, save_dir=save_dir, save_img=False,
                                          save_df=False, figsize=(18,8))

    plt.show()


def video_analysis():

    # ============================================================================================
    # play videos
    # ============================================================================================
    # vid = VideoData(save_dir='', fps=60, offset=0)
    # input_video = '../datasets/1013_2016/1013_2016_freeflow_cam2.mp4'
    # vid.play_video(input_video, x_coord=None, y_coord=(305, 350))


    # ============================================================================================
    # crop videos
    # ============================================================================================
    # vid = VideoData(save_dir='', fps=60, offset=0)
    # input_video = '../datasets/1013_2016/1013_2016_freeflow_cam2.mp4'
    # output_video = '../datasets/1013_2016/1013_2016_freeflow_cam2_cropped.mp4'

    # vid.crop_video(input_video=input_video, output_video=output_video, x_coord=None, y_coord=(305,350),
    #                frame_lim=None)


    # ============================================================================================
    # generate the heatmap
    # ============================================================================================
    # vid = VideoData(save_dir='', fps=60, offset=0)
    # input_video = '../datasets/1013_2016/1013_2016_freeflow_cam2_cropped.mp4'
    # heatmap = vid.generate_heatmap(input_video,save_npy='../datasets/1013_2016/1013_2016_freeflow_cam2_heatmap.npy')


    # ============================================================================================
    # plot the heatmap
    # ============================================================================================
    vid = VideoData(save_dir='', fps=60, offset=0)
    t1 = datetime.now()
    heatmap = np.load('../datasets/1013_2016/1013_2016_freeflow_cam2_heatmap.npy')
    t2 = datetime.now()
    print('Loaded heatmap {0} s'.format((t2-t1).total_seconds()))
    vid.plot_heatmap(heatmap,figsize=(18,8), plot=True,
                     save_img='../datasets/1013_2016/1013_2016_freeflow_cam2_heatmap.png',
                     title='2016 Oct 13 freeflow cam 2')
    t3 = datetime.now()
    print('Plotted heatmap {0} s'.format((t3-t2).total_seconds()))

    plt.show()






if __name__ == '__main__':
    main()