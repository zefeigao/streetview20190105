class args:
    seed = 0
    cuda = True
    bs = 100
    epochs = 100
    chkpt_path = './streetView_model/'
    data_path = '/media/deally/F48273B0827375C8/streetViewData'
    eval_data_root = '/media/deally/F48273B0827375C8/streetViewData/'
    log = './streetView_tmp/log.txt'

    z_dis = 10
    z_con = 2
    z_rnd = 70
    plot_samples = 10