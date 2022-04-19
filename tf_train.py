"""
Training deep mmwave+imu odometry from pseudo ground truth
"""
import os
import pickle
import tensorflow as tf
from os.path import join
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import numpy as np
import matplotlib as mpl
import yaml
mpl.use('Agg')
import glob
import json
import time
# keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D,LeakyReLU,Input,concatenate
import math

# utiliy
from utility.new_networks import build_model_cross_att
from utility.data_loader import load_data_multi, validation_stack,load_data_multi_timestamp
from tensorflow.keras.optimizers import RMSprop, Adam
from utility.test_util import convert_rel_to_44matrix, iround
from utility import plot_util

def test(seqs,model,model_name,epoch):
    with open(join(currentdir, 'config.yaml'), 'r') as f:
        cfg = yaml.full_load(f)
    data_dir = join(cfg['mvo']['multimodal_data_dir'],'test')
    IMU_LENGTH = cfg['nn_opt']['cross-mio_params']['imu_length']
    if IMU_LENGTH < 10:
        IMU_LENGTH = 10
    for seq in seqs:
        test_file = join(data_dir, 'turtle_seq_' + seq + '.h5')
  
        n_chunk, x_time, x_mm_t, x_imu_t, y_t = load_data_multi_timestamp(test_file,
                                                                              'mmwave_middle')  # y (1, 2142, 6)

        y_t = y_t[0]
        print('Data shape: ', np.shape(x_mm_t), np.shape(x_imu_t), np.shape(y_t))
        len_x_i = x_mm_t[0].shape[0]
        print(len_x_i)

        # Set initial pose for GT and prediction
        gt_transform_t_1 = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])
        pred_transform_t_1 = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])  # initial pose for prediction

        # Initialize value and counter
        count_img = 0
        ls_time = [0, 0, 0, 0]

        out_gt_array = []  # format (x,y) gt and (x,y) prediction
        out_pred_array = []  # format (x,y) gt and (x,y) prediction

        print('Reading images and imu ....')
        # for i in range(0, iround ((len_thermal_x_i-2)/2)):
        for i in range(0, (len_x_i - 1)):
            # Make prediction
            st_cnn_time = time.time()
            x_mm_1 = x_mm_t[0][i]
            x_mm_2 = x_mm_t[0][i + 1]

            # x_mm_1 = np.expand_dims(x_mm_1, axis=0)
            # x_mm_2 = np.expand_dims(x_mm_2, axis=0)
            # x_mm_1 = np.repeat(x_mm_1, 3, axis=-1)
            # x_mm_2 = np.repeat(x_mm_2, 3, axis=-1)            

            print('x_mm_2 shape is {}'.format(np.shape(x_mm_2)))

            # x_imu = x_imu_t[0]
            # x_imu = x_imu[i+1, 0:IMU_LENGTH, :]
            # x_imu = np.expand_dims(x_imu, axis=0)
            x_imu = []
            x_imu.extend(x_imu_t[0][i+1, 0:IMU_LENGTH, :])
            x_imu = np.expand_dims(x_imu, axis=0)

         
            predicted = model({"image_1":x_mm_1,"image_2":x_mm_2,"imu_data":x_imu})  
            # print(y_t[i])
            pred_pose = np.reshape(predicted,6)
            prediction_time = time.time() - st_cnn_time
            ls_time[0] += prediction_time
            
            print('Running (Hz)', 1.0 / (prediction_time))

            # Display the figure
            st_plot_time = time.time()
        
            # Composing the relative transformationnetwork_model for the prediction
     
            pred_transform_t = convert_rel_to_44matrix(0, 0, 0, pred_pose)
            abs_pred_transform = np.dot(pred_transform_t_1, pred_transform_t)
            # print(abs_pred_transform)

            # Composing the relative transformation for the ground truth
            gt_transform_t = convert_rel_to_44matrix(0, 0, 0, y_t[i])
            abs_gt_transform = np.dot(gt_transform_t_1, gt_transform_t)
            # print(abs_gt_transform)

            # Save the composed prediction and gt in a list
            out_gt_array.append(
                [abs_gt_transform[0, 0], abs_gt_transform[0, 1], abs_gt_transform[0, 2], abs_gt_transform[0, 3],
                 abs_gt_transform[1, 0], abs_gt_transform[1, 1], abs_gt_transform[1, 2], abs_gt_transform[1, 3],
                 abs_gt_transform[2, 0], abs_gt_transform[2, 1], abs_gt_transform[2, 2], abs_gt_transform[2, 3]])

            out_pred_array.append(
                [abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2], abs_pred_transform[0, 3],
                 abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2], abs_pred_transform[1, 3],
                 abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2],
                 abs_pred_transform[2, 3]])

            plot_time = time.time() - st_plot_time
            ls_time[1] += plot_time

            gt_transform_t_1 = abs_gt_transform
            pred_transform_t_1 = abs_pred_transform
            count_img += 1

        if not os.path.exists('./results'):
            os.makedirs('./results')
        # csv_location = join('./results', exp_name)


        fig_dir = join('./figs', model_name, seq)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        out_pred_array, out_gt_array = np.array(out_pred_array), np.array(out_gt_array)
        plot_util.plot2d(out_pred_array, out_gt_array,
                         join(fig_dir, model_name + '_ep' + str(epoch) + '_seq' + seq + '.png'))
        for a in range(2):
            if a == 0:
                ls_time[a] = ls_time[a] / count_img
                ls_time[a] = int(round(ls_time[a] * 1000, 0))
            else:
                ls_time[a] = ls_time[a] / count_img
                ls_time[a] = int(round(ls_time[a] * 1000, 0))

        print('Model Prediction: {0} ms. Plot: {1} ms.'.format(str(ls_time[0]), str(ls_time[1])))
        print("Seq {} Finished!".format(seq))


def main():
    print('For mmwave+imu odom!')

    with open(join(currentdir, 'config.yaml'), 'r') as f:
        cfg = yaml.full_load(f)

    MODEL_NAME = cfg['nn_opt']['cross-mio_params']['nn_name']


    IMU_LENGTH = cfg['nn_opt']['cross-mio_params']['imu_length']
    if IMU_LENGTH < 10:
        IMU_LENGTH = 10
    print('IMU LENGTH is {}'.format(IMU_LENGTH))

    model_dir = join('./models', MODEL_NAME)

    print("Building network model .....")
    print(cfg['nn_opt']['cross-mio_params'])
    model = build_model_cross_att(cfg['nn_opt']['cross-mio_params'],
                                     mask_att=cfg['nn_opt']['cross-mio_params']['cross_att_type'],
                                            imu_length=IMU_LENGTH,input_shape=(64,256,1))
    # for e in ['35']:
    #     e = str(e)
    #     model = tf.keras.models.load_model("C:/milliEgo/milliEgo-tf/models/cross-mio_turtle(new_data_4_huber_dense_ex67)/"+e)
    #     test(['9','10','11','12','13','14','15','16','17','18'],model,'cross-mio_turtle(new_data_4_huber_dense_ex67)',e)
    # sys.exit()
    model.summary()


    # grap training files
    path = cfg['mvo']['tf_data_dir']
    training_files = sorted(glob.glob(join(path, '*')))
    n_training_files = len(training_files)

    cfgg = cfg['nn_opt']['cross-mio_params']
    lr = cfgg['lr_rate']
    adam = Adam(learning_rate= lr)
    rmsProp = RMSprop(learning_rate= lr , rho=cfgg['rho'],
                                           epsilon=float(cfgg['epsilon']),
                                           decay=cfgg['decay'],clipvalue=0.02)

    model.compile(optimizer=adam, loss={'fc_trans':'mse', 'fc_rot':'mse'},
                      loss_weights={'fc_trans': cfgg['fc_trans'],
                                    'fc_rot': cfgg['fc_rot']})
    datasets = []
    for training_file in training_files:
            with open(training_file + '/element_spec' , 'rb') as in_:
                es = pickle.load(in_)

            loaded = tf.data.experimental.load(
                 training_file, es, compression='GZIP'
            )
            datasets.append(loaded)

    dataset = datasets[0]

    for i in range(len(datasets)-1):
        dataset = dataset.concatenate(datasets[1+i])


    for e in range(cfg['mvo']['epochs']+1):
        print("|-----> epoch %d" % e)
       
        if((e%1)==0 and e>0):
            model.optimizer.lr = model.optimizer.lr*0.95
        # if e <= 4: continue
        training_dataset = dataset.shuffle(120).batch(16)
        
        print(len(dataset)," training samples")

        for elem in training_dataset:
            x_1,x_2,imu,y = elem
            # print(y[0,3:])
            # x_1 = np.repeat(x_1, 3, axis=-1)
            # x_2 = np.repeat(x_2, 3, axis=-1)
            # x_1,x_2 = np.expand_dims(x_1,1),np.expand_dims(x_2,1)

            model.fit({'image_1': x_1, 'image_2': x_2, 'imu_data': imu[:,]},
                                {'fc_trans': y[:,0:3], 'fc_rot':y[:,3:6]},batch_size =1)

        if ((e % 1) == 0):
            model.save(join(model_dir, str(e)))
            test(['9'],model,MODEL_NAME,e)
        if e == 0:
            print('Saving nn options ....')
            with open(join(model_dir, 'nn_opt.json'), 'w') as fp:
                json.dump(cfg['nn_opt']['cross-mio_params'], fp)


    print("Training for model has finished!")

    # print('Saving training loss ....')
    # train_loss = np.array(training_loss)
    # loss_file_save = join(model_dir, 'training_loss.' + MODEL_NAME +'.h5')
    # with h5py.File(loss_file_save, 'w') as hf:
    #     hf.create_dataset('train_loss', data=train_loss)

    print('Finished training ', str(n_training_files), ' trajectory!')

if __name__ == "__main__":
    os.system("hostname")
    main()
