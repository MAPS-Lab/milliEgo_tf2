"""
Training deep mmwave+imu odometry from pseudo ground truth
"""
import os
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

import time

from utility.data_loader import load_data_multi_timestamp

from utility.test_util import convert_rel_to_44matrix
from utility import plot_util

def test(seqs,model,model_name,epoch):
    with open(join(currentdir, 'config.yaml'), 'r') as f:
        cfg = yaml.full_load(f)
    data_dir = join(cfg['mvo']['multimodal_data_dir'],'test')
    IMU_LENGTH = cfg['nn_opt']['cross-mio_params']['imu_length']
    if IMU_LENGTH < 10:
        IMU_LENGTH = 10
    for seq in seqs:
        test_file = join(data_dir, 'turtle_seq_' + str(seq) + '.h5')
  
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


        fig_dir = join('./figs', model_name, str(seq))
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        out_pred_array, out_gt_array = np.array(out_pred_array), np.array(out_gt_array)
        plot_util.plot2d(out_pred_array, out_gt_array,
                         join(fig_dir, model_name + '_ep' + str(epoch) + '_seq' + str(seq) + '.png'))
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
    with open(join(currentdir, 'config.yaml'), 'r') as f:
        cfg = yaml.full_load(f)
    test_model_path = join(cfg['mvo']['model_dir'],cfg['eval']['model'])
    seqs = cfg['eval']['seqs']
    epochs = cfg['eval']['eqochs']
    for e in epochs:
        model = tf.keras.models.load_model(join(test_model_path,str(e)))
        test(seqs,model,cfg['eval']['model'],e)

if __name__ == "__main__":
    os.system("hostname")
    main()
