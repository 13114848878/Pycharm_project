#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/30 21:44
# @Author  : Sen Tian
# @Email   : 414319563@qq.com 
# @File    : CNeTS.py
# @Software: PyCharm Community Edition

import tensorflow as tf
from mynet import shuffle_data,ShowInTime
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)#不用科学计数法


class CNeTS:
    """ 默认使用Adam算法进行反向计算。固定卷积核大小为[3,3],固定为最大池化，核大小为
            [2,2]
    """
    def __init__(self,conv_channels, fc_units):
        self.conv_channels = conv_channels
        self.fc_units = fc_units

    # input shape:[batch,high,width,channel]
    def training(self,X_train,Y_train,X_val,Y_val,lr=0.001,is_train=True,
                 is_save=False,max_epoch=1000,kp=1,batch_number=32,show=False,
                 early_stopping=True,max_fail=100,save_path='./my_model',
                 isshowlast=True):
        self.is_train = is_train
        self.batch_number = batch_number
        self.X_batch = tf.placeholder(tf.float32,
                                 shape=[None,X_train.shape[1],
                                        X_train.shape[2],
                                        X_train.shape[3]],
                                 name='Input')
        self.Y_batch = tf.placeholder(tf.float32, shape=[None, Y_train[1]],
                                 name='Output')
        self.keep_prob = tf.placeholder('float', name='keep_probability')
        a = tf.nn.batch_normalization(self.X_batch, training=is_train, momentum=0.999,
                                      name='Input batch normalization')

        for out_channel in self.conv_channels:
            a = self.conv_and_pool_layer(a,out_channel)

        fc_input_unit = int(a.shape[1]*a.shape[2]*a.shape[3])
        a = tf.reshape(a,[-1,fc_input_unit])

        for fc_unit in self.fc_units[:-1]:
            z = tf.layers.dense(a, units=fc_unit, use_bias=False,
                            name='fc'+str(fc_unit))
            z = tf.nn.batch_normalization(z, training=self.is_train,
                                          name='BN' + str(fc_unit))
            a = tf.nn.relu(z,name='fc_relu'+str(fc_unit))
            a = tf.nn.dropout(a, keep_prob=self.keep_prob,
                              name='fc_drop_out'+str(fc_unit))
        # output layer
        z = tf.layers.dense(a, units=Y_train.shape[1], use_bias=False,
                            name='Output')
        z = tf.nn.batch_normalization(z, training=self.is_train,
                                      name='BN Output')
        Y_e = tf.nn.softmax(z, name='Y_predict')
        # cross entropy，使用V2版本，当labels时用placeholder传入时没有区别
        cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.Y_batch, logits=z, )
        self.loss = tf.reduce_mean(cross_ent, name='loss')
        # accuracy
        correct_prediction = tf.equal(tf.argmax(Y_e, 1),
                                      tf.argmax(self.Y_batch, 1))

        self.acc = tf.reduce_mean(tf.cast(correct_prediction, "float"),
                                  name='accuracy')
        training = tf.train.AdamOptimizer(learning_rate=lr, name='training') \
            .minimize(self.loss,)
        # initial
        best_pref, best_index = np.Inf, 0
        loss_all = np.zeros([max_epoch, 3])
        acc_all = np.zeros([max_epoch, 3])

        cpu_num = 8
        config = tf.ConfigProto(device_count={"CPU": cpu_num},
                                inter_op_parallelism_threads=cpu_num,
                                intra_op_parallelism_threads=cpu_num,
                                #                                session_inter_op_thread_pool = 16,
                                log_device_placement=False)
        # save model
        if is_save:
            saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # print(kp)
            for step in range(max_epoch):
                X_train, Y_train = shuffle_data(X_train, Y_train)

                for s in range(0, Y_train.shape[0], batch_number):
                    #
                    e = s + self.batch_number

                    sess.run(training,
                             feed_dict={self.X_batch: X_train[s:e],
                                        self.Y_batch: Y_train[s:e],
                                        self.keep_prob: kp,
                                        self.is_train: True, })

                if early_stopping:
                    # Validation preform

                    loss_val, acc_val = self.do_evaluation(sess,X=X_val,Y=Y_val,)
                    # print(cm_val)
                    #                    loss_val,cm_val = self.do_evaluation(sess,X=X_val,Y=Y_val,)
                    # acc_val = cm_acc(cm_acc_val)
                    new_pref = loss_val
                    # new_pref = -ACC_val
                    if new_pref < best_pref:
                        best_pref = new_pref
                        best_index = step
                        best_acc = acc_val

                        if is_save:
                            saved_path = saver.save(sess, save_path + '/model',
                                                    global_step=step, )
                            best_val_res = [best_index, kp, self.batch_number,
                                            lr,best_pref, best_acc]
                            np.savetxt(save_path + '/parameter.txt',
                                       best_val_res)

                    # Visualization
                    if show:
                        # train and test preform
                        loss_train, acc_train = self.do_evaluation(sess,
                                                                   X=X_train,
                                                                   Y=Y_train, )
                        # loss_test, acc_test = self.do_evaluation(sess, X=X_test,
                        #                                          Y=Y_test, )

                        #                        acc_train = cm_acc(cm_train)
                        #                        acc_test = cm_acc(cm_test)

                        loss_all[step] = loss_train, loss_val,
                        acc_all[step] = acc_train, acc_val,

                        # ShowInTime().show_loss(loss_all[:i])

                        ShowInTime(isshowlast).show_loss_ACC(loss_all[:step],
                                                             acc_all[:step], )

                    # Early Stopping
                    if (step - best_index) > max_fail:

                        if show:
                            ShowInTime(isshowlast).last_show_ACC(
                                loss_all[:step], acc_all[:step], )
                            plt.suptitle(
                                'Best validation preform:{0:.3f}(Loss),{1:.3f}(ACC) at epoch:{2}'
                                .format(best_pref, acc_all[best_index, 1],
                                        best_index))

                        break
        # return result

            return best_pref, best_acc

    def do_evaluation(self, sess, X, Y):
        """分步计算损失和正确率
        """
        loss_sum = 0
        acc_sum = 0
        i = 0
        for s in range(0, X.shape[0], self.batch_number):
            step_loss, step_acc, = sess.run([self.loss, self.acc, ],
                                            feed_dict={
                                                self.X_batch: X[
                                                              s:s + self.batch_number],
                                                self.Y_batch: Y[
                                                              s:s + self.batch_number],
                                                self.keep_prob: 1.0,
                                                self.is_train: False,
                                            })
            i += 1
            loss_sum += step_loss  # 本身就是平均过的结果,与求和再平均之间的误差在数据量较大时可忽略
            acc_sum += step_acc
        return loss_sum / i, acc_sum / i

    # 卷积加池化，卷积核默认为3，池化默认为最大池化核大小默认为2
    def conv_and_pool_layer(self,a,out_channel,conv_size=3,pool_size=2):
        filter = [a.shape[1],a.shape[2],a.shape[3],out_channel]
        z = tf.nn.conv2d(a,filter,strides=[1,conv_size,conv_size,1],
                         padding='SAME',name='convolution'+str(out_channel),)

        z = tf.nn.batch_normalization(z, training=self.is_train,
                                      name='BN'+str(out_channel))
        a =tf.nn.relu(z, name='relu')
        # pooling
        a = tf.nn.max_pool(a,ksize=[1,pool_size,pool_size,1],padding='VALID',
                           name='max pool'+str(out_channel))
        return a

