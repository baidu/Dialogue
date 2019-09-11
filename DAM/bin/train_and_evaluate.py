import sys
import os
import time

import cPickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import utils.evaluation as eva


def train(conf, _model):
    
    if conf['rand_seed'] is not None:
        np.random.seed(conf['rand_seed'])

    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # load data
    print('starting loading data')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    train_data, val_data, test_data = pickle.load(open(conf["data_path"], 'rb'))    
    print('finish loading data')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    val_batches = reader.build_batches(val_data, conf)

    print("finish building test batches")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # refine conf
    batch_num = int(len(train_data['y']) / conf["batch_size"])
    val_batch_num = len(val_batches["response"])

    conf["train_steps"] = conf["num_scan_data"] * batch_num
    conf["save_step"] = int(max(1, batch_num / 10))
    conf["print_step"] = int(max(1, batch_num / 100))

    print('configurations: %s' %conf)

    print('model sucess')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    _graph = _model.build_graph()
    print('build graph sucess')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    with tf.Session(graph=_graph) as sess:
        _model.init.run();
        if conf["init_model"]:
            _model.saver.restore(sess, conf["init_model"])
            print("sucess init %s" %conf["init_model"])

        average_loss = 0.0
        batch_index = 0
        step = 0
        best_result = [0, 0, 0, 0]

        for step_i in xrange(conf["num_scan_data"]):
            #for batch_index in rng.permutation(range(batch_num)):
            print('starting shuffle train data')
            shuffle_train = reader.unison_shuffle(train_data)
            train_batches = reader.build_batches(shuffle_train, conf)
            print('finish building train data')
            for batch_index in range(batch_num):

                feed = {
                    _model.turns: train_batches["turns"][batch_index], 
                    _model.tt_turns_len: train_batches["tt_turns_len"][batch_index],
                    _model.every_turn_len: train_batches["every_turn_len"][batch_index],
                    _model.response: train_batches["response"][batch_index], 
                    _model.response_len: train_batches["response_len"][batch_index],
                    _model.label: train_batches["label"][batch_index]
                }

                batch_index = (batch_index + 1) % batch_num;

                _, curr_loss = sess.run([_model.g_updates, _model.loss], feed_dict = feed)

                
                average_loss += curr_loss

                step += 1

                if step % conf["print_step"] == 0 and step > 0:
                    g_step, lr = sess.run([_model.global_step, _model.learning_rate])
                    print('step: %s, lr: %s' %(g_step, lr))
                    print("processed: [" + str(step * 1.0 / batch_num) + "] loss: [" + str(average_loss / conf["print_step"]) + "]" )
                    average_loss = 0

                
                if step % conf["save_step"] == 0 and step > 0:
                    index = step / conf['save_step']
                    score_file_path = conf['save_path'] + 'score.' + str(index)
                    score_file = open(score_file_path, 'w')
                    print('save step: %s' %index)
                    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

                    for batch_index in xrange(val_batch_num):
                
                        feed = { 
                            _model.turns: val_batches["turns"][batch_index],
                            _model.tt_turns_len: val_batches["tt_turns_len"][batch_index],
                            _model.every_turn_len: val_batches["every_turn_len"][batch_index],
                            _model.response: val_batches["response"][batch_index],
                            _model.response_len: val_batches["response_len"][batch_index],
                            _model.label: val_batches["label"][batch_index]
                        }   
                
                        scores = sess.run(_model.logits, feed_dict = feed)
                    
                        for i in xrange(conf["batch_size"]):
                            score_file.write(
                                str(scores[i]) + '\t' + 
                                str(val_batches["label"][batch_index][i]) + '\n')
                    score_file.close()

                    #write evaluation result
                    result = eva.evaluate(score_file_path)
                    result_file_path = conf["save_path"] + "result." + str(index)
                    with open(result_file_path, 'w') as out_file:
                        for p_at in result:
                            out_file.write(str(p_at) + '\n')
                    print('finish evaluation')
                    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

                    if result[1] + result[2] > best_result[1] + best_result[2]:
                        best_result = result
                        _save_path = _model.saver.save(sess, conf["save_path"] + "model.ckpt." + str(step / conf["save_step"]))
                        print("succ saving model in " + _save_path)
                        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
                    
                

