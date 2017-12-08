import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils import test_set, train_set
from shutil import rmtree
from os import path


class MNIST(object):

    def __init__(self):
        self.opt = tf.app.flags.FLAGS
        self.reg_param = 0.000001
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.lmbd = 0.5

        if path.exists(self.opt.tboard_dir + '/train'):
            rmtree(self.opt.tboard_dir + '/train')

        if path.exists(self.opt.tboard_dir + '/validation'):
            rmtree(self.opt.tboard_dir + '/validation')



    def squash(self, s):
        len_s = tf.norm(s)
        return tf.scalar_mul(len_s/(1+len_s**2), s)


    def save_checkpoint(self, saver, sess, epoch, checkpoint_dir='./checkpoints_new', model_name='capsnet'):
        checkpoint_name = path.join(checkpoint_dir, model_name + '_' + str(epoch))
        print(" [*] Save checkpoint %s ..." % checkpoint_name)
        saver.save(sess, checkpoint_name)


    def load_checkpoint(self, saver, sess, epoch, checkpoint_dir='./checkpoints', model_name='capsnet'):
        try:
            checkpoint_name = path.join(checkpoint_dir, model_name + '_' + str(epoch))
            print(" [*] Reading checkpoint %s ..." % checkpoint_name)
            saver.restore(sess, checkpoint_name)
            print(" [*] Load SUCCESS")
        except Exception as exception:
            print(" [!] Load failed...")



    def fit(self):
        x_train, y_train = train_set()
        x_test, y_test = test_set()

        with tf.device('/gpu:0'):

            with tf.variable_scope('CapsNet'):

                inp_phld = tf.placeholder(dtype=tf.float32, shape=[None,28,28,1])
                outp_phld = tf.placeholder(dtype=tf.float32, shape = [None, 10])



                with tf.variable_scope('Conv_layer'):
                    conv1 = self.conv2d(inp_phld, n_output_filters=256,kernel_size=[9,9], name = 'conv1',padding='VALID')

                with tf.variable_scope('Capsule_layer'):
                    conv2 = self.conv2d(conv1, n_output_filters=256, kernel_size=[9,9], name = 'conv2', padding='VALID', stride=(2,2))
                    prim_caps = tf.reshape(conv2, shape = [-1,6,6,8,32], name = 'prim_caps')
                    prim_caps = tf.reshape(tf.transpose(prim_caps, perm = [3,0,1,2,4]),
                                           shape=[8,self.opt.batch_size,32,36]) #bringing the index to be squashed forward, in order to use the tf.map_fn()

                with tf.variable_scope('Capsules_squashed'):
                    with tf.variable_scope('W_ij'):
                        Wij = tf.get_variable(name='W',shape = [32,10,16,8],initializer=tf.truncated_normal_initializer(stddev=0.2)) #capsule layer weights

                    u_i = tf.map_fn(self.squash, prim_caps, name = 'caps_squashed')
                    u_ji = tf.reshape(tf.einsum('ijkl,lbic->kbicj', Wij, u_i),shape=[16,self.opt.batch_size,32*36,10])




                with tf.variable_scope('Routing'):
                    with tf.variable_scope('routing_vars'):
                        b_ij = tf.get_variable(name = 'bij',shape = [self.opt.batch_size,32*6*6,10],initializer=tf.zeros_initializer(), trainable=False)

                    for i in range(5):
                        with tf.variable_scope('Dynamic_routing_'+str(i)):
                            c_ij = tf.nn.softmax(b_ij, name = 'cij')
                            s_j = tf.einsum('bij,kbij->kbj',c_ij,u_ji)
                            v_j = tf.map_fn(self.squash, s_j, name = 's_j_squashed')
                            tf.assign_add(b_ij, tf.einsum('kbij,kbj->bij', u_ji, v_j), name='b_ij_'+str(i))


                with tf.variable_scope('final_capsules'):
                    c_ij = tf.nn.softmax(b_ij, name = 'c_ij_final_values')

                    s_j = tf.einsum('bij,kbij->kbj',c_ij, u_ji)
                    v_j = tf.map_fn(self.squash, s_j, name = 's_j_final_values')
                    pred_probs = tf.norm(v_j, axis=0, name = 'predicted_probs')


                with tf.variable_scope('reconstruction'):
                    v_masked = tf.einsum('bij,bj->bij',tf.transpose(v_j, perm=[1,0,2]), outp_phld)
                    v_flat = tf.reshape(v_masked, shape=[-1,16*10])

                    fc1 = tf.nn.relu(self.dense(v_flat, n_output=512, name='fc1'))
                    fc2 = tf.nn.relu(self.dense(fc1, n_output=1024, name='fc2'))
                    outp = tf.nn.sigmoid(self.dense(fc2, n_output=784, name='reconstr'), name='output')

                    reconstr_number = tf.reshape(outp, shape=[-1,28,28,1])




                with tf.variable_scope('losses'):
                    margin_loss = tf.multiply(outp_phld, tf.square(tf.maximum(-pred_probs+self.m_plus,0))) + tf.scalar_mul(self.lmbd, tf.multiply(
                        tf.subtract(1.0,outp_phld), tf.square(tf.maximum(pred_probs-self.m_minus,0))))
                    margin_loss = tf.reduce_mean(margin_loss)

                    reconstr_loss = tf.reduce_mean(tf.nn.l2_loss(inp_phld-reconstr_number))

                    loss = margin_loss + self.reg_param*reconstr_loss

                    correct_preds = tf.equal(tf.argmax(pred_probs, axis=1), tf.argmax(outp_phld, axis=1))
                    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))


                with tf.variable_scope('summaries'):
                    with tf.name_scope('accuracy'):
                        tf.summary.scalar('accuracy', accuracy)

                    with tf.name_scope('input_images'):
                        tf.summary.image(name='input_number', tensor=inp_phld, max_outputs=3)
                    with tf.name_scope('reconstructed_images'):
                        tf.summary.image(name='reconstr_images', tensor=reconstr_number, max_outputs=3)


                    with tf.name_scope('losses'):
                        tf.summary.scalar('margin loss', margin_loss)
                        tf.summary.scalar('reconstr loss', self.reg_param*reconstr_loss)
                        tf.summary.scalar('loss', loss)

                global_step = tf.Variable(0, trainable=False)


                with tf.variable_scope('learning_rate'):
                    learning_rate = tf.train.exponential_decay(self.opt.learning_rate*10, global_step, self.opt.epochs*len(x_train)/self.opt.batch_size,
                                                               0.95, staircase=True)
                    tf.summary.scalar('learning_rate', learning_rate)

                train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, name = 'adam').minimize(loss, global_step=global_step)

                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.per_process_gpu_memory_fraction = 0.7
                sess = tf.Session(config=config)

                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(self.opt.tboard_dir + '/train/', sess.graph)
                val_writer = tf.summary.FileWriter(self.opt.tboard_dir + '/validation/', sess.graph)

                saver = tf.train.Saver(max_to_keep=1000)

                init = tf.global_variables_initializer()
                sess.run(init)

                # # if load_checkpoint:
                # i=99
                # self.load_checkpoint(saver, sess, i)
                # trn_acc = list()
                # for j in range(55000/self.opt.batch_size):
                #     trn_acc.append(sess.run(accuracy,feed_dict={inp_phld:x_train[j*self.opt.batch_size:(j+1)*self.opt.batch_size,:,:,:],
                #                                                 outp_phld:y_train[j*self.opt.batch_size:(j+1)*self.opt.batch_size,:]}))
                # print 'Train accuracy after epoch ' + str(i+1) + ' is ' + str(np.mean(trn_acc))
                # tst_acc = list()
                # for j in range(10000/self.opt.batch_size):
                #     tst_acc.append(sess.run(accuracy,feed_dict={inp_phld:x_test[j*self.opt.batch_size:(j+1)*self.opt.batch_size],
                #                                                 outp_phld:y_test[j*self.opt.batch_size:(j+1)*self.opt.batch_size]}))
                # print 'Test accuracy after epoch ' + str(i+1) + ' is ' + str(np.mean(tst_acc))
                # quit()


                for i in range(self.opt.epochs):
                    print 'Epoch ', i+1
                    train_perm = np.random.permutation(len(y_train))
                    x_train_shuffled = x_train[train_perm,:,:,:]
                    y_train_shuffled = y_train[train_perm,:]
                    num_of_iters = len(y_train)/self.opt.batch_size
                    perc = 0.05

                    for iter in tqdm(range(num_of_iters)):
                        x_batch = x_train_shuffled[iter*self.opt.batch_size:(iter+1)*self.opt.batch_size]
                        y_batch = y_train_shuffled[iter*self.opt.batch_size:(iter+1)*self.opt.batch_size]

                        _, tr_acc, tr_lss, train_sum = sess.run([train_step, accuracy, loss, merged],
                                                                feed_dict={inp_phld:x_batch, outp_phld:y_batch})

                        test_perm = np.random.permutation(len(y_test))
                        y_test_shuffled = y_test[test_perm,:][:self.opt.batch_size*1]
                        x_test_shuffled = x_test[test_perm,:,:,:][:self.opt.batch_size*1]
                        test_acc, test_lss, val_sum = sess.run([accuracy, loss, merged], feed_dict={inp_phld:x_test_shuffled, outp_phld:y_test_shuffled})


                        if iter%100 == 0:
                            print 'Train loss and accuracy are ' + str(tr_lss) + ', ' + str(tr_acc)
                            print 'Test loss and accuracy are ' + str(test_lss) + ', ' + str(test_acc)


                        skip_first_summaries = (i==0 and iter <20)

                        if iter > perc * num_of_iters and not skip_first_summaries:
                            train_writer.add_summary(train_sum, int((i+perc)*20))
                            val_writer.add_summary(val_sum, int((i+perc)*20))
                            perc += 0.05

                    print '*****************'
                    trn_acc = list()
                    for j in range(55000/self.opt.batch_size):
                        trn_acc.append(sess.run(accuracy,feed_dict={inp_phld:x_train[j*self.opt.batch_size:(j+1)*self.opt.batch_size,:,:,:],
                                                                    outp_phld:y_train[j*self.opt.batch_size:(j+1)*self.opt.batch_size,:]}))
                    print 'Train accuracy after epoch ' + str(i+1) + ' is ' + str(np.mean(trn_acc))
                    tst_acc = list()
                    for j in range(10000/self.opt.batch_size):
                        tst_acc.append(sess.run(accuracy,feed_dict={inp_phld:x_test[j*self.opt.batch_size:(j+1)*self.opt.batch_size],
                                                                    outp_phld:y_test[j*self.opt.batch_size:(j+1)*self.opt.batch_size]}))
                    print 'Test accuracy after epoch ' + str(i+1) + ' is ' + str(np.mean(tst_acc))

                    #self.freeze_save_graph(sess, 'capsnet_' + '_' + str(i+1) + '.pb')
                    self.save_checkpoint(saver, sess, i+1)


    def conv2d(self, inp, n_output_filters, kernel_size, name, stride = (1,1), padding = 'SAME'):
        return tf.layers.conv2d(inputs=inp, filters = n_output_filters, kernel_size=kernel_size,strides=stride,
                                padding=padding, kernel_initializer=tf.truncated_normal_initializer(), activation=tf.nn.relu, name=name)

    def dense(self, inp, n_output, name):
        return tf.layers.dense(inputs=inp, units=n_output,kernel_initializer=tf.truncated_normal_initializer, name = name)



