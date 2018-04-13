import tensorflow as tf
import numpy as np
import os.path
import pickle
import matplotlib.pyplot as plt
import time
from mydatautil import mydata

def writeToFile(var1, var2, var3, var4, wfn):
    with open(wfn,'w+') as wf:
        wf.write('x                         y                      pred                     baseline\n')
        for i in range(len(var1)):
            wf.write(str(var1[i]) + '      ' +  str(var2[i]) + '      ' + str(var3[i]) + '      ' + str(var4[i]) + '\n')
    return

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def swish(x):
    return tf.multiply(x,tf.nn.sigmoid(x))

def mlp2Lhistplot(modelname, mlp_ninput = 3, testmode = 0, epochs = 1000, breaklim = 0, plot = 0,
mlp_learning_rate = 0.05, traindata = 'traindata', testdata = 'testdata',
plotrangel = 0, plotrangeh = 100, display_step = 20, mlp_nbatchs = 3, mlp_nhidden1 = 50,
mlp_nhidden2 = 40, plotTitle = '900MHz two ray model', normalize = False):

    #tf.set_random_seed(1)
    # structural param
    mlp_noutput = 1
    mlp_dropout = 1
    # training param
    mlp_momentum = 0.2
    mlp_train_epochs = epochs
    mlp_shuffle_count = 50

    # data and path param
    TRAIN_DATA = traindata
    TEST_DATA = testdata
    savepath_b1pred = 'baseline_1'
    savepath_nnpred = 'nnpred_1'
    verbose = 1 # display the original and baseline data

    # stage param
    #testmode = 0 # 1:Load and test only
    fp_test = 1 # fp function will test data after training
    ctrain_use_models = 0 # Note: it will continue learning from train model instead of the previous ctrain model

    if testmode == 1:
        traindata = 1
        init_train = 0
        ctrain = 0
        fp = 1
    else:
        traindata = 1
        init_train = 1
        ctrain = 0
        fp = 1

    writer = tf.summary.FileWriter("./tb/fm/model1")

    ### Tensors:
    # mlp tensors
    mlp_weights = {
        'h1': tf.Variable(tf.random_normal([mlp_ninput, mlp_nhidden1],stddev=0.05, dtype=tf.float32, name='h1')),
        'h2': tf.Variable(tf.random_normal([mlp_nhidden1, mlp_nhidden2],stddev=0.05, dtype=tf.float32, name='h2')),
        'out': tf.Variable(tf.random_normal([mlp_nhidden2, mlp_noutput],stddev=0.05, dtype=tf.float32, name='wout'))
    }
    mlp_biases = {
        'b1': tf.Variable(tf.random_normal([mlp_nhidden1],stddev=0.05, dtype=tf.float32, name='b1')),
        'b2': tf.Variable(tf.random_normal([mlp_nhidden2],stddev=0.05, dtype=tf.float32, name='b2')),
        'out': tf.Variable(tf.random_normal([mlp_noutput],stddev=0.05, dtype=tf.float32, name='bout'))
    }
    # placeholders
    X = tf.placeholder("float", [None, mlp_ninput])
    Y = tf.placeholder("float", [None, 1])
    Z = tf.placeholder("float", [1,None])
    mlp_keepprob = tf.placeholder("float")
    ### operators:
    def mlp_t2(x, mlp_dropout):
        with tf.name_scope('mlp'):
            layer_1 = tf.add(tf.matmul(x, mlp_weights['h1']), mlp_biases['b1'])
            #layer_1 = tf.nn.relu(layer_1)
            layer_1 = swish(layer_1)
            layer_2 = tf.add(tf.matmul(layer_1, mlp_weights['h2']), mlp_biases['b2'])
            #layer_2 = tf.nn.relu(layer_2)
            layer_2 = swish(layer_2)
            layer_2 = tf.nn.dropout(layer_2, mlp_dropout)
            #layer_3 = tf.add(tf.matmul(layer_2, mlp_weights['h3']), mlp_biases['b3'])
            #layer_3 = tf.nn.relu(layer_3)
            #layer_3 = tf.nn.dropout(layer_3, mlp_dropout)
            out_layer = tf.matmul(layer_2, mlp_weights['out'] + mlp_biases['out'])
        return out_layer


    def baseline(x):
        with tf.name_scope('baseline'):
            tmp = tf.add(-20*x[:,1], 40*x[:,2], name='add1')
            tmp = tf.add(-20*x[:,0], tmp, name='add2')
        return tmp

    #with tf.get_default_graph().device("/gpu:0"):
    with tf.name_scope('stddev'):
        mlp_fp = mlp_t2(X,mlp_keepprob)
    with tf.name_scope('cost'):
        mlp_cost = tf.reduce_mean(tf.pow(Y - mlp_fp , 2))

    baseline1 = baseline(X)
    baseline_cost = tf.reduce_mean(tf.pow(Z - baseline1, 2))
    mlp_train_optimizer = tf.train.AdamOptimizer(learning_rate=mlp_learning_rate).minimize(mlp_cost)
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    initall = tf.initialize_all_variables()
    ## traindata:
    data = mydata.load(TRAIN_DATA)

    #data.setscountlim(mlp_shuffle_count)
    mlp_batchsize = int(data.size/mlp_nbatchs)

    # normalize data if necessary:
    if normalize:
        data.normalize()
        print(data.x[1:10,:])

    with tf.Session() as sess:
        if verbose:
            print("Original X: ")
            print(data.x[1])
            print("Original Y: ")
            print(data.y[1])

        if init_train:
            sess.run(initall)
            costlist = []
            writer.add_graph(sess.graph)
            for epoch in range(mlp_train_epochs):
                ctotal = 0
                for i in range(mlp_nbatchs):
                    batch_xs, batch_ys = data.nextbatch(mlp_batchsize)
                    _,c = sess.run([mlp_train_optimizer, mlp_cost],
                     feed_dict={X:batch_xs, Y:batch_ys, mlp_keepprob:mlp_dropout})
                    ctotal += c
                c = ctotal/mlp_nbatchs
                costlist.append(c)
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch),
                          "cost=", "{:.8f}".format(c))
                # break if the cost lower than the limit
                if c < breaklim:
                    break

            if plot:
                plt.plot(np.arange(100,len(costlist)),costlist[100:])
                plt.legend(['Training cost'])
                plt.suptitle(plotTitle)
                plt.xlabel('epoch')
                plt.ylabel('cost(L2)')
                plt.show()
            print("Optimization Finished!")
            print("Final train cost: {:.8f}".format(c))
            save_path = saver.save(sess,"./models/%s_train.ckpt" % modelname)
            print("Initial mlp training model saved in file: %s" % save_path)
            #summary = sess.run(merged)
            #writer.add_summary(summary, 0)

        if ctrain:
            if os.path.isfile("./models/%s_ctrain.ckpt.meta" % modelname) and ctrain_use_train_model == 0:
                saver.restore(sess,"./models/%s_ctrain.ckpt" % modelname)
                print("restored: ./models/%s_ctrain.ckpt" % modelname)
            else:
                saver.restore(sess,"./models/%s_train.ckpt" % modelname)
                print("restored: ./models/%s_train.ckpt" % modelname)

            for epoch in range(mlp_train_epochs):
                ctotal = 0
                for i in range(mlp_nbatchs):
                    batch_xs, batch_ys = data.nextbatch(mlp_batchsize)
                    _,c = sess.run([mlp_ctrain_optimizer, mlp_cost], feed_dict={X:batch_xs, Z:batch_ys, mlp_keepprob:mlp_dropout})
                    ctotal += c
                c = ctotal/mlp_nbatchs
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch),
                          "cost=", "{:.8f}".format(c))

            #plt.plot(costlist)
            print("Optimization Finished!")
            print("Final train cost: {:.8f}".format(c))
            save_path = saver.save(sess,"./models/%s_ctrain.ckpt" % modelname)
            print("Initial mlp training model saved in file: %s" % save_path)

        ## Forward propagation:
        if fp:
            if fp_test:
                testdata = mydata.load(TEST_DATA)
                # set normalization
                if normalize:
                    testdata.x = data.setnormalize(testdata.x)
                print("Test dataset loaded")
            if os.path.isfile("./models/%s_ctrain.ckpt.meta" % modelname):
                saver.restore(sess,"./models/%s_ctrain.ckpt" % modelname)
                print("restored: ./models/%s_ctrain.ckpt" % modelname)
            else:
                saver.restore(sess,"./models/%s_train.ckpt" % modelname)
                print("restored: ./models/%s_train.ckpt" % modelname)
            c = sess.run(mlp_cost, feed_dict={X:testdata.x, Y:testdata.y, mlp_keepprob:1.0})
            d = sess.run(baseline_cost, feed_dict={X:testdata.x, Z:(testdata.y).reshape(1,-1)})
            #print(sess.run(mlp_fp, feed_dict={X:testdata.x, mlp_keepprob:1.0}))
            print("The model cost is: {:.8f}".format(c))
            print("The baseline cost is: {:.8f}".format(d))
            start = time.time()
            pred = sess.run(mlp_fp, feed_dict = {X:testdata.x, mlp_keepprob:1.0})
            end = time.time()
            extime = end - start
            baselinepred = sess.run(baseline1, feed_dict = {X:testdata.x})
            # restore normalization
            if normalize:
                testdata.x = data.setdenormalize(testdata.x)
            # plot the testdata
            if plot:
                # histogram
                diff = testdata.y - pred
                n, bins, patches = plt.hist(diff, 30, density = True, facecolor='b', alpha=0.75)
                plt.xlabel('Difference in dB')
                plt.ylabel('Probability')
                plt.title('Streetscene test prediction error distribution')
                #plt.title('Urban canyon test prediction error distribution')
                #plt.title('Two ray test prediction error distribution')
                plt.axis([min(diff), max(diff), 0, 0.15])
                plt.grid(True)
                plt.show()

                # plt.plot(testdata.x[plotrangel:plotrangeh,2],testdata.y[plotrangel:plotrangeh],
                #     'r^', testdata.x[plotrangel:plotrangeh,2],pred[plotrangel:plotrangeh],'b^')
                # plt.legend(['True', 'Predicted'])
                # plt.suptitle(plotTitle + ' test testdata result')
                # plt.xlabel('log10(d)')
                # plt.ylabel('negative signal strength in db')
                # plt.show()
            save2 = pickle.dump(pred, open(('./result/' + savepath_nnpred), 'wb'))
            writeToFile(testdata.x, testdata.y, pred, baselinepred, './result/result')
            print("Result is saved in the result folder")
            save_path = saver.save(sess,"./models/%s.ckpt" % modelname)
            print("Finalized model saved in file: %s" % save_path)
            return [pred,np.sqrt(c)]
    return 0



def mlp3L(modelname, mlp_ninput = 3, testmode = 0, epochs = 1000, breaklim = 0, plot = 0,
mlp_learning_rate = 0.05, traindata = 'traindata', testdata = 'testdata',
plotrangel = 0, plotrangeh = 100, display_step = 20, mlp_nbatchs = 3, mlp_nhidden1 = 50,
mlp_nhidden2 = 40, mlp_nhidden3 = 30, plotTitle = '900MHz two ray model', normalize = False):

    #tf.set_random_seed(1)
    # structural param
    mlp_noutput = 1
    mlp_dropout = 1
    # training param
    mlp_momentum = 0.2
    mlp_train_epochs = epochs
    mlp_shuffle_count = 50

    # data and path param
    TRAIN_DATA = traindata
    TEST_DATA = testdata
    savepath_b1pred = 'baseline_1'
    savepath_nnpred = 'nnpred_1'
    verbose = 1 # display the original and baseline data

    # stage param
    #testmode = 0 # 1:Load and test only
    fp_test = 1 # fp function will test data after training
    ctrain_use_models = 0 # Note: it will continue learning from train model instead of the previous ctrain model

    if testmode == 1:
        traindata = 1
        init_train = 0
        ctrain = 0
        fp = 1
    else:
        traindata = 1
        init_train = 1
        ctrain = 0
        fp = 1

    writer = tf.summary.FileWriter("./tb/fm/model1")

    ### Tensors:
    # mlp tensors
    mlp_weights = {
        'h1': tf.Variable(tf.random_normal([mlp_ninput, mlp_nhidden1],stddev=0.05, dtype=tf.float32, name='h1')),
        'h2': tf.Variable(tf.random_normal([mlp_nhidden1, mlp_nhidden2],stddev=0.05, dtype=tf.float32, name='h2')),
        'h3': tf.Variable(tf.random_normal([mlp_nhidden2, mlp_nhidden3],stddev=0.05, dtype=tf.float32, name='h3')),
        'out': tf.Variable(tf.random_normal([mlp_nhidden3, mlp_noutput],stddev=0.05, dtype=tf.float32, name='wout'))
    }
    mlp_biases = {
        'b1': tf.Variable(tf.random_normal([mlp_nhidden1],stddev=0.05, dtype=tf.float32, name='b1')),
        'b2': tf.Variable(tf.random_normal([mlp_nhidden2],stddev=0.05, dtype=tf.float32, name='b2')),
        'b3': tf.Variable(tf.random_normal([mlp_nhidden3],stddev=0.05, dtype=tf.float32, name='b3')),
        'out': tf.Variable(tf.random_normal([mlp_noutput],stddev=0.05, dtype=tf.float32, name='bout'))
    }
    # placeholders
    X = tf.placeholder("float", [None, mlp_ninput])
    Y = tf.placeholder("float", [None, 1])
    Z = tf.placeholder("float", [1,None])
    mlp_keepprob = tf.placeholder("float")
    ### operators:
    def mlp_t2(x, mlp_dropout):
        with tf.name_scope('mlp'):
            layer_1 = tf.add(tf.matmul(x, mlp_weights['h1']), mlp_biases['b1'])
            #layer_1 = tf.nn.relu(layer_1)
            layer_1 = swish(layer_1)

            layer_2 = tf.add(tf.matmul(layer_1, mlp_weights['h2']), mlp_biases['b2'])
            #layer_2 = tf.nn.relu(layer_2)
            layer_2 = swish(layer_2)
            layer_2 = tf.nn.dropout(layer_2, mlp_dropout)

            layer_3 = tf.add(tf.matmul(layer_2, mlp_weights['h3']), mlp_biases['b3'])
            layer_3 = swish(layer_3)
            #layer_3 = tf.nn.relu(layer_3)
            layer_3 = tf.nn.dropout(layer_3, mlp_dropout)
            out_layer = tf.matmul(layer_3, mlp_weights['out'] + mlp_biases['out'])
        return out_layer


    def baseline(x):
        with tf.name_scope('baseline'):
            tmp = tf.add(-20*x[:,1], 40*x[:,2], name='add1')
            tmp = tf.add(-20*x[:,0], tmp, name='add2')
        return tmp

    #with tf.get_default_graph().device("/gpu:0"):
    with tf.name_scope('stddev'):
        mlp_fp = mlp_t2(X,mlp_keepprob)
    with tf.name_scope('cost'):
        mlp_cost = tf.reduce_mean(tf.pow(Y - mlp_fp , 2))

    baseline1 = baseline(X)
    baseline_cost = tf.reduce_mean(tf.pow(Z - baseline1, 2))
    mlp_train_optimizer = tf.train.AdamOptimizer(learning_rate=mlp_learning_rate).minimize(mlp_cost)
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    initall = tf.initialize_all_variables()
    #initall = tf.global_variables_initializer()

    ## traindata:
    data = mydata.load(TRAIN_DATA)

    #data.setscountlim(mlp_shuffle_count)
    mlp_batchsize = int(data.size/mlp_nbatchs)

    # normalize data if necessary:
    if normalize:
        data.normalize()
        print(data.x[1:5,:])

    with tf.Session() as sess:
        if verbose:
            print("Original X: ")
            print(data.x[1])
            print("Original Y: ")
            print(data.y[1])

        if init_train:
            sess.run(initall)
            costlist = []
            writer.add_graph(sess.graph)
            for epoch in range(mlp_train_epochs):
                ctotal = 0
                for i in range(mlp_nbatchs):
                    batch_xs, batch_ys = data.nextbatch(mlp_batchsize)
                    _,c = sess.run([mlp_train_optimizer, mlp_cost],
                     feed_dict={X:batch_xs, Y:batch_ys, mlp_keepprob:mlp_dropout})
                    ctotal += c
                c = ctotal/mlp_nbatchs
                costlist.append(c)
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch),
                          "cost=", "{:.8f}".format(c))
                # break if the cost lower than the limit
                if c < breaklim:
                    break

            if plot:
                plt.plot(np.arange(100,len(costlist)),costlist[100:])
                plt.legend(['Training cost'])
                plt.suptitle(plotTitle)
                plt.xlabel('epoch')
                plt.ylabel('cost(L2)')
                plt.show()
            print("Optimization Finished!")
            print("Final train cost: {:.8f}".format(c))
            save_path = saver.save(sess,"./models/%s_train.ckpt" % modelname)
            print("Initial mlp training model saved in file: %s" % save_path)
            #summary = sess.run(merged)
            #writer.add_summary(summary, 0)

        if ctrain:
            if os.path.isfile("./models/%s_ctrain.ckpt.meta" % modelname) and ctrain_use_train_model == 0:
                saver.restore(sess,"./models/%s_ctrain.ckpt" % modelname)
                print("restored: ./models/%s_ctrain.ckpt" % modelname)
            else:
                saver.restore(sess,"./models/%s_train.ckpt" % modelname)
                print("restored: ./models/%s_train.ckpt" % modelname)

            for epoch in range(mlp_train_epochs):
                ctotal = 0
                for i in range(mlp_nbatchs):
                    batch_xs, batch_ys = data.nextbatch(mlp_batchsize)
                    _,c = sess.run([mlp_ctrain_optimizer, mlp_cost], feed_dict={X:batch_xs, Z:batch_ys, mlp_keepprob:mlp_dropout})
                    ctotal += c
                c = ctotal/mlp_nbatchs
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch),
                          "cost=", "{:.8f}".format(c))

            #plt.plot(costlist)
            print("Optimization Finished!")
            print("Final train cost: {:.8f}".format(c))
            save_path = saver.save(sess,"./models/%s_ctrain.ckpt" % modelname)
            print("Initial mlp training model saved in file: %s" % save_path)

        ## Forward propagation:
        if fp:
            if fp_test:
                testdata = mydata.load(TEST_DATA)
                # set normalization
                if normalize:
                    testdata.x = data.setnormalize(testdata.x)
                print("Test dataset loaded")
            if os.path.isfile("./models/%s_ctrain.ckpt.meta" % modelname):
                saver.restore(sess,"./models/%s_ctrain.ckpt" % modelname)
                print("restored: ./models/%s_ctrain.ckpt" % modelname)
            else:
                saver.restore(sess,"./models/%s_train.ckpt" % modelname)
                print("restored: ./models/%s_train.ckpt" % modelname)
            c = sess.run(mlp_cost, feed_dict={X:testdata.x, Y:testdata.y, mlp_keepprob:1.0})
            d = sess.run(baseline_cost, feed_dict={X:testdata.x, Z:(testdata.y).reshape(1,-1)})
            #print(sess.run(mlp_fp, feed_dict={X:testdata.x, mlp_keepprob:1.0}))
            print("The model cost is: {:.8f}".format(c))
            print("The baseline cost is: {:.8f}".format(d))
            start = time.time()
            pred = sess.run(mlp_fp, feed_dict = {X:testdata.x, mlp_keepprob:1.0})
            end = time.time()
            extime = end - start
            baselinepred = sess.run(baseline1, feed_dict = {X:testdata.x})
            # restore normalization
            if normalize:
                testdata.x = data.setdenormalize(testdata.x)
            # plot the testdata
            if plot:
                plt.plot(testdata.x[plotrangel:plotrangeh,2],testdata.y[plotrangel:plotrangeh],
                    'r^', testdata.x[plotrangel:plotrangeh,2],pred[plotrangel:plotrangeh],'b^')
                plt.legend(['True', 'Predicted'])
                plt.suptitle(plotTitle + ' test testdata result')
                plt.xlabel('log10(d)')
                plt.ylabel('negative signal strength in db')
                plt.show()
            save2 = pickle.dump(pred, open(('./result/' + savepath_nnpred), 'wb'))
            writeToFile(testdata.x, testdata.y, pred, baselinepred, './result/result')
            print("Result is saved in the result folder")
            save_path = saver.save(sess,"./models/%s.ckpt" % modelname)
            print("Finalized model saved in file: %s" % save_path)
            return [pred,np.sqrt(c)]
    return 0


def mlp2L(modelname, mlp_ninput = 3, testmode = 0, epochs = 1000, breaklim = 0, plot = 0,
mlp_learning_rate = 0.05, traindata = 'traindata', testdata = 'testdata',
plotrangel = 0, plotrangeh = 100, display_step = 20, mlp_nbatchs = 3, mlp_nhidden1 = 50,
mlp_nhidden2 = 40, plotTitle = '900MHz two ray model', normalize = False):

    #tf.set_random_seed(1)
    # structural param
    mlp_noutput = 1
    mlp_dropout = 1
    # training param
    mlp_momentum = 0.2
    mlp_train_epochs = epochs
    mlp_shuffle_count = 50

    # data and path param
    TRAIN_DATA = traindata
    TEST_DATA = testdata
    savepath_b1pred = 'baseline_1'
    savepath_nnpred = 'nnpred_1'
    verbose = 1 # display the original and baseline data

    # stage param
    #testmode = 0 # 1:Load and test only
    fp_test = 1 # fp function will test data after training
    ctrain_use_models = 0 # Note: it will continue learning from train model instead of the previous ctrain model

    if testmode == 1:
        traindata = 1
        init_train = 0
        ctrain = 0
        fp = 1
    else:
        traindata = 1
        init_train = 1
        ctrain = 0
        fp = 1

    writer = tf.summary.FileWriter("./tb/fm/model1")

    ### Tensors:
    # mlp tensors
    mlp_weights = {
        'h1': tf.Variable(tf.random_normal([mlp_ninput, mlp_nhidden1],stddev=0.05, dtype=tf.float32, name='h1')),
        'h2': tf.Variable(tf.random_normal([mlp_nhidden1, mlp_nhidden2],stddev=0.05, dtype=tf.float32, name='h2')),
        'out': tf.Variable(tf.random_normal([mlp_nhidden2, mlp_noutput],stddev=0.05, dtype=tf.float32, name='wout'))
    }
    mlp_biases = {
        'b1': tf.Variable(tf.random_normal([mlp_nhidden1],stddev=0.05, dtype=tf.float32, name='b1')),
        'b2': tf.Variable(tf.random_normal([mlp_nhidden2],stddev=0.05, dtype=tf.float32, name='b2')),
        'out': tf.Variable(tf.random_normal([mlp_noutput],stddev=0.05, dtype=tf.float32, name='bout'))
    }
    # placeholders
    X = tf.placeholder("float", [None, mlp_ninput])
    Y = tf.placeholder("float", [None, 1])
    Z = tf.placeholder("float", [1,None])
    mlp_keepprob = tf.placeholder("float")
    ### operators:
    def mlp_t2(x, mlp_dropout):
        with tf.name_scope('mlp'):
            layer_1 = tf.add(tf.matmul(x, mlp_weights['h1']), mlp_biases['b1'])
            #layer_1 = tf.nn.relu(layer_1)
            layer_1 = swish(layer_1)
            layer_2 = tf.add(tf.matmul(layer_1, mlp_weights['h2']), mlp_biases['b2'])
            #layer_2 = tf.nn.relu(layer_2)
            layer_2 = swish(layer_2)
            layer_2 = tf.nn.dropout(layer_2, mlp_dropout)
            #layer_3 = tf.add(tf.matmul(layer_2, mlp_weights['h3']), mlp_biases['b3'])
            #layer_3 = tf.nn.relu(layer_3)
            #layer_3 = tf.nn.dropout(layer_3, mlp_dropout)
            out_layer = tf.matmul(layer_2, mlp_weights['out'] + mlp_biases['out'])
        return out_layer


    def baseline(x):
        with tf.name_scope('baseline'):
            tmp = tf.add(-20*x[:,1], 40*x[:,2], name='add1')
            tmp = tf.add(-20*x[:,0], tmp, name='add2')
        return tmp

    #with tf.get_default_graph().device("/gpu:0"):
    with tf.name_scope('stddev'):
        mlp_fp = mlp_t2(X,mlp_keepprob)
    with tf.name_scope('cost'):
        mlp_cost = tf.reduce_mean(tf.pow(Y - mlp_fp , 2))

    baseline1 = baseline(X)
    baseline_cost = tf.reduce_mean(tf.pow(Z - baseline1, 2))
    #tmp = tf.reduce_mean(tf.pow(Y - baseline , 2),axis=0)
    #mlp_bcost1 = tf.reduce_mean(tf.pow(Y - baseline1 , 2))
    #mlp_bcost2 = tf.reduce_mean(tf.pow(Y , 2))
    #mlp_ctrain_optimizer = tf.train.MomentumOptimizer(learning_rate=mlp_learning_rate, momentum=mlp_momentum).minimize(mlp_cost)
    #mlp_train_optimizer = tf.train.RMSPropOptimizer(learning_rate=mlp_learning_rate).minimize(mlp_cost)
    mlp_train_optimizer = tf.train.AdamOptimizer(learning_rate=mlp_learning_rate).minimize(mlp_cost)
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    initall = tf.initialize_all_variables()
    #initall = tf.global_variables_initializer()

    ## traindata:
    data = mydata.load(TRAIN_DATA)

    #data.setscountlim(mlp_shuffle_count)
    mlp_batchsize = int(data.size/mlp_nbatchs)

    # normalize data if necessary:
    if normalize:
        data.normalize()
        print(data.x[1:10,:])

    with tf.Session() as sess:
        if verbose:
            print("Original X: ")
            print(data.x[1])
            print("Original Y: ")
            print(data.y[1])

        if init_train:
            sess.run(initall)
            costlist = []
            writer.add_graph(sess.graph)
            for epoch in range(mlp_train_epochs):
                ctotal = 0
                for i in range(mlp_nbatchs):
                    batch_xs, batch_ys = data.nextbatch(mlp_batchsize)
                    _,c = sess.run([mlp_train_optimizer, mlp_cost],
                     feed_dict={X:batch_xs, Y:batch_ys, mlp_keepprob:mlp_dropout})
                    ctotal += c
                c = ctotal/mlp_nbatchs
                costlist.append(c)
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch),
                          "cost=", "{:.8f}".format(c))
                # break if the cost lower than the limit
                if c < breaklim:
                    break

            if plot:
                plt.plot(np.arange(100,len(costlist)),costlist[100:])
                plt.legend(['Training cost'])
                plt.suptitle(plotTitle)
                plt.xlabel('epoch')
                plt.ylabel('cost(L2)')
                plt.show()
            print("Optimization Finished!")
            print("Final train cost: {:.8f}".format(c))
            save_path = saver.save(sess,"./models/%s_train.ckpt" % modelname)
            print("Initial mlp training model saved in file: %s" % save_path)
            #summary = sess.run(merged)
            #writer.add_summary(summary, 0)

        if ctrain:
            if os.path.isfile("./models/%s_ctrain.ckpt.meta" % modelname) and ctrain_use_train_model == 0:
                saver.restore(sess,"./models/%s_ctrain.ckpt" % modelname)
                print("restored: ./models/%s_ctrain.ckpt" % modelname)
            else:
                saver.restore(sess,"./models/%s_train.ckpt" % modelname)
                print("restored: ./models/%s_train.ckpt" % modelname)

            for epoch in range(mlp_train_epochs):
                ctotal = 0
                for i in range(mlp_nbatchs):
                    batch_xs, batch_ys = data.nextbatch(mlp_batchsize)
                    _,c = sess.run([mlp_ctrain_optimizer, mlp_cost], feed_dict={X:batch_xs, Z:batch_ys, mlp_keepprob:mlp_dropout})
                    ctotal += c
                c = ctotal/mlp_nbatchs
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch),
                          "cost=", "{:.8f}".format(c))

            #plt.plot(costlist)
            print("Optimization Finished!")
            print("Final train cost: {:.8f}".format(c))
            save_path = saver.save(sess,"./models/%s_ctrain.ckpt" % modelname)
            print("Initial mlp training model saved in file: %s" % save_path)

        ## Forward propagation:
        if fp:
            if fp_test:
                testdata = mydata.load(TEST_DATA)
                # set normalization
                if normalize:
                    testdata.x = data.setnormalize(testdata.x)
                print("Test dataset loaded")
            if os.path.isfile("./models/%s_ctrain.ckpt.meta" % modelname):
                saver.restore(sess,"./models/%s_ctrain.ckpt" % modelname)
                print("restored: ./models/%s_ctrain.ckpt" % modelname)
            else:
                saver.restore(sess,"./models/%s_train.ckpt" % modelname)
                print("restored: ./models/%s_train.ckpt" % modelname)
            c = sess.run(mlp_cost, feed_dict={X:testdata.x, Y:testdata.y, mlp_keepprob:1.0})
            d = sess.run(baseline_cost, feed_dict={X:testdata.x, Z:(testdata.y).reshape(1,-1)})
            #print(sess.run(mlp_fp, feed_dict={X:testdata.x, mlp_keepprob:1.0}))
            print("The model cost is: {:.8f}".format(c))
            print("The baseline cost is: {:.8f}".format(d))
            start = time.time()
            pred = sess.run(mlp_fp, feed_dict = {X:testdata.x, mlp_keepprob:1.0})
            end = time.time()
            extime = end - start
            baselinepred = sess.run(baseline1, feed_dict = {X:testdata.x})
            # restore normalization
            if normalize:
                testdata.x = data.setdenormalize(testdata.x)
            # plot the testdata
            if plot:
                plt.plot(testdata.x[plotrangel:plotrangeh,2],testdata.y[plotrangel:plotrangeh],
                    'r^', testdata.x[plotrangel:plotrangeh,2],pred[plotrangel:plotrangeh],'b^')
                plt.legend(['True', 'Predicted'])
                plt.suptitle(plotTitle + ' test testdata result')
                plt.xlabel('log10(d)')
                plt.ylabel('negative signal strength in db')
                plt.show()
            save2 = pickle.dump(pred, open(('./result/' + savepath_nnpred), 'wb'))
            writeToFile(testdata.x, testdata.y, pred, baselinepred, './result/result')
            print("Result is saved in the result folder")
            save_path = saver.save(sess,"./models/%s.ckpt" % modelname)
            print("Finalized model saved in file: %s" % save_path)
            return [pred,np.sqrt(c)]
    return 0


def mlp1L(modelname, mlp_ninput = 3, testmode = 0, epochs = 1000, breaklim = 0, plot = 0,
mlp_learning_rate = 0.05, traindata = 'traindata', testdata = 'testdata',
plotrangel = 0, plotrangeh = 100, display_step = 20, mlp_nbatchs = 3, mlp_nhidden1 = 50,
plotTitle = '900MHz two ray model', normalize = False):

    #tf.set_random_seed(1)
    # structural param
    mlp_noutput = 1
    mlp_dropout = 1
    # training param
    mlp_momentum = 0.2
    mlp_train_epochs = epochs
    mlp_shuffle_count = 50

    # data and path param
    TRAIN_DATA = traindata
    TEST_DATA = testdata
    savepath_b1pred = 'baseline_1'
    savepath_nnpred = 'nnpred_1'
    verbose = 1 # display the original and baseline data

    # stage param
    #testmode = 0 # 1:Load and test only
    fp_test = 1 # fp function will test data after training
    ctrain_use_models = 0 # Note: it will continue learning from train model instead of the previous ctrain model

    if testmode == 1:
        traindata = 1
        init_train = 0
        ctrain = 0
        fp = 1
    else:
        traindata = 1
        init_train = 1
        ctrain = 0
        fp = 1

    writer = tf.summary.FileWriter("./tb/fm/model1")

    ### Tensors:
    # mlp tensors
    mlp_weights = {
        'h1': tf.Variable(tf.random_normal([mlp_ninput, mlp_nhidden1],stddev=0.05, dtype=tf.float32, name='h1')),
        'out': tf.Variable(tf.random_normal([mlp_nhidden1, mlp_noutput],stddev=0.05, dtype=tf.float32, name='wout'))
    }
    mlp_biases = {
        'b1': tf.Variable(tf.random_normal([mlp_nhidden1],stddev=0.05, dtype=tf.float32, name='b1')),
        'out': tf.Variable(tf.random_normal([mlp_noutput],stddev=0.05, dtype=tf.float32, name='bout'))
    }
    # placeholders
    X = tf.placeholder("float", [None, mlp_ninput])
    Y = tf.placeholder("float", [None, 1])
    Z = tf.placeholder("float", [1,None])
    mlp_keepprob = tf.placeholder("float")
    ### operators:
    def mlp_t2(x, mlp_dropout):
        with tf.name_scope('mlp'):
            layer_1 = tf.add(tf.matmul(x, mlp_weights['h1']), mlp_biases['b1'])
            layer_1 = swish(layer_1)
            layer_2 = tf.nn.dropout(layer_1, mlp_dropout)

            out_layer = tf.matmul(layer_1, mlp_weights['out'] + mlp_biases['out'])
        return out_layer


    def baseline(x):
        with tf.name_scope('baseline'):
            tmp = tf.add(-20*x[:,1], 40*x[:,2], name='add1')
            tmp = tf.add(-20*x[:,0], tmp, name='add2')
        return tmp

    #with tf.get_default_graph().device("/gpu:0"):
    with tf.name_scope('stddev'):
        mlp_fp = mlp_t2(X,mlp_keepprob)
    with tf.name_scope('cost'):
        mlp_cost = tf.reduce_mean(tf.pow(Y - mlp_fp , 2))

    baseline1 = baseline(X)
    baseline_cost = tf.reduce_mean(tf.pow(Z - baseline1, 2))
    #tmp = tf.reduce_mean(tf.pow(Y - baseline , 2),axis=0)
    #mlp_bcost1 = tf.reduce_mean(tf.pow(Y - baseline1 , 2))
    #mlp_bcost2 = tf.reduce_mean(tf.pow(Y , 2))
    #mlp_ctrain_optimizer = tf.train.MomentumOptimizer(learning_rate=mlp_learning_rate, momentum=mlp_momentum).minimize(mlp_cost)
    #mlp_train_optimizer = tf.train.RMSPropOptimizer(learning_rate=mlp_learning_rate).minimize(mlp_cost)
    mlp_train_optimizer = tf.train.AdamOptimizer(learning_rate=mlp_learning_rate).minimize(mlp_cost)
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    initall = tf.initialize_all_variables()
    #initall = tf.global_variables_initializer()

    ## traindata:
    data = mydata.load(TRAIN_DATA)

    #data.setscountlim(mlp_shuffle_count)
    mlp_batchsize = int(data.size/mlp_nbatchs)

    # normalize data if necessary:
    if normalize:
        data.normalize()
        print(data.x[1:10,:])

    with tf.Session() as sess:
        if verbose:
            print("Original X: ")
            print(data.x[1])
            print("Original Y: ")
            print(data.y[1])

        if init_train:
            sess.run(initall)
            costlist = []
            writer.add_graph(sess.graph)
            for epoch in range(mlp_train_epochs):
                ctotal = 0
                for i in range(mlp_nbatchs):
                    batch_xs, batch_ys = data.nextbatch(mlp_batchsize)
                    _,c = sess.run([mlp_train_optimizer, mlp_cost],
                     feed_dict={X:batch_xs, Y:batch_ys, mlp_keepprob:mlp_dropout})
                    ctotal += c
                c = ctotal/mlp_nbatchs
                costlist.append(c)
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch),
                          "cost=", "{:.8f}".format(c))
                # break if the cost lower than the limit
                if c < breaklim:
                    break

            if plot:
                plt.plot(np.arange(100,len(costlist)),costlist[100:])
                plt.legend(['Training cost'])
                plt.suptitle(plotTitle)
                plt.xlabel('epoch')
                plt.ylabel('cost(L2)')
                plt.show()
            print("Optimization Finished!")
            print("Final train cost: {:.8f}".format(c))
            save_path = saver.save(sess,"./models/%s_train.ckpt" % modelname)
            print("Initial mlp training model saved in file: %s" % save_path)
            #summary = sess.run(merged)
            #writer.add_summary(summary, 0)

        if ctrain:
            if os.path.isfile("./models/%s_ctrain.ckpt.meta" % modelname) and ctrain_use_train_model == 0:
                saver.restore(sess,"./models/%s_ctrain.ckpt" % modelname)
                print("restored: ./models/%s_ctrain.ckpt" % modelname)
            else:
                saver.restore(sess,"./models/%s_train.ckpt" % modelname)
                print("restored: ./models/%s_train.ckpt" % modelname)

            for epoch in range(mlp_train_epochs):
                ctotal = 0
                for i in range(mlp_nbatchs):
                    batch_xs, batch_ys = data.nextbatch(mlp_batchsize)
                    _,c = sess.run([mlp_ctrain_optimizer, mlp_cost], feed_dict={X:batch_xs, Z:batch_ys, mlp_keepprob:mlp_dropout})
                    ctotal += c
                c = ctotal/mlp_nbatchs
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch),
                          "cost=", "{:.8f}".format(c))

            #plt.plot(costlist)
            print("Optimization Finished!")
            print("Final train cost: {:.8f}".format(c))
            save_path = saver.save(sess,"./models/%s_ctrain.ckpt" % modelname)
            print("Initial mlp training model saved in file: %s" % save_path)

        ## Forward propagation:
        if fp:
            if fp_test:
                testdata = mydata.load(TEST_DATA)
                # set normalization
                if normalize:
                    testdata.x = data.setnormalize(testdata.x)
                print("Test dataset loaded")
            if os.path.isfile("./models/%s_ctrain.ckpt.meta" % modelname):
                saver.restore(sess,"./models/%s_ctrain.ckpt" % modelname)
                print("restored: ./models/%s_ctrain.ckpt" % modelname)
            else:
                saver.restore(sess,"./models/%s_train.ckpt" % modelname)
                print("restored: ./models/%s_train.ckpt" % modelname)
            c = sess.run(mlp_cost, feed_dict={X:testdata.x, Y:testdata.y, mlp_keepprob:1.0})
            d = sess.run(baseline_cost, feed_dict={X:testdata.x, Z:(testdata.y).reshape(1,-1)})
            #print(sess.run(mlp_fp, feed_dict={X:testdata.x, mlp_keepprob:1.0}))
            print("The model cost is: {:.8f}".format(c))
            print("The baseline cost is: {:.8f}".format(d))
            start = time.time()
            pred = sess.run(mlp_fp, feed_dict = {X:testdata.x, mlp_keepprob:1.0})
            end = time.time()
            extime = end - start
            baselinepred = sess.run(baseline1, feed_dict = {X:testdata.x})
            # restore normalization
            if normalize:
                testdata.x = data.setdenormalize(testdata.x)
            # plot the testdata
            if plot:
                plt.plot(testdata.x[plotrangel:plotrangeh,2],testdata.y[plotrangel:plotrangeh],
                    'r^', testdata.x[plotrangel:plotrangeh,2],pred[plotrangel:plotrangeh],'b^')
                plt.legend(['True', 'Predicted'])
                plt.suptitle(plotTitle + ' test testdata result')
                plt.xlabel('log10(d)')
                plt.ylabel('negative signal strength in db')
                plt.show()
            save2 = pickle.dump(pred, open(('./result/' + savepath_nnpred), 'wb'))
            writeToFile(testdata.x, testdata.y, pred, baselinepred, './result/result')
            print("Result is saved in the result folder")
            save_path = saver.save(sess,"./models/%s.ckpt" % modelname)
            print("Finalized model saved in file: %s" % save_path)
            return [pred,np.sqrt(c)]
    return 0
