#!/usr/bin/env python
'''Validates a converted ImageNet model against the ILSVRC12 validation set.'''

import argparse
import numpy as np
import tensorflow as tf
import os.path as osp
import time

import models
import dataset


def load_model(name):
    '''Creates and returns an instance of the model given its class name.
    The created model has a single placeholder node for feeding images.
    '''
    # Find the model class from its name
    all_models = models.get_models()
    lut = {model.__name__: model for model in all_models}
    if name not in lut:
        print('Invalid model index. Options are:')
        # Display a list of valid model names
        for model in all_models:
            print('\t* {}'.format(model.__name__))
        return None
    NetClass = lut[name]

    # Create a placeholder for the input image
    spec = models.get_data_spec(model_class=NetClass)
    data_node = tf.placeholder(tf.float32,
                               shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct and return the model
    return NetClass({'data': data_node})


def validate(net, model_path, image_producer, model_name, top_k=5):
    '''Compute the top_k classification accuracy for the given network and images.'''
    # Get the data specifications for given network
    spec = models.get_data_spec(model_instance=net)
    # Get the input node for feeding in the images
    input_node = net.inputs['data']
    # Create a placeholder for the ground truth labels
    label_node = tf.placeholder(tf.int32)
    # Get the output of the network (class probabilities)
    probs = net.get_output()
    compute_ops = {}
    threshold = 50
    # Sparisty in conv layer inputs
    total_ops = 0
    for lname, op_count in net.layer_op_counts.iteritems():
        layer_inputs = net.layer_inputs[lname]
        assert(len(layer_inputs) == 1)
        for lin in layer_inputs:
            zero_actv = tf.less(tf.abs(lin), threshold)
            num_zeroes = tf.reduce_sum(tf.cast(zero_actv, tf.int32))
            sparsity = tf.div(tf.cast(num_zeroes, tf.float32),
                              tf.cast(tf.size(lin), tf.float32))

            compute_ops[lname] = sparsity
            compute_ops[lname + '_min'] = tf.reduce_min(lin)
            compute_ops[lname + '_max'] = tf.reduce_max(lin)

        total_ops = total_ops + op_count

    """
    for lname in sorted(net.layer_op_counts, key=net.layer_op_counts.get, reverse=True):
        op_count = net.layer_op_counts[lname]
        print("%s %.3f" %(lname, float(op_count)/total_ops))
    """
    # Create a top_k accuracy node
    top_k_op = tf.nn.in_top_k(probs, label_node, top_k)
    compute_ops['top_k'] = top_k_op
    # The number of images processed
    count = 0
    # The number of correctly classified images
    correct = 0
    # The total number of images
    total = len(image_producer)
    #merged = tf.merge_all_summaries()
    sparsity = {}
    input_min = {}
    input_max = {}
    for lname, ops in net.layer_op_counts.iteritems():
        sparsity[lname] = 0
        input_max[lname] = float('-inf')
        input_min[lname] = float('inf')

    with tf.Session() as sesh:
        #writer = tf.train.SummaryWriter('/tmp/' + model_name, graph = sesh.graph)
        coordinator = tf.train.Coordinator()
        # Load the converted parameters
        net.load(data_path=model_path, session=sesh)
        # Start the image processing workers
        threads = image_producer.start(session=sesh, coordinator=coordinator)
        # Iterate over and classify mini-batches
        batch_num = 0
        for (labels, images) in image_producer.batches(sesh):
            start = time.time()
            """
            summary, top_k_res = sesh.run([top_k_op],
                                          feed_dict={input_node: images,
                                                     label_node: labels})
            """
            out_dict = sesh.run(compute_ops, feed_dict={input_node: images,
                                                        label_node: labels})

            for lname, ops in net.layer_op_counts.iteritems():
                sparsity[lname] = sparsity[lname] + out_dict[lname]
                input_max[lname] = max(input_max[lname], out_dict[lname + '_max'])
                input_min[lname] = min(input_min[lname], out_dict[lname + '_min'])

            correct += np.sum(out_dict['top_k'])
            print('Inference time for batch_size=%d is %.2f ms.' % (len(labels), 1000 * (time.time() - start)))
            count += len(labels)
            cur_accuracy = float(correct) * 100 / count
            print('{:>6}/{:<6} {:>6.2f}%'.format(count, total, cur_accuracy))
            #writer.add_summary(summary, batch_num)
            batch_num = batch_num + 1
        # Stop the worker threads
        coordinator.request_stop()
        coordinator.join(threads, stop_grace_period_secs=2)
        #writer.close()
    print('Top {} Accuracy: {}'.format(top_k, float(correct) / total))

    for lname in sorted(net.layer_op_counts, key=net.layer_op_counts.get, reverse=True):
        print lname, float(net.layer_op_counts[lname])/total_ops, \
              sparsity[lname]/total, input_min[lname], input_max[lname]

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to the converted model parameters (.npy)')
    parser.add_argument('val_gt', help='Path to validation set ground truth (.txt)')
    parser.add_argument('imagenet_data_dir', help='ImageNet validation set images directory path')
    parser.add_argument('--model', default='GoogleNet', help='The name of the model to evaluate')
    args = parser.parse_args()

    # Load the network
    net = load_model(args.model)
    if net is None:
        exit(-1)

    # Load the dataset
    data_spec = models.get_data_spec(model_instance=net)
    image_producer = dataset.ImageNetProducer(val_path=args.val_gt,
                                              data_path=args.imagenet_data_dir,
                                              data_spec=data_spec)

    # Evaluate its performance on the ILSVRC12 validation set
    validate(net, args.model_path, image_producer, args.model)


if __name__ == '__main__':
    main()
