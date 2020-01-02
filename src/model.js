import * as tf from '@tensorflow/tfjs'
import {vgg16_model,style_img,gen_img} from "./index";

const TRAIN_STEPS = 100
var net_input;
var net_conv1_1;
var net_conv1_2;
var net_pool1;
var net_conv2_1;
var net_conv2_2;
var net_pool2;
var net_conv3_1;

var style_conv1_1;
var style_conv2_1;
var style_conv3_1;

var block1_conv1;
var block1_conv2;
var block2_conv1;
var block2_conv2;
var block3_conv1;

var M;
var N;
var weight;

//取vgg16模型layer_name层参数(w,b)
function get_wb(layer_name) {
    const kernel = vgg16_model.getLayer(layer_name)['kernel'].val
    const bias = vgg16_model.getLayer(layer_name)['bias'].val
    kernel.trainable = false
    bias.trainable = false
    return [kernel, bias]
}
//加载模型参数以及得到风格图片特征矩阵
export function binit() {
    block1_conv1 = get_wb('block1_conv1');
    block1_conv2 = get_wb('block1_conv2');
    block2_conv1 = get_wb('block2_conv1');
    block2_conv2 = get_wb('block2_conv2');
    block3_conv1 = get_wb('block3_conv1');
}

function conv_relu(input, wb) {
    return tf.tidy(() => {
            const conv = tf.conv2d(input, wb[0], [1, 1], 'same')
            const relu = tf.relu(tf.add(conv, wb[1]))
            return relu
        }
    )
}
function pool(input) {
    return tf.maxPool(input, [2, 2], [2, 2], 'same')
}

//建立模型，只用到了前若干层，所以不用取原模型的所有层
function buildModel(input) {
    net_input = input
    net_conv1_1 = conv_relu(net_input, block1_conv1)
    net_conv1_2 = conv_relu(net_conv1_1, block1_conv2)
    net_pool1 = pool(net_conv1_2)
    net_conv2_1 = conv_relu(net_pool1, block2_conv1)
    net_conv2_2 = conv_relu(net_conv2_1, block2_conv2)
    net_pool2 = pool(net_conv2_2)
    net_conv3_1 = conv_relu(net_pool2, block3_conv1)
}


export function initContentStyle() {
    buildModel(style_img);
    style_conv1_1 = net_conv1_1;
    style_conv2_1 = net_conv2_1;
    style_conv3_1 = net_conv3_1;
}

//计算feature矩阵的gram(相关性)矩阵
function gram(x, size, deep) {
    return tf.tidy(() => {
            const y = x.reshape([size, deep])
            const z = tf.matMul(tf.transpose(y), y)
            return z
        }
    )
}

//计算损失函数，这里仅计算风格损失
function loss() {
    return tf.tidy(() => {

            buildModel(gen_img)

            var style_loss = tf.scalar(0.)

            //computing style loss
            //conv1_1
            M = net_conv1_1.shape[1] * net_conv1_1.shape[2];
            N = net_conv1_1.shape[3];
            weight = 0.2
            const net_conv1_1_gram = gram(net_conv1_1, M, N)
            const style_conv1_1_gram = gram(style_conv1_1, M, N)
            style_loss = style_loss.add(tf.scalar(weight).div(tf.scalar(4 * M * M * N * N)).mul(tf.sum(tf.pow(tf.sub(net_conv1_1_gram, style_conv1_1_gram), 2))));

            //conv2_1
            M = net_conv2_1.shape[1] * net_conv2_1.shape[2];
            N = net_conv2_1.shape[3];
            weight = 0.2
            const net_conv2_1_gram = gram(net_conv2_1, M, N)
            const style_conv2_1_gram = gram(style_conv2_1, M, N)
            style_loss = style_loss.add(tf.scalar(weight).div(tf.scalar(4 * M * M * N * N)).mul(tf.sum(tf.pow(tf.sub(net_conv2_1_gram, style_conv2_1_gram), 2))));

            //conv3_1
            M = net_conv3_1.shape[1] * net_conv3_1.shape[2];
            N = net_conv3_1.shape[3];
            weight = 0.2
            const net_conv3_1_gram = gram(net_conv3_1, M, N)
            const style_conv3_1_gram = gram(style_conv3_1, M, N)
            style_loss = style_loss.add(tf.scalar(weight).div(tf.scalar(4 * M * M * N * N)).mul(tf.sum(tf.pow(tf.sub(net_conv3_1_gram, style_conv3_1_gram), 2))));

            return style_loss;
        }
    )
}
//训练模型，采用adam算法优化，learning_rate = 1.0
export function train() {
    const optimizer = tf.train.adam(1.0)
    for (let i = 0; i < TRAIN_STEPS; i++) {
        optimizer.minimize(
            () => {
                return loss()
            }
        )
    }
}

