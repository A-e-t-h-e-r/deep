import * as tf from '@tensorflow/tfjs'
import {contentImage_fast,styleImage_fast,resultimg_fast} from './index';



var styleNet
var transformNet
export async function loadAllModel() {
    styleNet = await tf.loadGraphModel(
        'saved_model_style_js/model.json');
    transformNet = await tf.loadGraphModel(
        'saved_model_transformer_separable_js/model.json'
    );
    console.log('Loaded!')
}


export async function startStyling() {
    await tf.nextFrame();
    await tf.nextFrame();
    let bottleneck = await tf.tidy(() => {
        return styleNet.predict(tf.browser.fromPixels(styleImage_fast).toFloat().div(tf.scalar(255)).expandDims());
    })
    await tf.nextFrame();
    const stylized = await tf.tidy(() => {
        return transformNet.predict([tf.browser.fromPixels(contentImage_fast).toFloat().div(tf.scalar(255)).expandDims(), bottleneck]).squeeze();
    })
    await tf.browser.toPixels(stylized, resultimg_fast);
    console.log("success")
    bottleneck.dispose();
    stylized.dispose();
}
