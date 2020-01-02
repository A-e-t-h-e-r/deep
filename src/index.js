import * as tf from '@tensorflow/tfjs'
import {train,initContentStyle,binit} from './model';
import {loadAllModel,startStyling} from "./fast";

const IMAGE_HEIGHT = 224;
const IMAGE_WEIGHT = 224;
const NOISE = tf.scalar(0.5);
const IMAGE_MEAN_VALUE = tf.scalar(128.)
export var vgg16_model;
var content_img;
export var style_img;
export var gen_img;


const contentImage = document.getElementById("contentimg");
const styleImage = document.getElementById("styleimg");
const resultimg =document.getElementById("resultimg")
export const contentImage_fast = document.getElementById("contentimg-fast");
export const styleImage_fast = document.getElementById("styleimg-fast");
export const resultimg_fast = document.getElementById("resultimg-fast")
const start = document.getElementById("confirm-fast")
const sinput =document.getElementById("sinput")
const cinput =document.getElementById("cinput")
const initbutton = document.getElementById("train")
const startbutton = document.getElementById("confirm")
const styleselect_fast = document.getElementById("styleselect-fast")
const contentselect_fast = document.getElementById("contentselect-fast")
const styleselect = document.getElementById("styleselect")
const contentselect = document.getElementById("contentselect")
const switchstate = document.getElementById("switch")
const main = document.getElementById("main")
const main_fast = document.getElementById("main-fast")

var state=1
main.hidden=false
main_fast.hidden=true

switchstate.onclick=function(){
    if(state==1){
        state=0
        console.log(state)
        switchstate.innerHTML="初始"
        main_fast.style.display='none'
        main.style.display='flex'
    }
    else{
        state=1
        console.log(state)
        switchstate.innerHTML="快速"
        main.style.display='none'
        main_fast.style.display='flex'

    }
}

contentImage.height = IMAGE_HEIGHT;
styleImage.height = IMAGE_HEIGHT;

start.onclick=function () {
    startStyling()
}
sinput.onchange=function () {
    var file = this.files[0];
    var reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function () {
        if(state==1){
            styleImage_fast.src = reader.result;
        }
        else{
            styleimg.src = reader.result;
        }

    }
}
cinput.onchange=function () {
    var file = this.files[0];
    var reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function () {
        if (state==1) {
            contentImage_fast.src = reader.result;
        }
        else{
            contentimg.src = reader.result;
        }
    }
}
styleselect.onchange=function () {
    if(styleselect.selectedIndex==2){
        styleimg.src="style.jpg"
    }
    else{
        sinput.click();
    }

}
contentselect.onchange=function () {
    if(contentselect.selectedIndex==2){
        contentimg.src="dog.jpg"
    }
    else{
        cinput.click();
    }
}
styleselect_fast.onchange=function () {
    if(styleselect_fast.selectedIndex==2){
        styleImage_fast.src="style.jpg"
    }
    else{
        sinput.click();
    }
}
contentselect_fast.onchange=function () {
    if(contentselect_fast.selectedIndex==2){
        contentImage_fast.src="dog.jpg"
    }
    else{
        cinput.click();
    }
}

console.log('Loading Model...');
async function loadModel() {

    vgg16_model = await tf.loadLayersModel('./model/model.json')
    console.log('Loaded!')
    binit()
}

loadModel()
loadAllModel()


async function init(){
    style_img = preprocess(styleImage)
    initbutton.disabled = true;
    startbutton.disabled= true;
    await initContentStyle()
    initbutton.disabled = false;
    startbutton.disabled= false;
}
initbutton.onclick=function () {
    init();
};

async function starttrain() {
    await tf.nextFrame();
    initbutton.disabled = true;
    startbutton.disabled = true;
    await tf.nextFrame();
    content_img = preprocess(contentImage)
    gen_img = get_random_img()
    await train()
    topixel(gen_img)
    initbutton.disabled = false;
    startbutton.disabled= false;
    console.log(tf.memory())
}
startbutton.onclick=function () {
    starttrain();
};

//预处理图片张量以符合输入网络的size
function preprocess(image) {
    console.log(image.width, image.height)
    return tf.tidy(
        () => {
            const tensor = tf.browser.fromPixels(image).toFloat()
            const resized = tf.image.resizeBilinear(tensor, [IMAGE_HEIGHT, IMAGE_WEIGHT])
            const reshape = resized.expandDims(0)
            const changed = reshape.sub(IMAGE_MEAN_VALUE)
            return changed
        }
    )
}
//根据内容图片产生噪声图片
function get_random_img() {
    return tf.tidy(() => {
            const noise_image = tf.randomUniform([1, IMAGE_HEIGHT, IMAGE_WEIGHT, 3], -20, 20,'float32')
            const random_image = tf.mul(noise_image, NOISE).add(tf.mul(content_img, tf.sub(tf.scalar(1), NOISE)))
            return tf.variable(random_image)
        }
    )
}

//将输出张量转化为图片
function topixel(image) {
    const tensor = image.add(IMAGE_MEAN_VALUE)
    const resized = tensor.reshape([IMAGE_HEIGHT, IMAGE_WEIGHT, 3])
    const modified = tf.image.resizeBilinear(resized, [IMAGE_HEIGHT, contentImage.width])
    const changed = modified.div(tf.scalar(255.)).clipByValue(0., 1.)
    tensor.dispose()
    resized.dispose()
    modified.dispose()
    tf.browser.toPixels(changed, resultimg)
}



