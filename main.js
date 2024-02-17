const tf = require("@tensorflow/tfjs")

async function run(){
    const MODEL_URL = 'http://192.168.42.41:8080/model.json';
    const model = await tf.loadLayersModel(MODEL_URL);
    console.log(model.summary());
    const input = tf.tensor2d([20.0], [1,1]);
    const prediction1 = (await model.predict(input).data());
    console.log(Math.round(prediction1))
}
run();