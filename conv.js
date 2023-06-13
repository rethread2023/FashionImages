const tf = require('@tensorflow/tfjs-node');
const cv = require('opencv4nodejs');
const fs = require('fs');
const { promisify } = require('util');
const path = require('path');

const readFile = promisify(fs.readFile);

async function loadModel() {
  const modelPath = 'file://path/to/model'; // Ganti dengan path yang sesuai ke model yang telah Anda simpan
  const model = await tf.loadLayersModel(modelPath);
  return model;
}

function preprocessImage(image) {
  const resizedImage = image.resize(28, 28);
  const grayImage = resizedImage.cvtColor(cv.COLOR_BGR2GRAY);
  const normalizedImage = grayImage.normalize(0, 255).toFloat().expandDims();
  return normalizedImage;
}

async function predictImage(model, image) {
  const preprocessedImage = preprocessImage(image);
  const predictions = await model.predict(preprocessedImage).data();
  const predictedClass = Array.from(predictions).indexOf(Math.max(...predictions));
  return predictedClass;
}

async function run() {
  const model = await loadModel();

  const imageFilePath = 'path/to/image.jpg'; // Ganti dengan path yang sesuai ke gambar yang ingin diprediksi
  const imageBuffer = await readFile(imageFilePath);
  const image = cv.imdecode(imageBuffer);

  const predictedClass = await predictImage(model, image);

  const stylesFilePath = 'path/to/styles.csv'; // Ganti dengan path yang sesuai ke file styles.csv Anda
  const stylesBuffer = await readFile(stylesFilePath, 'utf-8');
  const styles = stylesBuffer.split('\n').map(row => row.split(','));
  const label = styles[predictedClass][0];

  console.log('Predicted Label:', label);
}

run().catch(console.error);
