<!DOCTYPE html>
<html>
<head>
  <title>Fashion Image Classification</title>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.10.0/dist/tf.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter@3.10.0/dist/tfjs-converter.js"></script>
</head>
<body>
  <input type="file" id="image-selector" onchange="loadImage(event)" accept="image/*">
  <div>
    <h3>Prediction:</h3>
    <p id="prediction-result"></p>
  </div>
</body>
<script>
  // Fungsi untuk memuat gambar yang dipilih oleh pengguna
  function loadImage(event) {
    var image = document.createElement('img');
    image.src = URL.createObjectURL(event.target.files[0]);
    image.onload = async function() {
      await tf.ready();
      const model = await tf.loadLayersModel('model/model.json');

      // Mengubah gambar menjadi tensor
      const tensor = preprocessImage(image);

      // Melakukan prediksi
      const predictions = await model.predict(tensor).data();
      const predictedClass = Array.from(predictions).indexOf(Math.max(...predictions));

      // Mengambil label prediksi
      const labels = await fetch('model/labels.json');
      const labelData = await labels.json();
      const predictedLabel = labelData[predictedClass];

      // Menampilkan hasil prediksi
      document.getElementById('prediction-result').innerHTML = predictedLabel;
    };
  }

  // Fungsi untuk memproses gambar sebelum melakukan prediksi
  function preprocessImage(image) {
    // Mengubah ukuran gambar menjadi 28x28
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, 28, 28);
    const imageData = ctx.getImageData(0, 0, 28, 28);

    // Mengubah data piksel menjadi rentang [0, 1]
    const data = imageData.data;
    let imagePixels = [];
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i] / 255;
      const g = data[i + 1] / 255;
      const b = data[i + 2] / 255;
      const grayScale = (r + g + b) / 3;
      imagePixels.push(grayScale);
    }

    // Mengubah array piksel menjadi tensor 2D
    const tensor = tf.tensor2d(imagePixels, [1, 784]);

    return tensor;
  }
</script>
</html>
