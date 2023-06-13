
  // Fungsi untuk memuat gambar yang dipilih oleh pengguna
  async function loadImage(event) {
    const image = document.createElement('img');
    image.src = URL.createObjectURL(event.target.files[0]);
    image.onload = async function() {
      await tf.ready();
      const model = await tf.loadLayersModel('tfjs_model/model.json');

      // Mengubah gambar menjadi tensor
      const tensor = preprocessImage(image);

      // Melakukan prediksi
      const predictions = model.predict(tensor);
      const predictedClass = predictions.argMax(1).dataSync()[0];

      // Mengambil label prediksi
      const labels = await fetch('styles.csv');
      const labelData = await labels.text();
      const labelLines = labelData.split('\n');
      const predictedLabel = labelLines[predictedClass].trim();

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
    const imagePixels = [];
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

