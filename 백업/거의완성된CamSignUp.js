let classifier;
let imageModelURL = './model/';
let video;
let flippedVideo;
let label = "";
let captureButton;
let capturing = false;
let capturedImages = [];
let imageContainer;
let samplesList;
let modelfitButton;

function preload() {
  classifier = ml5.imageClassifier(imageModelURL + 'model.json');
}

function setup() {
  createCanvas(700, 400);
  video = createCapture(VIDEO);
  video.size(700, 400);
  video.hide();

  flippedVideo = ml5.flipImage(video);
  let captureButton = select('#captureButton');
  let downloadButton = select('#downloadButton');
  let resetButton = select('#resetButton');
  let modelfitButton = select('#modelfitButton')
  imageContainer = select('.imageContainer');
  samplesList = select('#samples');

  captureButton.mousePressed(startCapture);
  captureButton.mouseReleased(stopCapture);
  downloadButton.mousePressed(downloadImages);
  resetButton.mousePressed(resetImages);
  modelfitButton.mousePressed(modelfit);
  classifyVideo();
}

function modelfit() {
  const trainingData = [];
  const label = 'me';
  // const fixedSize = 700 * 400 * 4; //픽셀사이즈 1차원으로 만듬
  const fixedSize = 10000;
  for (let i = 0; i < capturedImages.length; i++) {
    const img = capturedImages[i];
    const pixels = new Array(fixedSize); // Initialize an array with a fixed size
    img.loadPixels();

    // Loop over pixels and flatten RGBA values
    for (let j = 0; j < fixedSize; j++) {
      pixels[j] = img.pixels[j] / 255.0; // Normalize to [0, 1]
    }

    trainingData.push({ input: pixels, output: label });
  }

  console.log('Number of training examples:', trainingData.length);
  console.log('Length of pixels array:', pixels.length);

  // ???
  const xs = tf.tensor2d(trainingData.map(item => item.input), [trainingData.length, fixedSize]);
  const ys = tf.tensor1d(trainingData.map(item => (item.output === 'me' ? 1 : 0)));

  // 모델링
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, inputShape: [fixedSize], activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  // 모델 컴파일
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  //모델 훈련
  model.fit(xs, ys, { epochs: 10 }).then((info) => {
    //모델 저장
    model.save('downloads://model');
    console.log('모델이 저장되었습니다..');
  });
}

function startCapture() {
  if (!capturing) {
    capturing = true;
  }
}

function stopCapture() {
  capturing = false;
}

function resetImages() {
  capturedImages = [];
  samplesList.html('');
}

function downloadImages() {
  let zip = new JSZip();
  for (let i = 0; i < capturedImages.length; i++) {
    let imgData = capturedImages[i].canvas.toDataURL('image/png').replace(/^data:image\/(png|jpg);base64,/, '');
    zip.file(`image${i + 1}.png`, imgData, { base64: true });
  }
  zip.generateAsync({ type: 'blob' })
    .then(function (blob) {
      let link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'captured_images.zip';
      link.click();
    });
}

function draw() {
  background(0);
  image(flippedVideo, 0, 0);
  fill(255);
  textSize(16);
  text(label, width / 2, height - 4);
  textAlign(CENTER);
  if (capturing) {
    let img = createImage(width, height);
    img.copy(flippedVideo, 0, 0, width, height, 0, 0, width, height);
    capturedImages.push(img);
    displayCapturedImage(img);     
  }
}

function displayCapturedImage(img) {
  let listItem = createElement('li', '');
  let imgElement = createImg(img.canvas.toDataURL(), 'captured image');
  imgElement.size(58, 58);
  listItem.child(imgElement);
  samplesList.child(listItem);
}

function classifyVideo() {
  flippedVideo = ml5.flipImage(video)
  classifier.classify(flippedVideo, gotResult);
  flippedVideo.remove();

}

function sleep(sec) {
  let start = Date.now(), now = start;
  while (now - start < sec * 1000 / 3) {
      now = Date.now();
      }
  }

function gotResult(error, results) {
  if (error) {
    console.error(error);
    return;
  }
  label = results[0].label;

  if (label === 'me') {

  }
  classifyVideo();
}