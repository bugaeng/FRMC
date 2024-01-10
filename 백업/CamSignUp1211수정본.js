let video;
let flippedVideo;
let label = "";
let captureButton;
let capturing = false;
let capturedImages = [];
let samplesList;
let imageContainer;
const numImages = capturedImages.length;

function setup() {
  createCanvas(700, 400);
  video = createCapture(VIDEO);
  video.size(700, 400);
  video.hide();

  let captureButton = select('#captureButton');
  let downloadButton = select('#downloadButton');
  let resetButton = select('#resetButton');
  let modelfitButton = select('#modelfitButton');
  imageContainer = select('.imageContainer');
  samplesList = select('#samples');

  captureButton.mousePressed(startCapture);
  captureButton.mouseReleased(stopCapture);
  downloadButton.mousePressed(downloadImages);
  resetButton.mousePressed(resetImages);
  modelfitButton.mousePressed(modelfit);
}

// capture
function startCapture() {
  console.log('startCapture');
  if (!capturing) {
    capturing = true;
  }
}

// capture
function stopCapture() {
  console.log('stopCapture');
  capturing = false;
}

// reset image
function resetImages() {
  capturedImages = [];
  samplesList.html('');
}

// download image
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

// 출력
function draw() {
  background(0);
  image(video, 0, 0, 700, 400);
  if (capturing) {
    let img = get();
    if (img) {
      capturedImages.push(img);
      displayCapturedImage(img);
      updateImageCountMessage();
    } else {
      console.error('Error capturing image.');
    }
  }
}

// 이미지 카운트
function updateImageCountMessage() {
  const imageCountMessage = select('#imageCountMessage');
  if (imageCountMessage) {
    const numImages = capturedImages.length;
    const message = `${numImages} 개의 이미지가 추가되었습니다.`;
    imageCountMessage.html(message);
  }
}
// 이미지 미리보기
function displayCapturedImage(img) {
  console.log('Displaying captured image:', img);
  let listItem = createElement('li', '');
  let imgElement = createImg(img.canvas.toDataURL(), 'captured image');
  imgElement.size(58, 58);
  listItem.child(imgElement);
  samplesList.child(listItem);
}

// 모델링
function modelfit() {
  const numClasses = capturedImages.length;
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [58, 58, 3] }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: numClasses, activation: 'sigmoid' }));

  const optimizer = tf.train.adam();
  model.compile({ optimizer, loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  // 오류 처리 -------------------------------------------------------------------------------------------------------- //

  if (capturedImages.length === 0) {
    console.error('No captured images for training.');
    return;
  }
  
  const labels = capturedImages.map((image, index) => index);
  const isValidImage = capturedImages.every(img => img instanceof p5.Image && img.canvas);

  if (!isValidImage) {
    console.error('Invalid images in the capturedImages array.');
    return;
  }

  if (labels.length !== numClasses) {
    console.error(`Number of classes (${labels.length}) doesn't match the model configuration (${numClasses}).`);
    return;
  }

  if (!labels.every(Number.isInteger)) {
    console.error('Labels must be integers.');
    return;
  }
  // 오류 처리 끝 -------------------------------------------------------------------------------------------------------- //

  const x_train = capturedImages.map(preprocessImage);
  const y_train = tf.oneHot(tf.tensor1d(labels, 'int32'), numClasses);
    // const y_train = tf.tensor1d(labels, 'int32');

  console.log(x_train.map(t => t.shape));
  console.log(x_train[0].shape);

  
  
  model.fit(x_train, y_train, { epochs: 10 }).then(() => {
    model.save('downloads://my_model');
  });
}

// 전처리 함수
function preprocessImage(imageData) {

  if (!(imageData instanceof p5.Image) || !imageData.canvas) {
    console.error('Invalid image format:', imageData);
    return null;
  }

  const tensor = tf.browser.fromPixels(imageData.canvas).toFloat();
  const resizedTensor = tf.image.resizeNearestNeighbor(tensor, [58, 58]);
  const normalizedTensor = resizedTensor.div(255.0);
  return normalizedTensor.expandDims();
}

function labelToIndex(label) {
  const labelMap = { "user": 0 };
  return labelMap[label];
}

function labelForImage(image) {
  return "user";
}