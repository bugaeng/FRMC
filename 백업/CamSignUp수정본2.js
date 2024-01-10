
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
  imageContainer = select('.imageContainer');
  samplesList = select('#samples');

  captureButton.mousePressed(startCapture);
  captureButton.mouseReleased(stopCapture);
  downloadButton.mousePressed(downloadImages);
  resetButton.mousePressed(resetImages);

  classifyVideo();
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
  
  classifyVideo();
}

