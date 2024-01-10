let video;
let classifier;
let label = "";
let capturing = false;
let capturedImages = [];
let samplesList;

function preload() {
  // Load your pre-trained model here
  classifier = ml5.imageClassifier('./model/face_id/model.json');
}

function setup() {
  createCanvas(700, 400);
  video = createCapture(VIDEO);
  video.size(700, 400);
  video.hide();

  samplesList = select('#samples');

  classifyVideo(); // Start continuous classification
}

function draw() {
  background(0);
  image(video, 0, 0, 700, 400);

  fill(255);
  textSize(16);
  text(label, width / 2, height - 4);
  textAlign(CENTER);

  if (capturing) {
    let img = createImage(width, height);
    img.copy(video, 0, 0, width, height, 0, 0, width, height);
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
  classifier.classify(video, gotResult);
}

function gotResult(error, results) {
  if (error) {
    console.error(error);
    return;
  }

  label = results[0].label;

  // Check if the label is the one you are interested in (e.g., "person")
  if (label === 'person') {
    // Perform the transition to main.html without needing a button
    window.location.href = 'main.html';
  }

  // Continue classifying in a loop
  classifyVideo();
}
