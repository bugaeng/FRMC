
  // Classifier Variable
  let classifier;
  // Model URL
  let imageModelURL = './model/face_id/';
  
  // Video
  let video;
  let flippedVideo;
  // To store the classification
  let label = "";
  // Load the model first
  function preload() {
    classifier = ml5.imageClassifier(imageModelURL + 'model.json');
  }

  function setup() {
    createCanvas(700, 400);
    // Create the video
    video = createCapture(VIDEO);
    video.size(700, 380);
    video.hide();

    flippedVideo = ml5.flipImage(video);
    // Start classifying
    classifyVideo();
  }

  function draw() {
    background(0);
    // Draw the video
    image(flippedVideo, 0, 0);

    // Draw the label
    fill(255);
    textSize(16);
    textAlign(CENTER);
    text(label, width / 2, height - 4);
  }

  // Get a prediction for the current video frame
  function classifyVideo() {
    flippedVideo = ml5.flipImage(video)
    classifier.classify(flippedVideo, gotResult);
    flippedVideo.remove();

  }
  function sleep(sec) {
    let start = Date.now(), now = start;
    while (now - start < sec * 1000) {
        now = Date.now();
        }
    }
    // When we get a result
  function gotResult(error, results) {
    // If there is an error
    if (error) {
      console.error(error);
      return;
    }
    // The results are in an array ordered by confidence.
    // console.log(results[0]);
    label = results[0].label;
    const resultElement1 = document.querySelector(".result1");
    const resultElement2 = document.querySelector(".result2");
  
  if (label === "person") {
    classifyVideo();
    resultElement1.innerHTML = "인증되었습니다.";
    resultElement2.innerHTML = "잠시 후 이동 됩니다.";
    sleep(3);
    window.location.href = "main.html";
    video.update();
    return;
    
  } 
  else 
  {
    classifyVideo();
  }
}

  