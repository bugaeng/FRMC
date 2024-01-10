
  // Classifier Variable
  let classifier;
  // Model URL
  let imageModelURL = './model/';
  
  // Video
  let video;
  let flippedVideo;
  // To store the classification
  let label = "";

  let captureButton;
// Flag to indicate if capturing is active
  let capturing = false;
  // Load the model first
  let capturedImages = [];
  
  let imageContainer;
  let samplesList;

  let modelTrainButton;

  function preload() {
    classifier = ml5.imageClassifier(imageModelURL + 'model.json');
  }

  function setup() {
    createCanvas(700, 400);
    // Create the video
    video = createCapture(VIDEO);
    video.size(700, 400);
    video.hide();
  
    flippedVideo = ml5.flipImage(video);
    // Start classifying
    let captureButton = select('#captureButton');
    let downloadButton = select('#downloadButton');
    let resetButton = select('#resetButton');
    let modelTrainButton = select('#modelBUtton'); // Fix typo in the variable name
  
    captureButton.mousePressed(startCapture);
    captureButton.mouseReleased(stopCapture);
    downloadButton.mousePressed(downloadImages);
    resetButton.mousePressed(resetImages);
  
    modelTrainButton.mousePressed(trainAndSaveModel);
  
    classifyVideo();
    imageContainer = select('.imageContainer');
    samplesList = select('#samples');
  }
// ------------------------------------------------------------------------------------------ //
  

  // Function to train and save the model
  function trainAndSaveModel() {
    // Prepare the data for training
    const trainingData = [];
    for (let i = 0; i < capturedImages.length; i++) {
      const imgData = capturedImages[i].canvas.toDataURL('image/png').replace(/^data:image\/(png|jpg);base64,/, '');
      const label = capturedImages[i].label;
  
      // Add image data and label to training data
      trainingData.push({ image: imgData, label });
    }
  
    // Create an ml5 image classifier with the MobileNet model
    const trainedClassifier = ml5.imageClassifier('MobileNet', () => {
      // Add each image to the classifier for training
      for (let i = 0; i < trainingData.length; i++) {
        const imgData = trainingData[i].image;
        const label = trainingData[i].label;
  
        // Add image to the classifier for training
        trainedClassifier.addImage(imgData, label, () => {
          console.log(`Added image ${i + 1} to training data.`);
        });
      }
  
      // Train the classifier
      trainedClassifier.train({ epochs: 50 }, () => {
        // Save the trained model
        trainedClassifier.save('./model');
  
        // Notify that the model has been trained and saved
        console.log('Model has been trained and saved.');
      });
    });
  }


// Function to load the trained model
  
// ------------------------------------------------------------------------------------------ //

function startCapture() {
  if (!capturing) {
    capturing = true;
    // Start capturing when the mouse is pressed
    captureImage();
  }
}

  
  // function stopCapture() {
  //   // Stop capturing when the mouse is released
  //   capturing = false;
  //   trainAndSaveModel();
  // }
  function stopCapture() {
    if (capturing) {
      // Stop capturing when the mouse is released
      capturing = false;
  
      // Check if Modeltrain button is pressed
      if (
        mouseX > modelTrainButton.position().x &&
        mouseX < modelTrainButton.position().x + modelTrainButton.width &&
        mouseY > modelTrainButton.position().y &&
        mouseY < modelTrainButton.position().y + modelTrainButton.height
      ) {
        // If Modeltrain button is pressed, train and save the model
        trainAndSaveModel();
        // After training, start video classification
        classifyVideo();
      }
    }
  }

  function resetImages() {
    capturedImages = [];
    samplesList.html('');
  }

  function downloadImages() {
    // Create a new zip file
    let zip = new JSZip();
  
    // Add captured images to the zip file
    for (let i = 0; i < capturedImages.length; i++) {
      let imgData = capturedImages[i].canvas.toDataURL('image/png').replace(/^data:image\/(png|jpg);base64,/, '');
      zip.file(`image${i + 1}.png`, imgData, { base64: true });
    }
  
    // Generate a blob containing the zip file
    zip.generateAsync({ type: 'blob' })
      .then(function (blob) {
        // Create a download link for the blob and trigger a click event
        let link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'captured_images.zip';
        link.click();
      });
  }

  // function captureImage() {
  //   // Save the current frame as an image
  //   saveCanvas('image' + frameCount, 'png');
  // }

  function draw() {
    background(0);
    // Draw the video
    image(flippedVideo, 0, 0);

    // Draw the label
    fill(255);
    textSize(16);
    textAlign(CENTER);
    text(label, width / 2, height - 4);
    // if (capturing) {
    //   saveCanvas('image' + frameCount, 'png');
    // }
    if (capturing) {
      // Capture every 20 frames (adjust as needed)
      let img = createImage(width, height);
      img.copy(flippedVideo, 0, 0, width, height, 0, 0, width, height);
      capturedImages.push(img);
      displayCapturedImage(img);
  
      // Display the captured image on the web page
      
    }

    // if (capturing) {
    //   let img = createImage(width, height);
    //   img.copy(flippedVideo, 0, 0, width, height, 0, 0, width, height);
    //   capturedImages.push(img);

    //   displayCapturedImage(img);
    // }
  }
  
  function displayCapturedImage(img) {
    // Create a new <li> element
    let listItem = createElement('li', '');
  
    // Create an <img> element
    let imgElement = createImg(img.canvas.toDataURL(), 'captured image');
    imgElement.size(58, 58); // Set the size as needed
  
    // Append the <img> element to the <li> element
    listItem.child(imgElement);
  
    // Append the <li> element to the samples list
    samplesList.child(listItem);

  }

  function classifyVideo() {
    flippedVideo = ml5.flipImage(video);
    classifier.classify(flippedVideo, gotResult);
    flippedVideo.remove();
  }
  
  function sleep(sec) {
    let start = Date.now(), now = start;
    while (now - start < sec * 1000 / 3) {
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
    // const resultElement1 = document.querySelector(".result1");
    // const resultElement2 = document.querySelector(".result2");
  
  // if (label === "Class 1") {
    classifyVideo();
    trainedClassifier.classify(flippedVideo, gotResult);
    // resultElement1.innerHTML = "인증되었습니다.";
    // resultElement2.innerHTML = "잠시 후 이동 됩니다.";
    // sleep(3);
    // window.location.href = "main.html";
    // video.update();
  //   return;
    
  // } 
  // else 
  // {
  //   classifyVideo();
  // }
}

  