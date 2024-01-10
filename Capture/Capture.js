const canvas = document.querySelector(".capture");
const samplesList = document.getElementById("samples");
let capturedImages = [];

document.querySelector("#captureButton").addEventListener("click", () => {
    let img = new Image();
    img.src = "http://localhost:8000/Capture";
    img.crossOrigin = 'Anonymous';
    img.onload = function () {
        const captureCanvas = document.createElement("canvas");
        captureCanvas.width = img.width;
        captureCanvas.height = img.height;
        const ctx = captureCanvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        // 이미지를 캡처하여 배열에 추가
        capturedImages.push({
            canvas: captureCanvas,
            // 다른 이미지 정보도 필요하다면 여기에 추가
        });

        // 이미지 미리보기 업데이트
        displayCapturedImage(capturedImages[capturedImages.length - 1]);

        // 이미지 카운트 업데이트
        updateImageCountMessage();
    };
});

// 이미지 미리보기
function displayCapturedImage(img) {
    console.log('Displaying captured image:', img);
    let listItem = document.createElement('li');
    let imgElement = document.createElement('img');
    imgElement.src = img.canvas.toDataURL();
    imgElement.width = 58;
    imgElement.height = 58;
    listItem.appendChild(imgElement);
    samplesList.appendChild(listItem);
}

// 리셋 버튼에 대한 이벤트 리스너 추가
document.querySelector("#resetButton").addEventListener("click", () => {
    // 이미지 배열 및 미리보기 초기화
    capturedImages = [];
    samplesList.innerHTML = '';

    // 이미지 카운트 업데이트
    updateImageCountMessage();
});

document.querySelector("#downloadButton").addEventListener("click", () => {
  downloadImages();
});

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

// 이미지 카운트 업데이트
function updateImageCountMessage() {
    const imageCountMessage = document.getElementById('imageCountMessage');
    const numImages = capturedImages.length;
    const message = `${numImages} 개의 이미지가 추가되었습니다.`;
    imageCountMessage.innerHTML = message;
}
