const quotes = [   
  {
  "quote": "카메라를 부위에 보이게하세요",
  "source": "피부병 체크하기"
  },
  {
    "quote": "7초간 기다리세요",
    "source": ""
  }

  
]

function randomQuote(){
  let random = quotes[Math.floor(Math.random() * quotes.length)];
  quotation.innerText = `"${random.quote}"`;
  source.innerText = random.source;
}

setInterval(randomQuote, 4000);
 /*

function gotResult(error, results) {
  // If there is an error
  if (error) {
    console.error(error);
    return;
  }
  label = results[0].label;




//질병분류 해서 특정값에 일치하면 실행되는 스크립트
  if (label === "mn") 
  {
    let input = select('#fname');

    // Set the input value with the label
    input.value(label);

             Swal.fire({
             icon: "success",
            title: "성공!",
            text: "메인메뉴로 이동합니다.",
            footer: '<a href="https://www.youtube.com/">유튜브나 볼래요</a>',
            onClose: function() {
                window.location.href = './final.html';

                if (label === "mn") {
                savedLabel = "mn"; //질병이름
              
                // 로컬 저장
                localStorage.setItem('disease', savedLabel); // 질병이라는 키값
              }
            }
            });
            if (isPopupShown) {
             return;
            }
        

  }




  if (label === "ph") 
  {
    let input = select('#fname');

    // Set the input value with the label
    input.value(label);


  }


  if (label === "no")
  {
    let input = select('#fname');
    input.value(null);
   
  }



  classifyVideo();
}
*/
