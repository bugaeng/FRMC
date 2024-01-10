
const outA = document.querySelector("#out");
const loginForm = document.querySelector("#form");
const usernameInput = document.querySelector("#usernameInput");
const ageInput = document.querySelector("#ageInput");
const weightInput = document.querySelector("#weightInput");
const logOut = document.querySelector("#logout");
const usrName = localStorage.getItem("username");
const age = localStorage.getItem("Age");
const weight = localStorage.getItem("weight");
const saveLabel = localStorage.getItem("saveLabel");






function onLoginSubmit(event) {
    event.preventDefault();
    loginForm.classList.add("hidden");
    loginForm.classList.remove("aaa");

    const username = usernameInput.value;
    const ageValue = ageInput.value;
    const weightValue = weightInput.value;

    localStorage.setItem("username", username);
    localStorage.setItem("Age", ageValue);
    localStorage.setItem("weight", weightValue);

    paintHello(username, "username");
    paintHello(ageValue, "age");
    paintHello(weightValue, "weight");

    location.reload();
}
let isButtonCreated = false; // 버튼이 생성되었는지를 나타내는 변수

function paintHello(value, type) { //출력이벤트
    if (type === "username") {
        outA.innerText += `${value}님\n`;
    } else if (type === "age") {
        outA.innerText += `나이 ${value}세\n`;
    } else if (type === "weight") {
        outA.innerText += `몸무게 ${value}kg\n`;
    }    

    outA.classList.remove("hidden");
    logOut.classList.remove("hidden");

    if (!isButtonCreated) { // 버튼이 생성되지 않았을 때만 버튼 생성
        const btn = document.createElement('button'); //버튼을 생성
        const btnText = document.createTextNode(`이동하기`); //이동하기 위치기 맘에안든다? login1 css 에서 #button1 수정

        btn.appendChild(btnText);
        btn.id = "button1"; // 버튼에 ID 지정 만약 지정안하고 같은 이름을 가질시에 다른버튼도 링크이동이됨 
        document.body.appendChild(btn);

        isButtonCreated = true; // 버튼을 생성했음을 표시
        document.getElementById('button1').addEventListener('click',e=>{
           
            window.location = '/camera/camera.html';
          

        });
    }

}


function logOutFunc() {
    if (usrName !== null) {
      localStorage.removeItem("username");
      localStorage.removeItem("Age");
      localStorage.removeItem("weight");
      localStorage.removeItem("saveLabel");
  
      logOut.classList.add("hidden");
      outA.classList.add("hidden");
      loginForm.classList.remove("hidden");
      loginForm.classList.add("aaa");
      localStorage.clear(); //로그아웃 로컬스트로지 모든정보초기화
      location.reload();
 
     displayPatientInfo(); //얘도 마찬가지 
    }
  }


if (usrName === null) {
    loginForm.classList.remove("hidden");
    loginForm.classList.add("aaa");
    logOut.classList.add("hidden");
    loginForm.addEventListener("submit", onLoginSubmit);
} else {
    paintHello(usrName, "username");
    paintHello(age, "age");
    paintHello(weight, "weight");
    logOut.addEventListener("click", logOutFunc);
}


//숫자외 한글 영어 금지 
ageInput.addEventListener("input", function(event) {
    event.target.value = event.target.value.replace(/[^0-9]/g, "");

//팝업으로 띄워서 경고시키려 했으나 무한반복됨

  });
  
  weightInput.addEventListener("input", function(event) {
    event.target.value = event.target.value.replace(/[^0-9]/g, "");

});


