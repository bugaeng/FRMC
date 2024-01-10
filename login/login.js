import { initializeApp } from "https://www.gstatic.com/firebasejs/10.5.2/firebase-app.js";
import { getDatabase } from "https://www.gstatic.com/firebasejs/10.5.2/firebase-database.js";
import { getAuth, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.5.2/firebase-auth.js";

 const firebaseConfig = {
    apiKey: "AIzaSyAk2qWaXTtViqQImiOMuxqKTJ80WGIvU6M",
    authDomain: "face-id-895f3.firebaseapp.com",
    databaseURL: "https://face-id-895f3-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "face-id-895f3",
    storageBucket: "face-id-895f3.appspot.com",
    messagingSenderId: "765216565408",
    appId: "1:765216565408:web:aabd0af2c49fdfd51a01c5",
    measurementId: "G-RYR9JRMR3G"
  };

 const app = initializeApp(firebaseConfig);
 const database = getDatabase(app);
 const auth = getAuth();

document.getElementById('signInButton').addEventListener('click', (event) => {

event.preventDefault()
const signInEmail = document.getElementById('signInEmail').value
const signInPassword = document.getElementById('signInPassword').value
signInWithEmailAndPassword(auth, signInEmail, signInPassword)
    .then((userCredential) => {
        // Signed in
        console.log(userCredential)
        const user = userCredential.user;
        alert('로그인 되었습니다.')
        window.location.href = "/Mainpage.html";
        // ...
    })
    .catch((error) => {
        const errorCode = error.code;
        const errorMessage = error.message;
        alert('로그인 실패\n이메일 형식과 비밀번호를 확인하세요.')
        alert(errorCode, errorMessage);
    });

})
