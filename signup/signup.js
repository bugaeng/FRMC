 import { initializeApp } from "https://www.gstatic.com/firebasejs/10.5.2/firebase-app.js";
 import { getDatabase , set,ref } from "https://www.gstatic.com/firebasejs/10.5.2/firebase-database.js"; //데이터베이스 링크 안에 추가적인것을 넣기위한 set,ref 
 import { getAuth, createUserWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.5.2/firebase-auth.js";

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

 form.addEventListener('submit',(e)=>{
  e.preventDefault();
  var email;
  var password;
  var username;
  email= document.getElementById('email').value;
  password= document.getElementById('password').value;
  username= document.getElementById('username').value;
  const usernameRef = ref(database, 'user');
  set(usernameRef, {
      "username" : username,
      "email" : email,
      "password" : password
  });

  

  createUserWithEmailAndPassword(auth, email, password )

  .then((userCredential) => {
    const user = userCredential.user;
    alert('회원가입 되었습니다.')
    window.location.href = "/Login/Login.html";

    
  })
  .catch((error) => {
    const errorCode = error.code;
    const errorMessage = error.message;
    
    alert('회원가입 실패\n이메일 형식과 비밀번호를 확인하세요.')
    alert(errorCode, errorMessage);
  });

  });
