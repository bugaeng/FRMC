const usrName = localStorage.getItem("username");
const age = localStorage.getItem("Age");
const weight = localStorage.getItem("weight");
const saveLabel = localStorage.getItem("saveLabel");




document.getElementById('button1').addEventListener('click',e=>{
           
      localStorage.removeItem("username");
      localStorage.removeItem("Age");
      localStorage.removeItem("weight");
      localStorage.removeItem("saveLabel");
     
      logOut.classList.add("hidden");
      outA.classList.add("hidden");
      loginForm.classList.remove("hidden");
      loginForm.classList.add("aaa");
   
      location.reload();
});


