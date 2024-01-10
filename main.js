/*

const express = require('express');

const app = express();
const port = 3000;
const fs = require('fs'); // 파일로드 
const http  = require('http');

app.use(express.static(__dirname + "/patient")); //자동으로 여기잡힘
app.use(express.static(__dirname + "/camera")); //자동으로 여기잡힘
app.use(express.static(__dirname + "/final")); //자동으로 여기잡힘
app.use(express.static(__dirname + "/testimage")); //절대경로들
app.use(express.static(__dirname + "/map")); //절대경로들
app.use(express.static(__dirname + "/start")); //절대경로들



app.get('/', (req, res) => {
    res.sendFile('C:/Users/202-4/Desktop/프로젝트/public/patient/index.html')

  })
  
  app.get('/camera.html', (req, res) => {
      res.sendFile('C:/Users/202-4/Desktop/프로젝트/public/camera/camera.html')
      
  })


  app.get('/final', (req, res) => {
    res.sendFile('C:/Users/202-4/Desktop/프로젝트/public/final/final.html')
    
})


app.get('/testimage', (req, res) => { //테스트이미지 모델위치
    res.sendFile('C:/Users/202-4/Desktop/프로젝트/public/testimage/')
    
})

app.get('/testimage', (req, res) => { //테스트이미지 모델위치
  res.sendFile('C:/Users/202-4/Desktop/프로젝트/public/map/map.html')
  
})

  
app.listen(port, () => {
    
  console.log(`Example app listening on port ${port}`)
})

//init 하고 node mian.js
*/