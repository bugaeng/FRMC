const express = require('express');
const server = express();
const PORT = process.env.PORT || 3000;
const fs = require('fs');
const http  = require('http');



server.use(express.static(__dirname + "/disease"));
server.use(express.static(__dirname + "/Login")); 
server.use(express.static(__dirname + "/SignUp")); 
server.use(express.static(__dirname + "/map"));
server.use(express.static(__dirname + "/Logo"));
server.use(express.static(__dirname + "/Capture"));
server.use(express.static(__dirname + "/age_prediction"));



server.get('/', (req, res) => { 
  res.sendFile('/intro/main.html')
})

server.get('/disease', (req, res) => {
  res.sendFile('disease/disease_info.html')
})

server.get('/Login', (req, res) => {
  res.sendFile('Login/Login.html')
})

server.get('/SignUp', (req, res) => {
  res.sendFile('/SignUp/SignUp.html')
})

server.get('/map', (req, res) => {
  res.sendFile('map/map.html')
})

server.get('/Logo', (req, res) => {
  res.sendFile('Logo/Logo.html')
})

server.get('/Capture', (req, res) => {
  res.sendFile('Capture/Capture.html')
})

server.get('/age_prediction', (req, res) => {
  res.sendFile('age_prediction/age_prediction.html')
})



server.listen(PORT, () => {
  console.log('Express server listening on port ' + PORT);
});