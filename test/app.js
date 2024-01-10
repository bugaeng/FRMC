const express = require('express')
const app = express()
const port = 3000

app.get('/', (req, res) => {
  res.sendFile('문서.html')
})

app.get('/patient', (req, res) => {
    res.sendFile('patient.html')
})

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})