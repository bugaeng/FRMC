var form = document.querySelector('.form')
form.onmousemove = function(e) {
  var x = e.pageX - form.offsetLeft
  var y = e.pageY - form.offsetTop
  form.style.setProperty('--x', x + 'px')
  form.style.setProperty('--y', y + 'px')
}

var form2 = document.querySelector('.form2')
form2.onmousemove = function(e) {
  var x = e.pageX - form2.offsetLeft
  var y = e.pageY - form2.offsetTop
  form2.style.setProperty('--x', x + 'px')
  form2.style.setProperty('--y', y + 'px')
}

var form3 = document.querySelector('.form3')
form3.onmousemove = function(e) {
  var x = e.pageX - form3.offsetLeft
  var y = e.pageY - form3.offsetTop
  form3.style.setProperty('--x', x + 'px')
  form3.style.setProperty('--y', y + 'px')
}