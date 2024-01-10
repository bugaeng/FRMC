document.addEventListener('DOMContentLoaded', async () => {
    const URL = "./model/";
    let model, webcam, mychart, maxPredictions;

    const modelURL = URL + "model.json";
    const metadataURL = URL + "metadata.json";
    model = await tmImage.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();

    const flip = true;
    webcam = new tmImage.Webcam(600, 300, flip);
    await webcam.setup();
    await webcam.play();
    window.requestAnimationFrame(loop);

    document.getElementById("webcam-container").appendChild(webcam.canvas);
    mychart = document.getElementById("myChart");
    for (let i = 0; i < maxPredictions; i++) {
        mychart.appendChild(document.createElement("div"));
    }

    // Create the chart outside the loop
    var ctx = document.getElementById('myChart').getContext('2d');
    var chart = new Chart(ctx, {
        type: 'bar',
        data: {
            // labels: [model.getClassLabels()[0], model.getClassLabels()[1], model.getClassLabels()[2], model.getClassLabels()[3], model.getClassLabels()[4], model.getClassLabels()[5], model.getClassLabels()[6], model.getClassLabels()[7]],
            labels: ['akiec','bcc','bkl','df','mel','vasc', 'normal'],
            datasets: [{
                label: 'Dataset Label',
                backgroundColor: ['rgb(255, 0, 0)', 'rgb(75, 192, 192)', 'rgb(255, 228, 0)', 'rgb(0, 84, 255)', 'rgb(255, 228, 0)', 'rgb(95, 9, 255)', 'rgb(29, 219, 22)'],
                borderColor: ['rgb(255, 99, 132)', 'rgb(75, 192, 192)', 'rgb(255, 228, 0)', 'rgb(0, 84, 255)', 'rgb(255, 228, 0)', 'rgb(95, 9, 255)', 'rgb(255, 255, 255)'],
                data: [0, 0]
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    color: 'white',
                },
                x: {
                    color: 'white',
                }

            },
            legend: {
                display: true,
                position: 'bottom',

            }
        }
    });

    async function loop() {
        webcam.update();
        await predict();
        window.requestAnimationFrame(loop);
    }

    async function predict() {
        const prediction = await model.predict(webcam.canvas);

        // Update the labelContainer
        for (let i = 0; i < maxPredictions; i++) {
            const classPrediction =
                prediction[i].className + ": " + prediction[i].probability.toFixed(2);
                mychart.childNodes[i].innerHTML = classPrediction;
        }

        // 차트 추가 부분
        chart.data.datasets[0].data = [
            prediction[0].probability.toFixed(2),
            prediction[1].probability.toFixed(2),
            prediction[2].probability.toFixed(2),
            prediction[3].probability.toFixed(2),
            prediction[4].probability.toFixed(2),
            prediction[5].probability.toFixed(2),
            prediction[6].probability.toFixed(2)
        ];
        // 차트만 추가하면 캠이 멈춘다. 계속해서 업데이트 해주는 부분
        chart.update();
    }
});
