var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
var size = 500;

var rStep = 8;
var minDist = 8;
var N = 20;
var n = 0;

function angle(d, r) {
    return Math.acos(1 - (d * d) / (2 * r * r));
}

function drawPoint(x, y) {
    ctx.beginPath();
    ctx.arc((size / 2 ) + x, (size/2) + y, 2, 0, Math.PI * 2);
    ctx.strokeStyle = "#FF00FF";
    ctx.fillStyle = "#FF00FF"
    ctx.stroke();
    ctx.fill();
}

function drawCircle(r) {
    ctx.beginPath();
    ctx.arc(size/2, size/2, r, 0, Math.PI * 2);
    ctx.strokeStyle = "#FFFFFF";
    ctx.stroke();
}

function background() {
    ctx.beginPath();
    ctx.fillStyle = "#000000";
    ctx.rect(0, 0, 500, 500);
    ctx.fill();
}

background();

var i = 0;
while(n < N) {
    i++;
    let cR = rStep * i;
    drawCircle(cR);
    let alpha = angle(minDist, cR);
    let points = Math.floor(Math.PI * 2 / alpha);
    points = Math.min(points, N - n);
    alpha = Math.PI * 2 / points;

    for (let z = 0; z < points; z++) {
        let x = Math.cos(alpha * z) * cR;
        let y = Math.sin(alpha * z) * cR;
        drawPoint(x, y);
        console.log(x, y);  
    }

    n += points;

}
