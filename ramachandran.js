// Incremental Ramachandran (phi vs psi) plot rendered on stacked canvases.
//
// Two layers: a persistent "cloud" canvas that accumulates every visited
// (phi, psi) over the simulation (axes drawn once), and a transparent overlay
// canvas that shows only the CURRENT position as a bright dot, cleared and
// redrawn each frame. reset() wipes the cloud and restarts the accumulation.

export function createRamachandran(cloudCanvas, dotCanvas) {
  const W = cloudCanvas.width;
  const H = cloudCanvas.height;
  const ctxC = cloudCanvas.getContext("2d");
  const ctxD = dotCanvas.getContext("2d");

  const padL = 26, padB = 22, padT = 8, padR = 8;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const x0 = padL, y0 = padT;

  // angle (deg, -180..180) -> pixel.  phi on x, psi on y (psi=+180 at top)
  const mapx = (a) => x0 + ((a + 180) / 360) * plotW;
  const mapy = (a) => y0 + ((180 - a) / 360) * plotH;

  function drawAxes() {
    ctxC.clearRect(0, 0, W, H);
    ctxC.fillStyle = "#0b0f18";
    ctxC.fillRect(x0, y0, plotW, plotH);
    ctxC.strokeStyle = "#2b3349";
    ctxC.lineWidth = 1;
    ctxC.strokeRect(x0, y0, plotW, plotH);
    ctxC.beginPath();
    for (const a of [-90, 0, 90]) {
      ctxC.moveTo(mapx(a), y0); ctxC.lineTo(mapx(a), y0 + plotH);
      ctxC.moveTo(x0, mapy(a)); ctxC.lineTo(x0 + plotW, mapy(a));
    }
    ctxC.stroke();

    ctxC.fillStyle = "#8b94a8";
    ctxC.font = "9px -apple-system, sans-serif";
    ctxC.textAlign = "center";
    for (const a of [-180, 0, 180]) ctxC.fillText(a, mapx(a), y0 + plotH + 12);
    ctxC.textAlign = "right";
    for (const a of [-180, 0, 180]) ctxC.fillText(a, x0 - 3, mapy(a) + 3);
    ctxC.textAlign = "center";
    ctxC.fillText("φ", x0 + plotW / 2, H - 1);
    ctxC.save();
    ctxC.translate(7, y0 + plotH / 2);
    ctxC.rotate(-Math.PI / 2);
    ctxC.fillText("ψ", 0, 0);
    ctxC.restore();
  }

  function reset() {
    drawAxes();
    ctxD.clearRect(0, 0, W, H);
  }

  // add one visited point to the persistent cloud
  function addPoint(phi, psi) {
    if (!Number.isFinite(phi) || !Number.isFinite(psi)) return;
    ctxC.fillStyle = "rgba(91,157,255,0.20)";
    ctxC.beginPath();
    ctxC.arc(mapx(phi), mapy(psi), 1.7, 0, 2 * Math.PI);
    ctxC.fill();
  }

  // draw the current position on the overlay (clears previous current dot)
  function showCurrent(phi, psi) {
    ctxD.clearRect(0, 0, W, H);
    if (!Number.isFinite(phi) || !Number.isFinite(psi)) return;
    const x = mapx(phi), y = mapy(psi);
    ctxD.beginPath();
    ctxD.arc(x, y, 4.5, 0, 2 * Math.PI);
    ctxD.fillStyle = "#ff5b6e";
    ctxD.fill();
    ctxD.lineWidth = 1.5;
    ctxD.strokeStyle = "#ffffff";
    ctxD.stroke();
  }

  reset();
  return { reset, addPoint, showCurrent };
}
