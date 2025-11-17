import { initializeDetectors } from './modules/detectors.js';
import { setupUI } from './modules/ui.js';

(async () => {
  const statusEl = document.querySelector('.status-text');
  const stateNameEl = document.querySelector('#state-name');
  const outputImageEl = document.querySelector('#output-image');
  const videoEl = document.querySelector('#camera');

  try {
    statusEl.textContent = 'Requesting camera…';
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    videoEl.srcObject = stream;

    await new Promise((resolve) => {
      videoEl.onloadedmetadata = () => {
        videoEl.play();
        resolve();
      };
    });

    statusEl.textContent = 'Initializing detectors…';
    const detectors = await initializeDetectors();
    const ui = setupUI({ stateNameEl, outputImageEl });

    detectors.holistic.onResults((results) => {
      ui.update(results);
    });

    statusEl.textContent = 'Ready! Perform gestures to change the meme.';

    const camera = new Camera(videoEl, {
      onFrame: async () => {
        await detectors.holistic.send({ image: videoEl });
      },
      width: 640,
      height: 480,
    });

    camera.start();
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Camera access or detector init failed. Check console logs.';
  }
})();
