export async function initializeDetectors() {
  const holistic = new Holistic({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
  });

  holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    refineFaceLandmarks: true,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6,
  });

  return { holistic };
}
