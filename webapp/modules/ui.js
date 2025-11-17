import { evaluateState } from './gestures.js';

const assetCache = new Map();

function preloadAsset(src) {
  if (assetCache.has(src)) return assetCache.get(src);
  const img = new Image();
  img.src = src;
  assetCache.set(src, img);
  return img;
}

export function setupUI({ stateNameEl, outputImageEl }) {
  return {
    update(results) {
      if (!results) return;
      const { state, asset, label } = evaluateState(results);
      stateNameEl.textContent = label;
      if (outputImageEl.src !== asset) {
        preloadAsset(asset);
        outputImageEl.src = asset;
      }
    },
  };
}
