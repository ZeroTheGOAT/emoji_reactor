const STATES = {
  IDLE: 'EMOJI_IDLE',
  FINGER_MOUTH: 'MONKEY_FINGER_MOUTH',
  FINGER_RAISE: 'MONKEY_FINGER_RAISE',
  SMILE: 'EMOJI_SMILE',
  HANDS_UP: 'AIR_HANDS_UP',
  THUMBS_UP: 'THUMBS_UP',
};

const STATE_ASSETS = {
  [STATES.IDLE]: './assets/plain.png',
  [STATES.FINGER_MOUTH]: './assets/monkey_finger_mouth.jpeg',
  [STATES.FINGER_RAISE]: './assets/monkey_finger_raise.jpg',
  [STATES.SMILE]: './assets/smile.jpg',
  [STATES.HANDS_UP]: './assets/air.jpg',
  [STATES.THUMBS_UP]: './assets/thumbsup.png',
};

const STATE_LABELS = {
  [STATES.IDLE]: 'Plain Idle',
  [STATES.FINGER_MOUTH]: 'Finger to Mouth',
  [STATES.FINGER_RAISE]: 'Raised Finger',
  [STATES.SMILE]: 'Big Smile',
  [STATES.HANDS_UP]: 'Hands Above Head',
  [STATES.THUMBS_UP]: 'Thumbs Up',
};

const CONFIG = {
  fingerMouthDistance: 0.15,
  indexExtendedDelta: 0.1,
  indexHighDelta: 0.15,
  middleRelaxDelta: 0.05,
  mouthOpenThreshold: 0.02,
  smileAspectThreshold: 0.25,
  handsOnHeadY: 0.55,
  handsPoseMargin: 0.02,
  thumbUpDelta: 0.08,
};

let previousState = STATES.IDLE;

const HAND_LANDMARKS = {
  WRIST: 0,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_TIP: 8,
  MIDDLE_TIP: 12,
  RING_TIP: 16,
};

const FACE_LANDMARKS = {
  UPPER_LIP: 13,
  LOWER_LIP: 14,
  MOUTH_LEFT: 61,
  MOUTH_RIGHT: 291,
};

const POSE_LANDMARKS = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
};

function distance2D(a, b) {
  return Math.hypot((a?.x ?? 0) - (b?.x ?? 0), (a?.y ?? 0) - (b?.y ?? 0));
}

function getHands(results) {
  const hands = [];
  if (results.leftHandLandmarks) hands.push(results.leftHandLandmarks);
  if (results.rightHandLandmarks) hands.push(results.rightHandLandmarks);
  return hands;
}

function getMouthInfo(faceLandmarks) {
  if (!faceLandmarks) return null;
  const upperLip = faceLandmarks[FACE_LANDMARKS.UPPER_LIP];
  const lowerLip = faceLandmarks[FACE_LANDMARKS.LOWER_LIP];
  const mouthLeft = faceLandmarks[FACE_LANDMARKS.MOUTH_LEFT];
  const mouthRight = faceLandmarks[FACE_LANDMARKS.MOUTH_RIGHT];
  const mouthHeight = distance2D(upperLip, lowerLip);
  const mouthWidth = distance2D(mouthLeft, mouthRight);
  const aspectRatio = mouthWidth ? mouthHeight / mouthWidth : 0;
  const cornersUp =
    mouthLeft?.y < (upperLip?.y ?? 1) - 0.01 && mouthRight?.y < (upperLip?.y ?? 1) - 0.01;
  return {
    centerX: (mouthLeft.x + mouthRight.x) / 2,
    centerY: (upperLip.y + lowerLip.y) / 2,
    height: mouthHeight,
    aspectRatio,
    cornersUp,
  };
}

function detectFingerToMouth(hands, mouthInfo) {
  if (!hands.length || !mouthInfo) return null;
  for (const hand of hands) {
    const indexTip = hand[HAND_LANDMARKS.INDEX_TIP];
    const distance = Math.hypot(indexTip.x - mouthInfo.centerX, indexTip.y - mouthInfo.centerY);
    if (distance < CONFIG.fingerMouthDistance) {
      return STATES.FINGER_MOUTH;
    }
  }
  return null;
}

function detectRaisedFinger(hands) {
  for (const hand of hands) {
    const indexTip = hand[HAND_LANDMARKS.INDEX_TIP];
    const indexMCP = hand[HAND_LANDMARKS.INDEX_MCP];
    const middleTip = hand[HAND_LANDMARKS.MIDDLE_TIP];
    const wrist = hand[HAND_LANDMARKS.WRIST];
    const indexExtended = indexTip.y < indexMCP.y - CONFIG.indexExtendedDelta;
    const indexHigh = indexTip.y < wrist.y - CONFIG.indexHighDelta;
    const middleRelaxed = middleTip.y > indexTip.y + CONFIG.middleRelaxDelta;
    if (indexExtended && indexHigh && middleRelaxed) {
      return STATES.FINGER_RAISE;
    }
  }
  return null;
}

function detectThumbsUp(hands) {
  for (const hand of hands) {
    const thumbTip = hand[HAND_LANDMARKS.THUMB_TIP];
    const thumbIP = hand[HAND_LANDMARKS.THUMB_IP];
    const wrist = hand[HAND_LANDMARKS.WRIST];
    const indexTip = hand[HAND_LANDMARKS.INDEX_TIP];
    const middleTip = hand[HAND_LANDMARKS.MIDDLE_TIP];
    const ringTip = hand[HAND_LANDMARKS.RING_TIP];
    const thumbUp = thumbTip.y < wrist.y - CONFIG.thumbUpDelta && thumbTip.y < thumbIP.y;
    const fingersFolded =
      indexTip.y > wrist.y - 0.02 && middleTip.y > wrist.y - 0.02 && ringTip.y > wrist.y - 0.02;
    if (thumbUp && fingersFolded) {
      return STATES.THUMBS_UP;
    }
  }
  return null;
}

function detectSmile(mouthInfo) {
  if (mouthInfo && mouthInfo.aspectRatio > CONFIG.smileAspectThreshold) {
    return STATES.SMILE;
  }
  return null;
}

function detectHandsUp(hands, poseLandmarks) {
  let poseHandsUp = false;
  if (poseLandmarks) {
    const leftShoulder = poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER];
    const rightShoulder = poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER];
    const leftWrist = poseLandmarks[POSE_LANDMARKS.LEFT_WRIST];
    const rightWrist = poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST];
    if (
      (leftWrist?.y ?? 1) < (leftShoulder?.y ?? 1) - CONFIG.handsPoseMargin ||
      (rightWrist?.y ?? 1) < (rightShoulder?.y ?? 1) - CONFIG.handsPoseMargin
    ) {
      poseHandsUp = true;
    }
  }

  let handsUp = false;
  if (hands.length) {
    const wristPositions = hands.map((hand) => hand[HAND_LANDMARKS.WRIST]?.y ?? 1);
    handsUp = wristPositions.length && wristPositions.every((y) => y < CONFIG.handsOnHeadY);
  }

  return poseHandsUp || handsUp ? STATES.HANDS_UP : null;
}

export function evaluateState(results) {
  const hands = getHands(results);
  const mouthInfo = getMouthInfo(results.faceLandmarks);
  const poseLandmarks = results.poseLandmarks ?? null;

  const detectors = [
    () => detectFingerToMouth(hands, mouthInfo),
    () => detectRaisedFinger(hands),
    () => detectThumbsUp(hands),
    () => detectHandsUp(hands, poseLandmarks),
    () => detectSmile(mouthInfo),
  ];

  for (const detector of detectors) {
    const state = detector();
    if (state) {
      previousState = state;
      return { state, asset: STATE_ASSETS[state], label: STATE_LABELS[state] };
    }
  }

  previousState = STATES.IDLE;
  return { state: STATES.IDLE, asset: STATE_ASSETS[STATES.IDLE], label: STATE_LABELS[STATES.IDLE] };
}

export { STATES, STATE_ASSETS, STATE_LABELS };
