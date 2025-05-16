const GRID_SIZE = 13;
const NUM_ANCHORS = 5;
const NUM_CLASSES = 4;
const VALUES_PER_ANCHOR = 5 + NUM_CLASSES; // 9
const ANCHORS = [
  [0.57273, 0.677385],
  [1.87446, 2.06253],
  [3.33843, 5.47434],
  [7.88282, 3.52778],
  [9.77052, 9.16828],
];
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}
function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map((x: number) => Math.exp(x - max));
  const sum = exps.reduce((a: number, b: number) => a + b, 0);
  return exps.map((e: number) => e / sum);
}
interface Detection {
  class: number;
  confidence: number;
  box: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}
function parseYoloOutput(
  output: Float32Array,
  confThreshold = 0.5,
): Detection[] {
  'worklet';
  const detections: Detection[] = [];
  for (let i = 0; i < GRID_SIZE; i++) {
    for (let j = 0; j < GRID_SIZE; j++) {
      for (let a = 0; a < NUM_ANCHORS; a++) {
        const offset =
          i * GRID_SIZE * NUM_ANCHORS * VALUES_PER_ANCHOR +
          j * NUM_ANCHORS * VALUES_PER_ANCHOR +
          a * VALUES_PER_ANCHOR;
        const x = sigmoid(output[offset + 0]);
        const y = sigmoid(output[offset + 1]);
        const w = output[offset + 2];
        const h = output[offset + 3];
        const objectness = sigmoid(output[offset + 4]);
        const classScores: number[] = [];
        for (let c = 0; c < NUM_CLASSES; c++) {
          classScores.push(output[offset + 5 + c]);
        }
        const classProbs = softmax(classScores);
        const maxClassProb = Math.max(...classProbs);
        const classIdx = classProbs.indexOf(maxClassProb);
        const confidence = objectness * maxClassProb;

        // Debug log for first cell, first 2 anchors
        if (i === 0 && j === 0 && a < 2) {
          console.log(
            `Cell[${i},${j}], Anchor ${a}: objectness=${objectness.toFixed(
              4,
            )}, maxClassProb=${maxClassProb.toFixed(
              4,
            )}, confidence=${confidence.toFixed(4)}`,
          );
        }

        if (confidence > confThreshold) {
          const anchor = ANCHORS[a];
          const bx = (j + x) / GRID_SIZE;
          const by = (i + y) / GRID_SIZE;
          const bw = (Math.exp(w) * anchor[0]) / GRID_SIZE;
          const bh = (Math.exp(h) * anchor[1]) / GRID_SIZE;
          detections.push({
            class: classIdx,
            confidence,
            box: {
              x1: bx - bw / 2,
              y1: by - bh / 2,
              x2: bx + bw / 2,
              y2: by + bh / 2,
            },
          });
        }
      }
    }
  }
  return detections;
}
