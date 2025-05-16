const data = require('./raw_model_output_py.json');
// Constants matching Python implementation

//const labels = ['oropharynx', 'tonsil region a', 'tonsil region b', 'uvula'];
const ANCHORS = [
  [0.573, 0.677],
  [1.87, 2.06],
  [3.34, 5.47],
  [7.88, 3.53],
  [9.77, 9.17],
];
const IOU_THRESHOLD = 0.45;

// Helper functions
function logistic(x) {
  return x > 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

function reshapeTo3D(flatArray, dim1 = 13, dim2 = 13, dim3 = 45) {
  const result = [];
  for (let i = 0; i < dim1; i++) {
    const slice = [];
    for (let j = 0; j < dim2; j++) {
      const row = [];
      for (let k = 0; k < dim3; k++) {
        const index = i * dim2 * dim3 + j * dim3 + k;
        row.push(flatArray[index]);
      }
      slice.push(row);
    }
    result.push(slice);
  }
  return result;
}

function extractBB(predictionOutput, anchors) {
  const height = predictionOutput.length;
  const width = predictionOutput[0].length;
  const numAnchor = anchors.length;
  const channels = predictionOutput[0][0].length;
  const numClass = channels / numAnchor - 5;

  // Extract bounding box information
  const boxes = [];
  const classProbs = [];

  for (let i = 0; i < height; i++) {
    for (let j = 0; j < width; j++) {
      for (let a = 0; a < numAnchor; a++) {
        const offset = a * (numClass + 5);

        // Calculate x, y, w, h - Note the order of operations matches Python
        const x = (logistic(predictionOutput[i][j][offset]) + j) / width;
        const y = (logistic(predictionOutput[i][j][offset + 1]) + i) / height;
        const w =
          (Math.exp(predictionOutput[i][j][offset + 2]) * anchors[a][0]) /
          width;
        const h =
          (Math.exp(predictionOutput[i][j][offset + 3]) * anchors[a][1]) /
          height;

        // Convert to top-left coordinates
        const x1 = x - w / 2;
        const y1 = y - h / 2;

        boxes.push([x1, y1, w, h]);

        // Get objectness score
        const objectness = logistic(predictionOutput[i][j][offset + 4]);

        // Get class probabilities
        const classScores = [];
        for (let c = 0; c < numClass; c++) {
          classScores.push(predictionOutput[i][j][offset + 5 + c]);
        }

        // Softmax with numerical stability
        const maxScore = Math.max(...classScores);
        const expScores = classScores.map(score => Math.exp(score - maxScore));
        const sumExpScores = expScores.reduce((a, b) => a + b, 0);
        const probs = expScores.map(
          score => (score / sumExpScores) * objectness,
        );

        classProbs.push(probs);
      }
    }
  }

  return [boxes, classProbs];
}

function nonMaximumSuppression(
  boxes,
  classProbs,
  maxDetections,
  probThreshold,
) {
  const maxProbs = classProbs.map(probs => Math.max(...probs));
  const maxClasses = classProbs.map(probs => probs.indexOf(Math.max(...probs)));

  const areas = boxes.map(box => box[2] * box[3]);

  const selectedBoxes = [];
  const selectedClasses = [];
  const selectedProbs = [];

  while (selectedBoxes.length < maxDetections) {
    const i = maxProbs.indexOf(Math.max(...maxProbs));
    if (maxProbs[i] < probThreshold) break;

    selectedBoxes.push(boxes[i]);
    selectedClasses.push(maxClasses[i]);
    selectedProbs.push(maxProbs[i]);

    const box = boxes[i];
    const otherIndices = [...Array(boxes.length).keys()].filter(
      idx => idx !== i,
    );
    const otherBoxes = otherIndices.map(idx => boxes[idx]);

    // Calculate IoU
    const x1 = otherBoxes.map(other => Math.max(box[0], other[0]));
    const y1 = otherBoxes.map(other => Math.max(box[1], other[1]));
    const x2 = otherBoxes.map(other =>
      Math.min(box[0] + box[2], other[0] + other[2]),
    );
    const y2 = otherBoxes.map(other =>
      Math.min(box[1] + box[3], other[1] + other[3]),
    );

    const w = x2.map((x, idx) => Math.max(0, x - x1[idx]));
    const h = y2.map((y, idx) => Math.max(0, y - y1[idx]));

    const overlapArea = w.map((width, idx) => width * h[idx]);
    const iou = overlapArea.map(
      (area, idx) => area / (areas[i] + areas[otherIndices[idx]] - area),
    );

    // Update probabilities for overlapping boxes
    const overlappingIndices = otherIndices.filter(
      (_, idx) => iou[idx] > IOU_THRESHOLD,
    );
    overlappingIndices.push(i);

    overlappingIndices.forEach(idx => {
      classProbs[idx][maxClasses[i]] = 0;
      maxProbs[idx] = Math.max(...classProbs[idx]);
      maxClasses[idx] = classProbs[idx].indexOf(maxProbs[idx]);
    });
  }

  return [selectedBoxes, selectedClasses, selectedProbs];
}

function postprocess(
  predictionOutput,
  labels = ['oropharynx', 'tonsil region a', 'tonsil region b', 'uvula'],
  probThreshold = 0.8,
  maxDetections = 20,
) {
  // Reshape the output to 3D if it's flat
  const output3D = Array.isArray(predictionOutput[0])
    ? predictionOutput
    : reshapeTo3D(predictionOutput);

  // Extract bounding boxes and class probabilities
  const [boxes, classProbs] = extractBB(output3D, ANCHORS);

  // Remove overlapping boxes
  const [selectedBoxes, selectedClasses, selectedProbs] = nonMaximumSuppression(
    boxes,
    classProbs,
    maxDetections,
    probThreshold,
  );

  // Format results to match Python output
  return selectedBoxes.map((box, i) => ({
    probability: Number(selectedProbs[i].toFixed(8)),
    tagId: selectedClasses[i],
    tagName: labels[selectedClasses[i]],
    boundingBox: {
      left: Number(box[0].toFixed(8)),
      top: Number(box[1].toFixed(8)),
      width: Number(box[2].toFixed(8)),
      height: Number(box[3].toFixed(8)),
    },
  }));
}

// Export the main function
module.exports = {
  postprocess,
  ANCHORS,
  IOU_THRESHOLD,
};
/* 
// Example usage

// Debug: Check if we have data
console.log('Raw data length:', data.length);
console.log('First few values of raw data:', data.slice(0, 10));

// Reshape and check the 3D output
const output3D = reshapeTo3D(data);
console.log('3D output shape:', {
  height: output3D.length,
  width: output3D[0].length,
  channels: output3D[0][0].length,
});

// Extract bounding boxes and check
const [boxes, classProbs] = extractBB(output3D, ANCHORS);
console.log('Number of boxes extracted:', boxes.length);
console.log('Number of class probabilities:', classProbs.length);

// Debug: Check confidence scores before NMS
const maxProbs = classProbs.map(probs => Math.max(...probs));
console.log('Max confidence scores before NMS:', {
  min: Math.min(...maxProbs),
  max: Math.max(...maxProbs),
  mean: maxProbs.reduce((a, b) => a + b, 0) / maxProbs.length,
});

// Post-process the results
const results = postprocess(data, labels, 0.8);

// Debug: Check if we got any results
console.log('\nNumber of detections:', results.length);

// Print the results in a readable format
console.log('\nDetection Results:');
results.forEach((detection, index) => {
  console.log(`\nDetection ${index + 1}:`);
  console.log(`Class: ${detection.tagName}`);
  console.log(`Confidence: ${(detection.probability * 100).toFixed(2)}%`);
  console.log(`Bounding Box:`);
  console.log(`  Left: ${detection.boundingBox.left.toFixed(4)}`);
  console.log(`  Top: ${detection.boundingBox.top.toFixed(4)}`);
  console.log(`  Width: ${detection.boundingBox.width.toFixed(4)}`);
  console.log(`  Height: ${detection.boundingBox.height.toFixed(4)}`);
});
 */