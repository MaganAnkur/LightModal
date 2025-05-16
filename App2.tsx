/**
 * React Native App for processing static images
 */

import React, {useState, useEffect} from 'react';

import {
  StyleSheet,
  View,
  Text,
  ActivityIndicator,
  Button,
  Alert,
} from 'react-native';
import {
  Tensor,
  TensorflowModel,
  useTensorflowModel,
} from 'react-native-fast-tflite';
import {flattenDeep} from 'lodash';

// Direct import for the Float32Array data
import rgbImageData from './float32_rgb_image.json';
// Direct import for the Float32Array data BGR
import bgrImageData from './float32_bgr_image.json';

// The imported JSON is a 3D array [height][width][channels]
//const rgbImage3D: number[][][] = rgbImageData as unknown as number[][][];
const bgrImage3D: number[][][] = bgrImageData as unknown as number[][][];
function tensorToString(tensor: Tensor): string {
  return `\n  - ${tensor.dataType} ${tensor.name}[${tensor.shape}]`;
}

function modelToString(model: TensorflowModel): string {
  return (
    `TFLite Model (${model.delegate}):\n` +
    `- Inputs: ${model.inputs.map(tensorToString).join('')}\n` +
    `- Outputs: ${model.outputs.map(tensorToString).join('')}`
  );
}

function App2(): React.JSX.Element {
  const model = useTensorflowModel(require('./model.tflite'));
  const actualModel = model.state === 'loaded' ? model.model : undefined;
  const [loading, setLoading] = useState(false);
  const [detections, setDetections] = useState<any[]>([]);
  const [imageTensor, setImageTensor] = useState<Float32Array | null>(null);
  const [processingError, setProcessingError] = useState<string | null>(null);

  // Load the test image on component mount
  useEffect(() => {
    prepareImageTensor();
  }, []);

  // Create a tensor from the pre-built Float32Array in JSON
  const prepareImageTensor = async (): Promise<void> => {
    try {
      setLoading(true);
      setProcessingError(null);

      console.log('Loading 3D array from JSON file...');

      try {
        // The JSON file contains a 3D array [height][width][channels]
        // Use lodash's flattenDeep to flatten the 3D array into a 1D array
       // const flattened = flattenDeep(rgbImage3D);
        const flattened = flattenDeep(bgrImage3D);
        //const wtf = postprocess(rgbImage3D);
       // console.log('wtf', wtf);
        const rgbBuffer = new Float32Array(flattened);

        const IMAGE_SIZE = 416; // Expected size 416x416x3
        const expectedLength = IMAGE_SIZE * IMAGE_SIZE * 3;

        // Verify dimensions
        if (rgbBuffer.length !== expectedLength) {
          console.warn(
            `Warning: Expected tensor length ${expectedLength}, got ${rgbBuffer.length}`,
          );
        }

        console.log(`Created flattened tensor with ${rgbBuffer.length} values`);
        console.log('First few values:', Array.from(rgbBuffer.slice(0, 12)));

        // Log example pixel values
        console.log('Original top-left pixel:', bgrImage3D[0][0]);
        console.log(
          'Flattened top-left pixel:',
          rgbBuffer[0],
          rgbBuffer[1],
          rgbBuffer[2],
        );

        setImageTensor(rgbBuffer);
      } catch (err) {
        console.error('Error processing JSON data:', err);
        setProcessingError(
          `Error processing image data: ${(err as Error).message}`,
        );
      }
    } catch (error) {
      console.error('Error preparing image tensor:', error);
      setProcessingError(`Error processing image: ${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  // Process the image through the model
  const processImage = async () => {
    if (!actualModel) {
      Alert.alert('Error', 'Model not loaded');
      return;
    }

    if (!imageTensor) {
      Alert.alert('Error', 'Image not processed yet');
      return;
    }

    setLoading(true);
    try {
      // Run the model with the image tensor
      const result = actualModel.runSync([imageTensor]);

      if (result && result.length > 0 && result[0]) {
        console.log('App2.tsx - result[0] length:', result[0].length);

        // Process the output
        const output = result[0] as Float32Array;
        const processedResults = postprocess(output);
        console.log('Detections:', processedResults);
        setDetections(processedResults);
      }
    } catch (error) {
      console.error('Error processing image:', error);
      Alert.alert(
        'Error',
        `Error processing image: ${(error as Error).message}`,
      );
    } finally {
      setLoading(false);
    }
  };

  // Effect to log model info once loaded
  React.useEffect(() => {
    if (actualModel == null) {
      return;
    }
    console.log(`Model loaded! Shape:\n${modelToString(actualModel)}]`);
    if (actualModel.outputs && actualModel.outputs.length > 0) {
      console.log(
        'App2.tsx - Model Output Details:',
        JSON.stringify(actualModel.outputs),
      );
    }
  }, [actualModel]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Static Image Processor</Text>

      {model.state === 'loading' && (
        <ActivityIndicator size="large" color="#0000ff" />
      )}

      {model.state === 'error' && (
        <Text style={styles.error}>
          Failed to load model! {model.error.message}
        </Text>
      )}

      {processingError && <Text style={styles.error}>{processingError}</Text>}

      <View style={styles.buttonContainer}>
        <Button
          title="Reload Image"
          onPress={prepareImageTensor}
          disabled={loading}
        />
        <Button
          title="Process Image"
          onPress={processImage}
          disabled={!actualModel || loading || !imageTensor}
        />
      </View>

      {loading && <ActivityIndicator size="large" color="#0000ff" />}

      {detections.length > 0 && (
        <View style={styles.resultsContainer}>
          <Text style={styles.resultsTitle}>Detection Results:</Text>
          {detections.map((detection, index) => (
            <Text key={index} style={styles.resultItem}>
              {detection.tagName}: {(detection.probability * 100).toFixed(1)}%
              {'\n'}BBox: ({(detection.boundingBox.left * 416).toFixed(1)},
              {(detection.boundingBox.top * 416).toFixed(1)}) -
              {(detection.boundingBox.width * 416).toFixed(1)}x
              {(detection.boundingBox.height * 416).toFixed(1)}
            </Text>
          ))}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
    marginVertical: 20,
  },
  imageContainer: {
    width: '100%',
    height: 250,
    marginVertical: 20,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
    padding: 5,
  },
  image: {
    width: '100%',
    height: '100%',
  },
  error: {
    color: 'red',
    marginVertical: 10,
  },
  resultsContainer: {
    width: '100%',
    marginTop: 10,
    padding: 10,
    backgroundColor: '#f0f0f0',
    borderRadius: 5,
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  resultItem: {
    marginVertical: 5,
    fontSize: 14,
  },
});

// Helper functions
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
  predictionOutput: Float32Array,
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

export default App2;
