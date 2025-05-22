/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 */

import {
  AlphaType,
  Canvas,
  ColorType,
  Image,
  PaintStyle,
  Rect,
  Skia,
} from '@shopify/react-native-skia';

import React from 'react';

import {StyleSheet, View, Text, ActivityIndicator} from 'react-native';
import {
  Tensor,
  TensorflowModel,
  useTensorflowModel,
} from 'react-native-fast-tflite';
import {
  useSharedValue,
  useAnimatedReaction,
  runOnJS,
} from 'react-native-reanimated';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  useCameraFormat,
  useSkiaFrameProcessor,
} from 'react-native-vision-camera';
import {useRunOnJS} from 'react-native-worklets-core';
import {useResizePlugin} from 'vision-camera-resize-plugin';

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

function App(): React.JSX.Element {
  const {hasPermission, requestPermission} = useCameraPermission();
  const device = useCameraDevice('back');
  // from https://www.kaggle.com/models/tensorflow/efficientdet/frameworks/tfLite
  const model = useTensorflowModel(require('./model.tflite'));
  const actualModel = model.state === 'loaded' ? model.model : undefined;
  const [isCameraActive, setIsCameraActive] = React.useState(true);
  //const [displayDetections, setDisplayDetections] = React.useState([]);

  // Create shared value for detections
  //const sharedDetections = useSharedValue([]);

  /*  useAnimatedReaction(
    () => sharedDetections.value,
    (result) => {
      // Use runOnJS to call setState from worklet
      runOnJS(setDisplayDetections)(result || []);
    }
  ); */

  /*  const handleCameraActive = useRunOnJS(() => {
    setIsCameraActive(false);
  }, []);
 */
  const processOutput = useRunOnJS((output: Float32Array) => {
    const detections = postprocess(output); // Drastically lowered threshold for debugging

    console.log('Detections:', JSON.stringify(detections, null, 2));

    //sharedDetections.value = detections;
    //setDisplayDetections(detections);
    /*   const topDetections = detections
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 10);
    console.log('Top detections:');
    topDetections.forEach((det, idx) => {
      console.log(`Detection ${idx + 1}:`);
      console.log(`  Class: ${det.class}`);
      console.log(`  Confidence: ${(det.confidence * 100).toFixed(1)}%`);
      console.log(
        `  Box: (${(det.box.x1 * 416).toFixed(1)}, ${(det.box.y1 * 416).toFixed(
          1,
        )}) to (${(det.box.x2 * 416).toFixed(1)}, ${(det.box.y2 * 416).toFixed(
          1,
        )})`,
      );
    }); */
    /*  if (topDetections.length >= 30) {
      console.log('Would show detection summary alert for 10+ detections.');
      handleCameraActive();
    } */
  }, []);
  /* 
  const makeSkiaImage = useRunOnJS((resized: Float32Array) => {
    // log out first row of image (and 1st pixel R value of the second row)
    console.log('first skia row', resized.slice(0, 416 * 3 + 1));
    // Verify the input data

    console.log('makeSkiaImage received data:', {
      type: resized.constructor.name,
      length: resized.length,
      firstFewValues: Array.from(resized.slice(0, 10)),
    });

    // Convert Float32Array to Uint8Array
    const uint8Array = new Uint8Array(resized.length);
    for (let i = 0; i < resized.length; i++) {
      uint8Array[i] = Math.min(255, Math.max(0, Math.round(resized[i] * 255)));
    }

    // Convert RGB to RGBA
    const rgbaArray = new Uint8Array(416 * 416 * 4);
    for (let i = 0, j = 0; i < uint8Array.length; i += 3, j += 4) {
      rgbaArray[j] = uint8Array[i]; // R
      rgbaArray[j + 1] = uint8Array[i + 1]; // G
      rgbaArray[j + 2] = uint8Array[i + 2]; // B
      rgbaArray[j + 3] = 255; // A (fully opaque)
    }
    // Log the final uint8Array
    console.log(
      'Final uint8Array first 10 values:',
      Array.from(uint8Array.slice(0, 10)),
    );

    // Rest of your code...
    // const buffer = uint8Array.buffer.slice(0);
    const pixels = Skia.Data.fromBytes(new Uint8Array(rgbaArray));

    console.log('Skia.Data created, buffer size:', rgbaArray.byteLength);
    try {
      const image = Skia.Image.MakeImage(
        {
          width: 416,
          height: 416,
          alphaType: AlphaType.Unpremul,
          colorType: ColorType.RGB_888x,
        },
        pixels,
        416 * 3, // bytes per row (width * channels)
      );

      console.log('Skia.Image created:', image ? 'success' : 'failed');

      if (image) {
        setResizedImage(image);
        console.log('resizedImage state updated');
      } else {
        console.error('Failed to create Skia image');
      }
    } catch (error) {
      console.error('Error creating Skia image:', error);
    }
  }, []);
 */
  React.useEffect(() => {
    if (actualModel == null) {
      return;
    }
    console.log(`Model loaded! Shape:\n${modelToString(actualModel)}]`);
    if (actualModel.outputs && actualModel.outputs.length > 0) {
      console.log(
        'App.tsx - Model Output Details from plugin:',
        JSON.stringify(actualModel.outputs),
      );
    }
  }, [actualModel]);

  const {resize} = useResizePlugin();

  const frameProcessor = useFrameProcessor(
    frame => {
      'worklet';
      if (actualModel == null) {
        return;
      }

      const resized = resize(frame, {
        scale: {
          width: 416,
          height: 416,
        },

        pixelFormat: 'rgb',
        dataType: 'float32',
      });

      // console.log('🔍 resized RGB[0–9]:', resized.slice(0, 10));
      // const min = Math.min(...resized.slice(0, 1000));
      // const max = Math.max(...resized.slice(0, 1000));
      // console.log(`📏 resized RGB range: min=${min}, max=${max}`);

      const scaled = new Float32Array(resized.length);
      for (let i = 0; i < resized.length; i++) {
        scaled[i] = resized[i] * 255;
      }

      // console.log('🎛 scaled RGB[0–9]:', scaled.slice(0, 10));

      // Convert RGB → BGR
      /*  const bgr = new Float32Array(resized.length);
      for (let i = 0; i < resized.length; i += 3) {
        bgr[i] = resized[i + 2]; // B
        bgr[i + 1] = resized[i + 1]; // G
        bgr[i + 2] = resized[i]; // R
      } */
      // Convert RGB → BGR
      const bgr = new Float32Array(scaled.length);
      for (let i = 0; i < scaled.length; i += 3) {
        bgr[i] = scaled[i + 2]; // B
        bgr[i + 1] = scaled[i + 1]; // G
        bgr[i + 2] = scaled[i]; // R
      }

      // Optional: Log first row
      // console.log('Runtime BGR slice [0..29]:', bgr.slice(0, 30));

      // log out first row of image (and 1st pixel R value of the second row)
      // console.log('first row plus 1px', resized.slice(0, 416 * 3 + 1));
      //const result = actualModel.runSync([resized]);
      const result = actualModel.runSync([bgr]);
      /* if (result && result.length > 0 && result[0]) {
        console.log(
          'App.tsx - Frame processor - result[0] length:',
          result[0].length,
        );
        console.log('whole result', result);
      } */

      try {
        const output = result[0] as Float32Array; // Ensure correct type
        //console.log('🔬 Raw output (first 20):', output.slice(0, 20));
        //processOutput(output);
        const detections = postprocess(output);
        console.log('New Detections:', JSON.stringify(detections, null, 2));
        //frame.render();
        /*  for (const detection of detections) {
          const paint = Skia.Paint();
          paint.setColor(Skia.Color('red'));
          paint.setStyle(PaintStyle.Stroke);
          paint.setStrokeWidth(2);

          // Use a scale factor to make boxes smaller
          const scaleFactor = 0.2; // Try with 80% of the original size

          // Center the box better
          const centerX =
            detection.boundingBox.left + detection.boundingBox.width / 2;
          const centerY =
            detection.boundingBox.top + detection.boundingBox.height / 2;

          // Calculate new dimensions
          const newWidth =
            detection.boundingBox.width * frame.width * scaleFactor;
          const newHeight =
            detection.boundingBox.height * frame.height * scaleFactor;

          frame.drawRect(
            {
              x: centerX * frame.width - newWidth / 2,
              y: centerY * frame.height - newHeight / 2,
              width: newWidth,
              height: newHeight,
            },
            paint,
          );
        } */
      } catch (error) {
        console.error('Error processing output:', error);
      }
    },
    [actualModel],
  );

  React.useEffect(() => {
    requestPermission();
  }, [requestPermission]);

  console.log(`Model: ${model.state} (${model.model != null})`);

  const format = useCameraFormat(device, [
    {
      videoResolution: {
        width: 416,
        height: 416,
      },
    },
  ]);

  return (
    <View style={styles.container}>
      {hasPermission && device != null ? (
        <>
          <Camera
            device={device}
            style={StyleSheet.absoluteFill}
            isActive={isCameraActive}
            frameProcessor={frameProcessor}
            pixelFormat="rgb"
            videoStabilizationMode="auto"
            format={format}
          />

          {/* Overlay Canvas with direct onDraw prop */}
          {/* <Canvas style={StyleSheet.absoluteFill}>
            {displayDetections?.map((detection, index) => (
              <Rect
                key={index}
                x={detection.boundingBox.left * 416}
                y={detection.boundingBox.top * 416}
                width={detection.boundingBox.width * 416}
                height={detection.boundingBox.height * 416}
                color="red"
                style="stroke"
                strokeWidth={2}
              />
            ))}
          </Canvas> */}
        </>
      ) : (
        <Text>No Camera available.</Text>
      )}

      {model.state === 'loading' && (
        <ActivityIndicator size="small" color="white" />
      )}

      {model.state === 'error' && (
        <Text>Failed to load model! {model.error.message}</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default App;

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
  'worklet';
  return x > 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

function reshapeTo3D(flatArray, dim1 = 13, dim2 = 13, dim3 = 45) {
  'worklet';
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
  'worklet';
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
  'worklet';
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
  probThreshold = 0.6,
  maxDetections = 20,
) {
  'worklet';
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

/**
 * 
 * for (let i = 0; i < resized.data.length; i++) {
  resized.data[i] /= 255;
}
 */
