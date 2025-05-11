/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 */

import React from 'react';

import {StyleSheet, View, Text, ActivityIndicator} from 'react-native';
import {
  Tensor,
  TensorflowModel,
  useTensorflowModel,
} from 'react-native-fast-tflite';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
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

  const handleCameraActive = useRunOnJS(() => {
    setIsCameraActive(false);
  }, []);

  React.useEffect(() => {
    if (actualModel == null) {
      return;
    }
    console.log(`Model loaded! Shape:\n${modelToString(actualModel)}]`);
  }, [actualModel]);

  const {resize} = useResizePlugin();

  const frameProcessor = useFrameProcessor(
    frame => {
      'worklet';
      if (actualModel == null) {
        // model is still loading...
        return;
      }

      console.log(`Running inference on ${frame}`);
      const resized = resize(frame, {
        scale: {
          width: 416,
          height: 416,
        },
        pixelFormat: 'rgb',
        dataType: 'float32',
      });
      const result = actualModel.runSync([resized]);

      // Debug the model output structure
      console.log('Model output structure:', {
        length: result.length,
        types: result.map((tensor, index) => ({
          index,
          type: tensor?.constructor?.name,
          length: tensor?.length,
        })),
      });

      // Safely access the scores with error handling
      let scores: number[] = [];
      try {
        if (result && result.length > 0) {
          // Try to find the scores tensor - it might be in a different position
          const scoresTensor = result.find(
            tensor => tensor && tensor.length > 0,
          );
          if (scoresTensor) {
            scores = Array.from(scoresTensor as Float32Array);
          }
        }
      } catch (error) {
        console.error('Error processing model output:', error);
      }

      const validDetections = scores.filter(score => score > 0.5).length;
      console.log('Number of valid detections: ' + validDetections);

      // Show scores in a more readable format
      const detectionDetails = scores
        .map((score, index) => ({
          detection: index + 1,
          confidence: `${(score * 100).toFixed(1)}%`,
          isValid: score > 0.5,
        }))
        .filter(det => det.isValid);

      console.log('Valid detections with confidence scores:');
      console.log(JSON.stringify(detectionDetails, null, 2));
      if (validDetections >= 10) {
        handleCameraActive();
      }
    },
    [actualModel],
  );

  React.useEffect(() => {
    requestPermission();
  }, [requestPermission]);

  console.log(`Model: ${model.state} (${model.model != null})`);

  return (
    <View style={styles.container}>
      {hasPermission && device != null ? (
        <Camera
          device={device}
          style={StyleSheet.absoluteFill}
          isActive={isCameraActive}
          frameProcessor={frameProcessor}
          pixelFormat="yuv"
        />
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
