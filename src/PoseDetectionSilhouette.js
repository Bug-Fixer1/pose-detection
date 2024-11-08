import React, { useState, useEffect } from 'react';
import { StyleSheet, View, Image } from 'react-native';
import { Camera } from 'expo-camera';
import * as PoseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';

const PoseDetectionSilhouette = () => {
  const [hasPermission, setHasPermission] = useState(null);
  const [detector, setDetector] = useState(null);
  const [poses, setPoses] = useState([]);

  // Silhouette dimensions
  const silhouetteWidth = 300;
  const silhouetteHeight = 400;

  useEffect(() => {
    (async () => {
      // Request camera permission
      const { status } = await Camera.requestPermissionsAsync();
      setHasPermission(status === 'granted');

      // Initialize TensorFlow.js and pose detection model
      await tf.setBackend('cpu');
      const detector = await PoseDetection.createDetector(PoseDetection.SupportedModels.MoveNet);
      setDetector(detector);
    })();
  }, []);

  useEffect(() => {
    let interval;
    if (hasPermission && detector) {
      interval = setInterval(async () => {
        try {
          // Capture camera frame
          const { status, type, ...cameraRef } = await Camera.useCameraStream();
          if (status !== 'active') return;

          // Detect poses
          const estimatedPoses = await detector.estimatePoses(cameraRef.current);
          setPoses(estimatedPoses);
        } catch (error) {
          console.error('Error detecting poses:', error);
        }
      }, 100);
    }

    return () => clearInterval(interval);
  }, [hasPermission, detector]);

  return (
    <View style={styles.container}>
      {hasPermission !== null && (
        <Camera style={styles.camera}>
          <View style={styles.silhouetteContainer}>
            <Image
              source={require('./assets/silhouette.png')}
              style={[
                styles.silhouette,
                {
                  width: silhouetteWidth,
                  height: silhouetteHeight,
                },
              ]}
            />
            {poses.length > 0 && (
              <View style={styles.poseContainer}>
                {poses[0].keypoints
                  .filter((keypoint) => keypoint.score > 0.5 && keypoint.name === 'nose')
                  .map((keypoint, index) => (
                    <View
                      key={index}
                      style={[
                        styles.poseIndicator,
                        {
                          top: keypoint.y - 20,
                          left: keypoint.x - 20,
                          width: 40,
                          height: 40,
                        },
                      ]}
                    />
                  ))}
              </View>
            )}
          </View>
        </Camera>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  camera: {
    flex: 1,
    width: '100%',
    height: '100%',
  },
  silhouetteContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  silhouette: {
    resizeMode: 'contain',
  },
  poseContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  poseIndicator: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: 'green',
    borderRadius: 20,
  },
});

export default PoseDetectionSilhouette;