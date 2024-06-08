import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import numpy as np
import cv2
import threading
import time
import global_vars
import struct


class CaptureThread(threading.Thread):
    cap = None
    ret = None
    frame = None
    isRunning = False
    counter = 0
    timer = 0.0
    latency = 0.0

    def run(self):
        self.cap = cv2.VideoCapture(global_vars.WEBCAM_INDEX)
        if global_vars.USE_CUSTOM_CAM_SETTINGS:
            self.cap.set(cv2.CAP_PROP_FPS, global_vars.FPS)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, global_vars.WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, global_vars.HEIGHT)
        time.sleep(1)

        print("Opened Capture @ %s fps" % str(self.cap.get(cv2.CAP_PROP_FPS)))
        while not global_vars.KILL_THREADS:
            start_time = time.perf_counter()
            self.ret, self.frame = self.cap.read()
            self.latency = (time.perf_counter() - start_time) * 1000
            self.isRunning = True
            if global_vars.DEBUG:
                self.counter = self.counter + 1
                if time.time() - self.timer >= 3:
                    print("Capture FPS: ", self.counter / (time.time() - self.timer))
                    self.counter = 0
                    self.timer = time.time()


class BodyThread(threading.Thread):
    data = ""
    dirty = True
    pipe = None
    timeSinceCheckedConnection = 0
    timeSincePostStatistics = 0

    processing_latency = 0.0
    communication_latency = 0.0

    def compute_real_world_landmarks(self, world_landmarks, image_landmarks, image_shape):
        try:
            frame_height, frame_width, channels = image_shape
            focal_length = frame_width * .6
            center = (frame_width / 2, frame_height / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            distortion = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(objectPoints=world_landmarks,
                                                                        imagePoints=image_landmarks,
                                                                        cameraMatrix=camera_matrix,
                                                                        distCoeffs=distortion,
                                                                        flags=cv2.SOLVEPNP_SQPNP)
            transformation = np.eye(4)
            transformation[0:3, 3] = translation_vector.squeeze()

            model_points_hom = np.concatenate((world_landmarks, np.ones((33, 1))), axis=1)
            world_points = model_points_hom.dot(np.linalg.inv(transformation).T)

            return world_points
        except AttributeError:
            print("Attribute Error")
            return world_landmarks

    def run(self):
        mp_drawing = mediapipe.solutions.drawing_utils
        mp_pose = mediapipe.solutions.pose

        capture = CaptureThread()
        capture.start()

        with mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.5, model_complexity
        = global_vars.MODEL_COMPLEXITY, static_image_mode=False, enable_segmentation=True) as pose:

            while not global_vars.KILL_THREADS and capture.isRunning == False:
                print("Waiting for camera and capture thread.")
                time.sleep(0.5)
            print("Beginning capture")

            while not global_vars.KILL_THREADS and capture.cap.isOpened():
                start_time = time.perf_counter()
                ret = capture.ret
                image = capture.frame
                image.flags.writeable = global_vars.DEBUG
                results_pose = pose.process(image.copy())
                self.processing_latency = (time.perf_counter() - start_time) * 1000

                if global_vars.DEBUG:
                    if time.time() - self.timeSincePostStatistics >= 0.1:  #(100 ms)
                        print("Instaneous FPS: %f" % (1 / (self.processing_latency / 1000)))
                        print(f"Capture Latency: {capture.latency:.2f} ms")
                        print(f"Processing Latency: {self.processing_latency:.2f} ms")
                        print(f"Communication Latency: {self.communication_latency:.2f} ms")

                        visible_landmark_count = 0
                        average_confidence = 0

                        if results_pose.pose_world_landmarks:
                            visible_landmarks = [
                                landmark for landmark in results_pose.pose_world_landmarks.landmark if landmark.visibility > 0.5
                            ]
                            visible_landmark_count = len(visible_landmarks)
                            average_confidence = (
                                sum(landmark.visibility for landmark in visible_landmarks) / visible_landmark_count
                                if visible_landmark_count > 0 else 0
                            )

                        # Simpan data ke file CSV performance_data.csv
                        with open('performance_data.csv', 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([
                                time.time(),  # Timestamp
                                capture.latency,
                                self.processing_latency,
                                self.communication_latency,
                                1 / (self.processing_latency / 1000),
                                visible_landmark_count,
                                average_confidence
                            ])

                        self.timeSincePostStatistics = time.time()

                    if results_pose.pose_landmarks:
                        mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2,
                                                                         circle_radius=4),
                                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                         circle_radius=2),
                                                  )

                    cv2.imshow('Body Tracking', image)
                    cv2.waitKey(1)

                if self.pipe == None and time.time() - self.timeSinceCheckedConnection >= 1:
                    try:
                        self.pipe = open(r'\\.\pipe\UnityMediaPipeBody', 'r+b', 0)
                    except FileNotFoundError:
                        print("Waiting for Unity project.")
                        self.pipe = None
                    self.timeSinceCheckedConnection = time.time()

                if self.pipe != None:
                    self.data = ""
                    i = 0

                    if results_pose.pose_world_landmarks:
                        image_landmarks = results_pose.pose_landmarks
                        world_landmarks = results_pose.pose_world_landmarks

                        model_points = np.float32([[-l.x, -l.y, -l.z] for l in world_landmarks.landmark])
                        image_points = np.float32(
                            [[l.x * image.shape[1], l.y * image.shape[0]] for l in image_landmarks.landmark])

                        body_world_landmarks_world = self.compute_real_world_landmarks(model_points, image_points,
                                                                                       image.shape)
                        body_world_landmarks = results_pose.pose_world_landmarks

                        for i in range(0, 33):
                            self.data += "FREE|{}|{}|{}|{}\n".format(i, body_world_landmarks_world[i][0],
                                                                     body_world_landmarks_world[i][1],
                                                                     body_world_landmarks_world[i][2])
                        for i in range(0, 33):
                            self.data += "ANCHORED|{}|{}|{}|{}\n".format(i, -body_world_landmarks.landmark[i].x,
                                                                         -body_world_landmarks.landmark[i].y,
                                                                         -body_world_landmarks.landmark[i].z)
                            # Kirim timestamp
                            self.data += f"TIMESTAMP|{time.time()}\n"

                            # Hitung dan kirim jumlah landmark yang terlihat dan confidence score rata-rata
                            visible_landmarks = [
                                landmark for landmark in results_pose.pose_world_landmarks.landmark if landmark.visibility >= 1
                            ]
                            visible_landmark_count = len(visible_landmarks)
                            average_confidence = (
                                sum(landmark.visibility for landmark in visible_landmarks) / visible_landmark_count
                                if visible_landmark_count > 0 else 0
                            )

                            self.data += f"VISIBLE_LANDMARK_COUNT|{visible_landmark_count}\n"
                            self.data += f"AVERAGE_CONFIDENCE|{average_confidence}\n"

                    start_com_time = time.perf_counter()
                    s = self.data.encode('utf-8')
                    try:
                        self.pipe.write(struct.pack('I', len(s)) + s)
                        self.pipe.seek(0)
                    except Exception as ex:
                        print("Failed to write.")
                        self.pipe = None
                    self.communication_latency = (time.perf_counter() - start_com_time) * 1000

        self.pipe.close()
        capture.cap.release()
        cv2.destroyAllWindows()
