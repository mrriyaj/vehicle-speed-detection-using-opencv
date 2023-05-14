import cv2
import dlib
import math
import time
from flask import Flask, render_template, Response
import configparser

app = Flask(__name__)

config = configparser.ConfigParser()
config.read('config.ini')

carCascade = cv2.CascadeClassifier('myhaar.xml')
video = None
WIDTH = 1280
HEIGHT = 720
line_pos1 = 400
line_pos2 = 180

def estimateSpeed(location1, location2, fps, ppm):
    x1, y1 = location1
    x2, y2 = location2

    # Calculate the distance between the two points
    distance_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    distance_meters = distance_pixels / ppm

    # Calculate the time taken in seconds
    time_seconds = 1 / fps

    # Calculate the speed in meters per second
    speed_mps = distance_meters / time_seconds

    # Convert speed to kilometers per hour
    speed_kph = speed_mps * 3.6

    return speed_kph

def draw_lines(image):
    # Draw the two lines on the image
    cv2.line(image, (0, line_pos1), (WIDTH, line_pos1), (0, 0, 255), 2)
    cv2.line(image, (0, line_pos2), (WIDTH, line_pos2), (0, 0, 255), 2)

def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    lineColor = (0, 0, 255)  # Red color for lines
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = {}

    while True:
        start_time = time.time()
        ret, image = video.read()
        if not ret:
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        frameCounter += 1

        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)
            speed.pop(carID, None)

        if frameCounter % 10 == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (x, y, w, h) in cars:
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if t_x <= x_bar <= t_x + t_w and t_y <= y_bar <= t_y + t_h:
                        matchCarID = carID

                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = (x, y, w, h)
                    currentCarID += 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()
            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())
            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

            if carID in carLocation1:
                (x1, y1, w1, h1) = carLocation1[carID]
                (x2, y2, w2, h2) = (t_x, t_y, t_w, t_h)

                # Calculate speed if car has crossed a certain y-coordinate threshold
                if y1 >= line_pos1 and y2 >= line_pos1:
                    speed[carID] = estimateSpeed((x1, y1), (x2, y2), fps, 8.8)

                # Display the speed, car ID, and crossed line on the image
                if carID in speed:
                    if y1 >= line_pos2 and y2 >= line_pos2:
                        info_text = "Car ID: {} | Speed: {:.2f} km/hr".format(carID, speed[carID])
                        cv2.putText(resultImage, info_text, (int(x1 + w1/2), int(y1 + h1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                # Check if the car crosses Line 1
                if y1 + h1 >= line_pos1 and y1 <= line_pos1:
                    cv2.putText(resultImage, "Crossed Line 1", (int(x1 + w1/2), int(y1 + h1 + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                # Check if the car crosses Line 2
                if y1 + h1 >= line_pos2 and y1 <= line_pos2:
                    cv2.putText(resultImage, "Crossed Line 2", (int(x1 + w1/2), int(y1 + h1 + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                # Update the previous location
                carLocation1[carID] = (x2, y2, w2, h2)

        end_time = time.time()

        if not (end_time == start_time):
            fps = 1.0 / (end_time - start_time)

        cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        draw_lines(resultImage)

        _, jpeg = cv2.imencode('.jpg', resultImage)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global video
    video = cv2.VideoCapture('cars.mp4')
    return Response(trackMultipleObjects(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
