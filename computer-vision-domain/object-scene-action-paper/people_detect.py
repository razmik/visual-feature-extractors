'''
example to show people detection using opencv SVM detector
Keys:
 1 - Pause
 2 - Continue
Keys:
    ESC    - exit
'''

import cv2


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


if __name__ == '__main__':

    print(__doc__)

    track_range = list(range(5))

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    filepath = "E:\Projects\computer-vision\computer-vision-python\opencv-starter\data/video\AVSS/AVSS_AB_Easy.avi".replace('\\', '/')
    filepath = "C:/Users\pc\Downloads\Datasets/sidewalk_people.mp4".replace('\\', '/')
    filepath = "C:/Users\pc\Downloads\Datasets/vtest.avi".replace('\\', '/')

    cap = cv2.VideoCapture(filepath)

    isRecording = True

    while True:

        if isRecording:
            ch = cv2.waitKey(5)
            if ch == ord('1'):  # Pause
                isRecording = not isRecording
        else:
            ch = cv2.waitKey(5)
            if ch == ord('2'):  # Pause
                isRecording = not isRecording
            continue

        _, frame = cap.read()

        found, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)

        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)

        draw_detections(frame, found)
        draw_detections(frame, found_filtered, 3)

        cv2.imshow("Display", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()