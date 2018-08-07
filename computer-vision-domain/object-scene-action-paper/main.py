import cv2
import numpy as np


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


def draw_detection(img, rect, thickness=1):
    x, y, w, h = rect
    pad_w, pad_h = int(0.15*w), int(0.05*h)
    cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


def find_track_rois(dict):
    # set up the ROI for tracking
    dict['roi_hists'] = []
    for x, y, w, h in dict['track_windows']:
        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        dict['roi_hists'].append(roi_hist)


if __name__ == '__main__':

    track_range = list(range(5))

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    filepath = "E:\Projects\computer-vision\computer-vision-python\opencv-starter\data/video\AVSS/AVSS_AB_Easy.avi".replace('\\', '/')
    filepath = "E:\Projects\computer-vision\computer-vision-python\opencv-starter\data/video/Sequence_6/visor_1246523090489_new_6_camera1.avi".replace(
        '\\', '/')

    cap = cv2.VideoCapture(filepath)

    while True:
        _, frame = cap.read()
        found, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)
        draw_detections(frame, found)

        cv2.imshow("Recognizer View", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


    # Setup the termination criteria, either 10 iteration or move by at-least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:

        track_dictonary = {}

        _, frame = cap.read()

        # Identify initial bounding boxes of people (x,y,w,h)
        track_dictonary['track_windows'], _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)

        if len(track_dictonary['track_windows']) > 0:

            find_track_rois(track_dictonary)

            # for roi_in in track_dictonary['roi_hists']:
            #
            #     cv2.imshow("Recognizer View", roi_in)
            #
            #     if cv2.waitKey(25) & 0xFF == ord('q'):
            #         cv2.destroyAllWindows()
            #         break

            for _ in track_range:

                ret, frame = cap.read()

                track_count = len(track_dictonary['roi_hists'])
                print('\n\n track count ', track_count)

                if track_count > 0:

                    for count in range(track_count):

                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        dst = cv2.calcBackProject([hsv], [0], track_dictonary['roi_hists'][count], [0, 180], 1)

                        # apply meanshift to get the new location
                        track_window = (track_dictonary['track_windows'][count][0], track_dictonary['track_windows'][count][1], track_dictonary['track_windows'][count][2], track_dictonary['track_windows'][count][3])
                        ret, track_dictonary['track_windows'][count] = cv2.meanShift(dst, track_window, term_crit)

                    draw_detections(frame, track_dictonary['track_windows'])

                    cv2.imshow("Recognizer View", frame)

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break

                else:

                    cv2.imshow("Recognizer View", frame)

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break

        else:

            cv2.imshow("Recognizer View", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break




