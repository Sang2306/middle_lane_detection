import cv2 as cv
import numpy as np
import sys

def edge_detect(image, threshold = 50):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(gray_image, (5, 5), 0)
    threshold1 = threshold                     
    threshold2 = 3*threshold     
    canny_image = cv.Canny(blurred, threshold1, threshold2)
    return canny_image

def ROI(image):
    x1, y1 = 0, 197
    x2, y2 = 64, 96
    x3, y3 = 198, 30
    x4, y4 = 342, 30
    x5, y5 = 460, 120
    x6, y6 = width, 197
    polygons = np.array([
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)]
    ])
    mask = np.zeros_like(image)
    WHITE = 255
    cv.fillPoly(mask, polygons, WHITE)
    roi = cv.bitwise_and(image, mask)
    return roi

def calculate_line(lines):
    left, right = [], []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        #vi chi di qua hai diem nen chon phuong trinh bac nhat
        X = np.array([[1, x1], [1, x2]])
        y = np.array([y1, y2])
        X_inv = np.linalg.inv(X)
        y_T = y.T
        polynomial_fitted = np.dot(X_inv, y_T)
        a0 = polynomial_fitted[0]
        a1 = polynomial_fitted[1]
        if a1 == 0:
            continue
        if a1 < 0:
            left.append((a1, a0))
        else:
            right.append((a1, a0))           
    if len(left) > 0 and len(right) > 0:
        left_avg = np.average(left, axis = 0)
        right_avg = np.average(right, axis = 0)
        left_line = calculate_points(left_avg)
        right_line = calculate_points(right_avg)
        global center_top, center_bottom
        center_top = (int((right_line[2] - left_line[2])/2) + left_line[2], 30)
        center_bottom = (int((right_line[0] - left_line[0])/2) + left_line[0], 197)
        return np.array([left_line, right_line])
    else:
        pass

def calculate_points(poly_coef):
    a1, a0 = poly_coef
    y1 = 197 #diem y duoi cung
    y2 = 30 #diem y tren cung cua ROI
    #y1 = a1x1 + a0 --> x1
    x1 = int((y1 - a0) / a1)
    #y2 = a1x2 + a0 --> x2
    x2 = int((y2 - a0) / a1)
    return np.array([x1, y1, x2, y2])
    
def draw(image, lines):
    mask = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(mask, (x1, y1), (x2, y2), (255, 0, 0), 5)
            cv.line(mask, center_bottom, center_top, (0, 255, 255), 2)
    return mask

if __name__ == "__main__":
    video_name = sys.argv[1] #python lane.py 2.avi
    camera = cv.VideoCapture(video_name)
    if not camera.isOpened():
        print('Video khong ton tai')
    height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
    width =  int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
    fps = camera.get(cv.CAP_PROP_FRAME_COUNT)
    with open(video_name + '.output', 'w') as file:
        file.write(str(int(fps)) + '\n')
    while camera.isOpened():
        ret, frame = camera.read()
        current_frame = camera.get(cv.CAP_PROP_POS_FRAMES)
        if ret is False:
            break
        edge_detected = edge_detect(frame)
        roi = ROI(edge_detected)
        hough_lines = cv.HoughLinesP(
            image = roi, 
            rho = 2, 
            theta = np.pi/180, 
            threshold = 100, 
            lines = np.array([]), 
            minLineLength = 10, 
            maxLineGap = 40
        )
        try:
            lines = calculate_line(hough_lines)
            mask = draw(frame, lines)
            output = cv.addWeighted(frame, 0.9, mask, 1, 1)
            # cv.circle(output, center_top, 2, (0, 255, 255), 4)
            # cv.circle(output, center_bottom, 2, (0, 255, 255), 4)
            center_average_x = int((center_top[0] - center_bottom[0])/2) + center_bottom[0]
            center_average_y = int((center_top[1] - center_bottom[1])/2) + center_bottom[1]
            with open(video_name + '.output', 'a') as file:
                file.writelines(str(int(current_frame)) + ' ' + str(int(center_average_x)) + ' ' + str(int(center_average_y)) + '\n')           
            cv.imshow('output', output)
            cv.imshow('ROI', roi)
        except Exception:
            pass
        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    camera.release()
    cv.destroyAllWindows()
