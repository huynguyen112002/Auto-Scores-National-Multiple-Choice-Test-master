from math import sqrt
import cv2
import imutils
import numpy as np

# Đáp án đúng A la 1, B la 2, C la 3, D la 4
ANSWER_KEY = {
    1: 'A', 2: 'A', 3: 'B', 4: 'C', 5: 'D',
    6: 'A', 7: 'B', 8: 'A', 9: 'C', 10: 'A'
}

def preprocess_image(input_path, output_path):
    # Đọc ảnh
    image = cv2.imread(input_path)

    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Xử lý ngưỡng (Thresholding) để làm nổi bật các vùng
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # 3. Tìm các đường viền (Contours) để xác định khung phiếu
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Tìm contour lớn nhất (giả định là khung phiếu trắc nghiệm)
    largest_contour = max(contours, key=cv2.contourArea)

    # 5. Áp dụng hiệu chỉnh phối cảnh (Perspective Transform)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:  # Nếu tìm được đúng 4 góc
        # Sắp xếp các điểm theo thứ tự (trên trái, trên phải, dưới phải, dưới trái)
        points = np.array(sorted(approx[:, 0], key=lambda x: (x[1], x[0])))
        # Phân loại các điểm
        top_left, top_right = points[:2]  # Hai điểm trên cùng (y nhỏ nhất)
        bottom_left, bottom_right = points[2:]  # Hai điểm dưới cùng (y lớn nhất)

        # Đảm bảo sắp xếp theo chiều trái-phải
        if top_left[0] > top_right[0]:
            top_left, top_right = top_right, top_left
        if bottom_left[0] > bottom_right[0]:
            bottom_left, bottom_right = bottom_right, bottom_left

        # Tạo lại mảng `points` với thứ tự chuẩn
        points = np.array([top_left, top_right, bottom_right, bottom_left])
        # Tạo ma trận điểm đích (để ảnh vuông góc)
        
        dst_points = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")

        # Tính toán ma trận biến đổi
        M = cv2.getPerspectiveTransform(points.astype("float32"), dst_points)

        # Biến đổi phối cảnh
        warped = cv2.warpPerspective(image, M, (width, height))

        # 6. Lưu ảnh đã xử lý
        cv2.imwrite(output_path, warped)
        print(f"Ảnh đã được xử lý và lưu tại: {output_path}")
    else:
        print("Không tìm thấy đủ 4 góc để hiệu chỉnh phối cảnh.")

def crop_image(img):  # cat anh thanh cac vung thong tin va vung tra loi cau hoi
    # convert image from BGR to GRAY to apply canny edge detection algorithm chuyen sang anh xam
    image = cv2.imread(img)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # remove noise by blur image loai bo nhieu
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # apply canny edge detection algorithm dung thuat toan canny de phat hien cac canh
    # img_canny = cv2.Canny(blurred, 100, 200)
    # Create binary image
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
    cv2.imshow('thresh', thresh)
    # find contours tim duong vien
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    

    # Define the borders of both inscription number area and answers area
    # format = [x y w h]
    
    INS_NUM_AREA = [width*0.7, height*0.1, width*0.8, height*0.25]
    ANSWERS_AREA = [width*0.1, height*0.35, width*0.9, height*0.7]

    # loop over the contours

    # INS_NUM_Cnts = []
    # ANSWERS_Cnts = []
        
    # previous_ins_cnt = [0, 0]
    # previous_ans_cnt = [0, 0]

    # ins_thresh = 3
    # ans_thresh = 8

    MIN_O = 30
    MAX_O = 50
   
    # for c in cnts:
    #     # Get coordinatees 
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     # print(x, y, w, h)
    #     # Define restrictions for dimensions
    #     if w >= MIN_O and h >= MIN_O and w <= MAX_O and h <= MAX_O:
    #         # Case 1: Inscription number area
    #         if x >= INS_NUM_AREA[0] and x <= INS_NUM_AREA[0]+INS_NUM_AREA[2] and y >= INS_NUM_AREA[1] and y <= INS_NUM_AREA[1]+INS_NUM_AREA[3]:
    #             if sqrt((x-previous_ins_cnt[0])**2 + (y-previous_ins_cnt[1])**2) >= ins_thresh:
    #                 # cv2.drawContours(image, c, -1, (255, 0, 0), 5)
    #                 INS_NUM_Cnts.append(c)
    #                 previous_ins_cnt = [x, y]
    #         # Case 2: Answers area
    #         if x >= ANSWERS_AREA[0] and x <= ANSWERS_AREA[0]+ANSWERS_AREA[2] and y >= ANSWERS_AREA[1] and y <= ANSWERS_AREA[1]+ANSWERS_AREA[3]:
    #             if sqrt((x-previous_ans_cnt[0])**2 + (y-previous_ans_cnt[1])**2) >= ans_thresh:
    #                 cv2.drawContours(image, c, -1, (0, 0, 255), 5)
    #                 ANSWERS_Cnts.append(c)
    #                 previous_ans_cnt = [x, y]

    # Duyệt qua các contours
    question_cnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= MIN_O and h >= MIN_O and w <= MAX_O and h <= MAX_O:
            if x >= ANSWERS_AREA[0] and x <= ANSWERS_AREA[2] and y >= ANSWERS_AREA[1] and y <= ANSWERS_AREA[3]:
                question_cnts.append((c, (x, y, w, h)))

    # Sắp xếp contours theo hàng và cột
    question_cnts = sorted(question_cnts, key=lambda x: (x[1][1], x[1][0]))
    
    # Chấm điểm
    current_question = 1
    correct = 0
    for c, (x, y, w, h) in question_cnts:

        # Đáp án của câu hỏi hiện tại
        correct_answer = ANSWER_KEY.get(current_question, None)

        if correct_answer:
            # Xác định đáp án được chọn
            answer_index = (x - 90) // 100
            selected_answer = chr(64 + answer_index)

            # Tô màu đáp án đúng hoặc sai
            if selected_answer == correct_answer:
                color = (0, 255, 0)
                correct += 1
            else:
                color = (0, 0, 255)

            cv2.drawContours(image, [c], -1, color, 3)

        current_question += 1

    # Hiển thị ảnh kết quả
    score = (correct / 10.0) * 100
    print("score: {:.2f}%".format(score))
    cv2.putText(image, "{:.2f}%".format(score), (600, 900), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 10)
    cv2.namedWindow("Graded", cv2.WINDOW_NORMAL)
    cv2.imshow("Graded", image)
    cv2.waitKey(0)


# Đường dẫn đến ảnh đầu vào và đầu ra
input_image_path = "data/test3.jpg"
output_image_path = "data/preprocessed_image.jpg"
width, height = 2000, 3000

# Gọi hàm xử lý
preprocess_image(input_image_path, output_image_path)
crop_image(output_image_path)
