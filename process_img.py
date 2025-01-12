import imutils
import numpy as np
import cv2
import os
from math import ceil, sqrt
from model import CNN_Model
from collections import defaultdict



# lay thong tin toa do duong vien contours
def get_x(s):
    return s[1][0]


def get_y(s):
    return s[1][1]


def get_h(s):
    return s[1][3]


def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]


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
        width, height = 1000, 2000  # Kích thước chuẩn
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
    


def crop_image(img):  # cắt ảnh thành các khối câu hỏi
    # Đọc ảnh và chuyển sang ảnh xám
    image = cv2.imread(img)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Loại bỏ nhiễu bằng làm mờ Gaussian
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Áp dụng thuật toán Canny để phát hiện cạnh
    img_canny = cv2.Canny(blurred, 100, 200)

    # Tìm các đường viền
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0

    # Đảm bảo có ít nhất một đường viền được tìm thấy
    if len(cnts) > 0:
        # Sắp xếp các đường viền theo diện tích giảm dần
        cnts = sorted(cnts, key=get_x_ver1)

        for i, c in enumerate(cnts):
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)

            if w_curr * h_curr > 100000:  # lọc ra các đường viền lớn hơn 100000 px
                # Kiểm tra chồng lấp đường viền
                check_xy_min = x_curr * y_curr - x_old * y_old
                check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - (x_old + w_old) * (y_old + h_old)

                # Thêm khối câu hỏi vào danh sách nếu không trùng lặp
                if len(ans_blocks) == 0:
                    cropped_block = gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr]
                    ans_blocks.append((cropped_block, [x_curr, y_curr, w_curr, h_curr]))

                    # Hiển thị hoặc lưu khối câu hỏi
                    cv2.imshow(f"Answer Block {len(ans_blocks)}", cropped_block)


                    # Cập nhật tọa độ
                    x_old, y_old, w_old, h_old = x_curr, y_curr, w_curr, h_curr
                elif check_xy_min > 20000 and check_xy_max > 20000:
                    cropped_block = gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr]
                    ans_blocks.append((cropped_block, [x_curr, y_curr, w_curr, h_curr]))

                    # Hiển thị hoặc lưu khối câu hỏi
                    # cv2.imshow(f"Answer Block {len(ans_blocks)}", cropped_block)


                    # Cập nhật tọa độ
                    x_old, y_old, w_old, h_old = x_curr, y_curr, w_curr, h_curr

    # Sắp xếp ans_blocks theo tọa độ x từ trái qua phải
    sorted_ans_blocks = sorted(ans_blocks, key=get_x)

    # Thoát các cửa sổ hiển thị
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return sorted_ans_blocks



def process_ans_blocks(ans_blocks):  # Xử lý khối câu hỏi đã cắt
    """
    Hàm này xử lý 2 khối chứa các đáp án và trả về danh sách các ô đáp án đã chia nhỏ.
    :param ans_blocks: danh sách 2 khối, mỗi khối có định dạng [ảnh, [x, y, w, h]].
    """
    list_answers = []

    # Duyệt qua từng khối chứa câu hỏi
    for block_idx, ans_block in enumerate(ans_blocks):
        ans_block_img = np.array(ans_block[0])

        offset1 = ceil(ans_block_img.shape[0] / 6)  # Chia khối thành 6 hàng
        for i in range(6):
            box_img = np.array(ans_block_img[i * offset1:(i + 1) * offset1, :])
            height_box = box_img.shape[0]

            # Cắt bỏ lề trên và dưới để làm gọn hàng
            box_img = box_img[14:height_box - 14, :]
            offset2 = ceil(box_img.shape[0] / 5)  # Chia hàng thành 5 dòng
            # cv2.imshow(f"Answer Block {block_idx + 1} - Line {i + 1}", box_img)
        
            # Duyệt qua từng dòng trong hàng
            for j in range(5):
                answer_img = box_img[j * offset2:(j + 1) * offset2, :]
                list_answers.append(answer_img)

                # Hiển thị ảnh
                # cv2.imshow(f"Answer Block {block_idx + 1} - Line {i + 1}, Bubble {j + 1}", answer_img)


    # Đợi nhấn phím để tiếp tục và đóng tất cả các cửa sổ hiển thị
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return list_answers



def process_list_ans(list_answers):  # Xử lý ảnh ô tròn
    list_choices = []
    offset = 44  # Khoảng cách giữa các ô tròn
    start = 32   # Vị trí bắt đầu của ô tròn đầu tiên
    count = 0

    for idx, answer_img in enumerate(list_answers):
        for i in range(4):  # Mỗi hàng có 4 ô tròn
            # Cắt ô tròn
            bubble_choice = answer_img[:, start + i * offset:start + (i + 1) * offset]

            # Làm nổi bật vùng tô bằng threshold
            bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # Resize về kích thước chuẩn 28x28 px
            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)

            # Thêm vào danh sách kết quả
            bubble_choice = bubble_choice.reshape((28, 28, 1))  # Chuyển đổi thành định dạng phù hợp
            list_choices.append(bubble_choice)

            # Hiển thị tối đa 10 ảnh đầu tiên
        #     if count < 10:
        #         cv2.imshow(f"Bubble {count + 1}", bubble_choice)
        #         count += 1
        #     if count == 10:
        #         break  # Dừng sau khi hiển thị 10 ảnh
        # if count == 10:
        #     break  # Dừng sau khi hiển thị 10 ảnh

    # Đợi nhấn phím để tiếp tục và đóng tất cả các cửa sổ hiển thị
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return list_choices



def map_answer(idx):  # anh xa vi tri o tron thanh a, b, c, d
    if idx % 4 == 0:
        answer_circle = "A"
    elif idx % 4 == 1:
        answer_circle = "B"
    elif idx % 4 == 2:
        answer_circle = "C"
    else:
        answer_circle = "D"
    return answer_circle


def get_answers(list_answers, output_file):  # Xác định ô tròn được tô đen
    results = defaultdict(list)
    model = CNN_Model('weight.h5').build_model(rt=True)  # Nạp mô hình CNN
    list_answers = np.array(list_answers)
    scores = model.predict_on_batch(list_answers / 255.0)
    displayed_images = 0

    for idx, score in enumerate(scores):
        question = idx // 4  # Xác định số câu hỏi
        
        # Hiển thị chỉ 10 ảnh đầu tiên
        # if displayed_images < 10:
        #     answer_image = list_answers[idx].reshape(28, 28)  # Chuyển đổi lại về ảnh 28x28
        #     cv2.imshow(f"Q{question + 1} Choice {map_answer(idx)}", answer_image)
        #     displayed_images += 1
        
        # Xác định lựa chọn với confidence > 0.9
        if score[1] > 0.9:  # Nếu xác suất chọn > 0.9
            chosed_answer = map_answer(idx)
            results[question + 1].append(chosed_answer)

    # Ghi kết quả ra file text
    with open(output_file, 'w', encoding='utf-8') as f:
        for question, answers in results.items():
            f.write(f"{question}. {', '.join(answers)}\n")

    print(f"Ảnh đã chuyển sang văn bản và lưu tại: {output_file}")

    # Đợi nhấn phím để tiếp tục và đóng tất cả các cửa sổ hiển thị
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results


if __name__ == '__main__':
    in_path = 'data/test2.jpg'
    out_path = 'data/output.jpg'
    isExist = os.path.exists(in_path) 
    if isExist:
        pre_img = preprocess_image(in_path, out_path)
        list_ans_boxes = crop_image(out_path)
        list_ans1 = process_ans_blocks(list_ans_boxes)
        list_ans2 = process_list_ans(list_ans1)
        get_answers(list_ans2, 'student_answers.txt')
        
    else:
        print("Duong dan khong ton tai")

