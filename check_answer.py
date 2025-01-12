import re


def read_answers(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Đọc file và chỉ trích xuất ký tự cuối cùng (đáp án) sau số thứ tự
        answers = [re.sub(r"^\d+\.\s*", "", line.strip()) for line in file if line.strip()]
    return answers


def grade_exam(student_file, answer_file, result_file):
    # Đọc đáp án và câu trả lời từ các file
    correct_answers = read_answers(answer_file)
    student_answers = read_answers(student_file)

    # Đảm bảo độ dài câu trả lời của học sinh và đáp án khớp nhau
    if len(correct_answers) != len(student_answers):
        raise ValueError("Số lượng câu trả lời của học sinh không khớp với đáp án")

    # Chấm điểm từng câu và đếm số câu đúng
    results = []
    correct_count = 0

    for i, (student, correct) in enumerate(zip(student_answers, correct_answers), start=1):
        if student == correct:
            results.append(f"Câu {i}: Đúng")
            correct_count += 1
        else:
            results.append(f"Câu {i}: Sai (Đáp án đúng: {correct})")

    # In kết quả chi tiết và tổng kết
    total_questions = len(correct_answers)
    results.append(f"Tổng số câu đúng: {correct_count}/{total_questions}")

    # Ghi kết quả vào file
    with open(result_file, 'w', encoding='utf-8') as file:
        for line in results:
            file.write(line + "\n")
    print(f"Kết quả đã được lưu vào {result_file}")
    return results


grade_exam("student_answers.txt", "right_answers.txt", "result.txt")
