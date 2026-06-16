from pathlib import Path

# File output
output_file = "content.txt"

# Thư mục hiện tại (nơi chạy script)
root_dir = Path.cwd()

with open(output_file, "w", encoding="utf-8") as out:
    # Tìm tất cả file .py ở mọi cấp thư mục
    for py_file in root_dir.rglob("*.py"):
        # Bỏ qua chính script hiện tại nếu muốn
        # if py_file.name == Path(__file__).name:
        #     continue

        try:
            relative_path = py_file.relative_to(root_dir)

            # Đọc nội dung file
            content = py_file.read_text(encoding="utf-8")

            # Ghi theo format yêu cầu
            out.write(f"======= {relative_path} =======\n\n")
            out.write(content)
            out.write("\n\n")

        except Exception as e:
            out.write(f"======= {py_file} =======\n\n")
            out.write(f"[ERROR READING FILE: {e}]\n\n")

print(f"Đã ghi toàn bộ nội dung file .py vào {output_file}")