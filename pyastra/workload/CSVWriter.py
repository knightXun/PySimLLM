import os
import csv

class CSVWriter:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.full_path = os.path.join(path, name)

    def write_line(self, data):
        try:
            with open(self.full_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([data])
        except Exception as e:
            print(f"写入行时出错: {e}")

    def write_res(self, data):
        try:
            with open(self.full_path, 'r') as file:
                current_content = file.readlines()
            with open(self.full_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([data])
                file.writelines(current_content)
        except Exception as e:
            print(f"写入结果时出错: {e}")

    def initialize_csv(self, rows, cols):
        try:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            with open(self.full_path, 'w', newline='') as file:
                pass
            print(f"CSV 路径和文件名: {self.full_path}")
        except Exception as e:
            print(f"初始化 CSV 时出错: {e}")
            exit(1)

    def finalize_csv(self, dims):
        try:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            print(f"创建 CSV 的路径是: {self.path}")
            with open(self.full_path, 'w', newline='') as file:
                writer = csv.writer(file)
                headers = [" time (us) "]
                for i in range(1, len(dims) + 1):
                    headers.append(f"dim{i} util")
                writer.writerow(headers)
                dims_it = [iter(dim) for dim in dims]
                while True:
                    row = []
                    finished = 0
                    compare = None
                    for i, it in enumerate(dims_it):
                        try:
                            item = next(it)
                            if i == 0:
                                row.append(item[0] / FREQ)
                                compare = item[0]
                            else:
                                assert compare == item[0]
                            row.append(item[1])
                        except StopIteration:
                            finished += 1
                            row.append(None)
                    if finished == len(dims_it):
                        break
                    writer.writerow(row)
        except Exception as e:
            print(f"完成 CSV 时出错: {e}")
            exit(1)

    def write_cell(self, row, column, data):
        try:
            with open(self.full_path, 'r', newline='') as file:
                reader = csv.reader(file)
                lines = list(reader)
            if row >= len(lines):
                for _ in range(row - len(lines) + 1):
                    lines.append([])
            if column >= len(lines[row]):
                for _ in range(column - len(lines[row]) + 1):
                    lines[row].append(None)
            lines[row][column] = data
            with open(self.full_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(lines)
        except Exception as e:
            print(f"写入单元格时出错: {e}")


# 这里需要定义 FREQ，在原 C++ 代码中它应该是一个全局常量
FREQ = 1.0