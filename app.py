from flask import Flask, request, render_template, send_file
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
CONVERTED_FOLDER = "converted"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)


def convert_notebook_to_script(notebook_path, output_path):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as nb_file:
            notebook = json.load(nb_file)  # 尝试加载 JSON
    except json.JSONDecodeError as e:
        raise ValueError(f"无法解析 Jupyter Notebook 文件。错误: {e}")

    script_lines = []

    for cell in notebook.get('cells', []):
        cell_type = cell.get('cell_type')
        if cell_type == 'markdown':
            markdown_content = cell.get('source', [])
            script_lines.append('"""\n' + ''.join(markdown_content) + '\n"""\n')
        elif cell_type == 'code':
            code_content = cell.get('source', [])
            script_lines.append(''.join(code_content) + '\n')

    with open(output_path, 'w', encoding='utf-8') as script_file:
        script_file.writelines(script_lines)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("notebook")
        if file and file.filename.endswith(".ipynb"):
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            output_filename = file.filename.rsplit(".", 1)[0] + ".py"
            output_path = os.path.join(CONVERTED_FOLDER, output_filename)

            file.save(input_path)
            try:
                convert_notebook_to_script(input_path, output_path)
            except ValueError as e:
                return f"文件转换失败：{e}", 400  # 返回错误信息到页面

            return send_file(output_path, as_attachment=True)

        return "请上传一个有效的 Jupyter Notebook 文件 (.ipynb)", 400

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
