from flask import Flask, request, render_template, jsonify
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
CONVERTED_FOLDER = "converted"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)


def convert_notebook_to_script(notebook_content):
    """
    Convert Jupyter Notebook content to Python script content.
    """
    script_lines = []
    try:
        for cell in notebook_content.get('cells', []):
            cell_type = cell.get('cell_type')
            if cell_type == 'markdown':
                markdown_content = cell.get('source', [])
                script_lines.append('"""\n' + ''.join(markdown_content) + '\n"""\n')
            elif cell_type == 'code':
                code_content = cell.get('source', [])
                script_lines.append(''.join(code_content) + '\n')
    except Exception as e:
        raise ValueError(f"Notebook 解析失败：{e}")

    return ''.join(script_lines)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("notebook")
        if file and file.filename.endswith(".ipynb"):
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(input_path)

            try:
                with open(input_path, 'r', encoding='utf-8') as nb_file:
                    notebook_content = json.load(nb_file)

                python_script = convert_notebook_to_script(notebook_content)
                return jsonify({
                    "notebook": json.dumps(notebook_content, indent=4, ensure_ascii=False),
                    "python": python_script,
                })
            except (ValueError, json.JSONDecodeError) as e:
                return jsonify({"error": f"文件解析失败：{e}"}), 400

        return jsonify({"error": "请上传一个有效的 Jupyter Notebook 文件 (.ipynb)"}), 400

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
