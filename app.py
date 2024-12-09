from flask import Flask, request, render_template, jsonify
import os
import json
from markdown import markdown

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def render_notebook(notebook_content):
    """
    Render Jupyter Notebook content into HTML.
    """
    rendered_cells = []
    for cell in notebook_content.get("cells", []):
        cell_type = cell.get("cell_type")
        if cell_type == "markdown":
            # Render Markdown as HTML
            markdown_content = markdown("".join(cell.get("source", [])))
            rendered_cells.append(f'<div class="markdown-cell">{markdown_content}</div>')
        elif cell_type == "code":
            # Render code with syntax highlighting
            code_content = "".join(cell.get("source", []))
            rendered_cells.append(
                f'<div class="code-cell"><pre><code>{code_content}</code></pre></div>'
            )
    return "".join(rendered_cells)


def convert_notebook_to_script(notebook_content):
    """
    Convert Jupyter Notebook content to Python script content.
    """
    script_lines = []
    try:
        for cell in notebook_content.get("cells", []):
            cell_type = cell.get("cell_type")
            if cell_type == "markdown":
                markdown_content = cell.get("source", [])
                script_lines.append('"""\n' + "".join(markdown_content) + '\n"""\n')
            elif cell_type == "code":
                code_content = cell.get("source", [])
                script_lines.append("".join(code_content) + "\n")
    except Exception as e:
        raise ValueError(f"Notebook 解析失败：{e}")

    return "".join(script_lines)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("notebook")
        if file and file.filename.endswith(".ipynb"):
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(input_path)

            try:
                with open(input_path, "r", encoding="utf-8") as nb_file:
                    notebook_content = json.load(nb_file)

                rendered_html = render_notebook(notebook_content)
                python_script = convert_notebook_to_script(notebook_content)

                return jsonify(
                    {
                        "notebook_html": rendered_html,
                        "python": python_script,
                    }
                )
            except (ValueError, json.JSONDecodeError) as e:
                return jsonify({"error": f"文件解析失败：{e}"}), 400

        return jsonify({"error": "请上传一个有效的 Jupyter Notebook 文件 (.ipynb)"}), 400

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
