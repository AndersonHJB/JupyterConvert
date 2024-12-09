document.addEventListener("DOMContentLoaded", () => {
    const uploadInput = document.getElementById("upload-notebook");
    const notebookRender = document.getElementById("notebook-render");
    const pythonContent = document.getElementById("python-content");
    const copyCodeButton = document.getElementById("copy-code");
    const downloadPythonButton = document.getElementById("download-python");

    uploadInput.addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("notebook", file);

        try {
            const response = await fetch("/", {
                method: "POST",
                body: formData,
            });

            const result = await response.json();

            if (response.ok) {
                // Render Notebook content
                notebookRender.innerHTML = result.notebook.cells
                    .map((cell) => {
                        if (cell.cell_type === "markdown") {
                            return `<div class="markdown">${cell.source.join("")}</div>`;
                        } else if (cell.cell_type === "code") {
                            return `<pre><code>${cell.source.join("")}</code></pre>`;
                        }
                    })
                    .join("");

                // Render Python content with line numbers
                pythonContent.innerHTML = result.python
                    .split("\n")
                    .map((line, i) => `<span>${line}</span>`)
                    .join("\n");

                // Enable download button
                downloadPythonButton.addEventListener("click", () => {
                    const blob = new Blob([result.python], { type: "text/plain" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = file.name.replace(".ipynb", ".py");
                    a.click();
                });
            } else {
                alert(result.error || "转换失败");
            }
        } catch (error) {
            alert("上传失败，请重试！");
        }
    });

    copyCodeButton.addEventListener("click", () => {
        const code = pythonContent.textContent;
        navigator.clipboard.writeText(code).then(() => {
            alert("代码已复制到剪贴板！");
        });
    });
});
