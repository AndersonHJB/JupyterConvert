document.addEventListener("DOMContentLoaded", () => {
    const uploadInput = document.getElementById("upload-notebook");
    const notebookRender = document.getElementById("notebook-render");
    const pythonContent = document.getElementById("python-content");
    const copyCodeButton = document.getElementById("copy-code");
    const copySuccess = document.getElementById("copy-success");
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
                notebookRender.innerHTML = result.notebook_html;
                pythonContent.textContent = result.python;

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
            copySuccess.classList.remove("hidden");
            setTimeout(() => {
                copySuccess.classList.add("hidden");
            }, 1500);
        });
    });
});
