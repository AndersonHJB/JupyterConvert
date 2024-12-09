document.addEventListener("DOMContentLoaded", () => {
    const uploadInput = document.getElementById("upload-notebook");
    const notebookContent = document.getElementById("notebook-content");
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
                notebookContent.value = result.notebook;
                pythonContent.textContent = result.python;

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
        navigator.clipboard.writeText(pythonContent.textContent).then(() => {
            alert("代码已复制到剪贴板！");
        });
    });
});
