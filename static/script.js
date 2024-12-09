document.addEventListener("DOMContentLoaded", () => {
    const copyButton = document.getElementById("copy-button");
    const codeBlock = document.getElementById("code-block");

    if (copyButton) {
        copyButton.addEventListener("click", () => {
            navigator.clipboard.writeText(codeBlock.textContent).then(() => {
                alert("代码已复制到剪贴板！");
            });
        });
    }
});
