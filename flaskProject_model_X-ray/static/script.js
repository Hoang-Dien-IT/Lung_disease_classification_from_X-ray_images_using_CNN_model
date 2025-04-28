// scripts.js
document.getElementById("imageInput").addEventListener("change", function(event) {
    let file = event.target.files[0];
    if (file) {
        let formData = new FormData();
        formData.append("file", file);
        fetch("/upload", { method: "POST", body: formData })
            .then(response => response.json())
            .then(data => {
                if (data.image_url) {
                    let previewImage = document.getElementById("previewImage");
                    previewImage.src = data.image_url;
                    previewImage.style.display = "block";
                    previewImage.style.opacity = 1;
                    document.getElementById("predictButton").dataset.imagePath = data.image_url;
                    document.getElementById("predictButton").style.display = "inline-block";
                }
            })
            .catch(error => console.error("Upload error:", error));
    }
});

document.getElementById("predictButton").addEventListener("click", function() {
    let imagePath = this.dataset.imagePath;
    if (!imagePath) return;
    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_path: imagePath })
    })
    .then(response => response.json())
    .then(data => {
        if (data.confidence) {
            let resultContainer = document.getElementById("resultContainer");
            resultContainer.style.display = "block";
            resultContainer.innerHTML = `<h3>Xác suất dự đoán:</h3>`;
            let confidenceHTML = "<div class='confidence'>";
            for (let [label, prob] of Object.entries(data.confidence)) {
                confidenceHTML += `<p>${label}: ${prob}%</p>`;
            }
            confidenceHTML += "</div>";
            resultContainer.innerHTML += confidenceHTML;
        }
    })
    .catch(error => console.error("Predict error:", error));
});