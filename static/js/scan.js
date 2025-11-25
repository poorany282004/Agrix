// CAMERA
const cameraBtn = document.getElementById("cameraBtn");
const cameraBox = document.getElementById("cameraBox");
const cameraPreview = document.getElementById("cameraPreview");
const captureBtn = document.getElementById("captureBtn");

// FILE UPLOAD
const fileInput = document.getElementById("fileInput");
const previewArea = document.getElementById("previewArea");

// SUBMIT
const submitBtn = document.getElementById("submitBtn");

let capturedImageFile = null;

// Start Camera
cameraBtn.onclick = async () => {
    cameraBox.style.display = "block";
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    cameraPreview.srcObject = stream;
};

// Capture from camera
captureBtn.onclick = () => {
    const canvas = document.createElement("canvas");
    canvas.width = cameraPreview.videoWidth;
    canvas.height = cameraPreview.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(cameraPreview, 0, 0);

    canvas.toBlob(blob => {
        capturedImageFile = new File([blob], "camera.jpg", { type: "image/jpeg" });
        previewArea.innerHTML = "";
        addPreview(URL.createObjectURL(blob));
    });
};

// File Upload
fileInput.onchange = () => {
    capturedImageFile = null;
    previewArea.innerHTML = "";

    [...fileInput.files].forEach(file => {
        addPreview(URL.createObjectURL(file));
    });
};

function addPreview(url) {
    const img = document.createElement("img");
    img.src = url;
    img.classList.add("preview-item");
    previewArea.appendChild(img);
}

// Submit
submitBtn.onclick = async () => {
    const formData = new FormData();

    if (capturedImageFile) formData.append("image", capturedImageFile);
    else if (fileInput.files.length > 0) formData.append("image", fileInput.files[0]);
    else return alert("Upload or capture an image first.");

    const res = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const html = await res.text();

    // Replace current page with result page
    document.open();
    document.write(html);
    document.close();
};
