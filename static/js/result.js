document.addEventListener("DOMContentLoaded", () => {
    const resultData = window.resultData; 
    if (!resultData) return;

    const prediction = resultData.prediction;
    const fusion = resultData.fusion;

    const disease = prediction.label.toLowerCase().replace(" ", "_");

    // =============================
    //  SET DISEASE IMAGE
    // =============================
    const diseaseImage = document.getElementById("disease-image");
    diseaseImage.src = `/static/diseases/images/${disease}.jpg`;

    // =============================
    //  SET DISEASE NAME
    // =============================
    document.getElementById("disease-name").textContent = prediction.label;

    // =============================
    //  SET AFFECTED STAGE
    // =============================
    const stageBox = document.getElementById("disease-stage");
    stageBox.textContent = fusion.stage;

    if (fusion.stage === "Healthy") {
        stageBox.classList.add("stage-healthy");
    } else if (fusion.stage === "Partially Affected") {
        stageBox.classList.add("stage-partial");
    } else {
        stageBox.classList.add("stage-full");
    }

    // =============================
    //  LOAD DISEASE DESCRIPTION FILE
    // =============================
    fetch(`/static/diseases/${disease}.txt`)
        .then(response => response.text())
        .then(text => {
            const sections = parseDiseaseText(text);
            renderSection("about-text", sections.ABOUT);
            renderList("symptoms-list", sections.SYMPTOMS);
            renderList("prevention-list", sections.PREVENTION);
        })
        .catch(err => {
            console.error("Error loading disease text file:", err);
        });

    // =============================
    //  HELPERS
    // =============================

    function parseDiseaseText(text) {
        const lines = text.split("\n");

        let section = "";
        const data = { ABOUT: "", SYMPTOMS: [], PREVENTION: [] };

        lines.forEach(line => {
            line = line.trim();

            if (line === "ABOUT:") section = "ABOUT";
            else if (line === "SYMPTOMS:") section = "SYMPTOMS";
            else if (line === "PREVENTION:") section = "PREVENTION";
            else if (line.length > 0) {
                if (section === "ABOUT") data.ABOUT += line + " ";
                if (section === "SYMPTOMS") data.SYMPTOMS.push(line);
                if (section === "PREVENTION") data.PREVENTION.push(line);
            }
        });

        return data;
    }

    function renderSection(id, text) {
        document.getElementById(id).textContent = text;
    }

    function renderList(id, items) {
        const ul = document.getElementById(id);
        ul.innerHTML = "";
        items.forEach(item => {
            const li = document.createElement("li");
            li.textContent = item;
            ul.appendChild(li);
        });
    }
});
