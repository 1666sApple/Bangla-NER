document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("predictionForm");
  const resultTable = document.querySelector("#resultTable tbody");

  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    const sentence = document.getElementById("inputSentence").value;

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ sentence: sentence }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      resultTable.innerHTML = "";
      data.prediction.forEach((item) => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${item[0]}</td>
            <td>${item[1]}</td>
            <td>${item[2]}</td>
          `;
        resultTable.appendChild(row);
      });
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred while processing your request.");
    }
  });
});
