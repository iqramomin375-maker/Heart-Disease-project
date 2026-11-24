const form =
    document.getElementById("predictionForm");

const loading =
    document.getElementById("loading");

const resultSection =
    document.getElementById("resultSection");

form.addEventListener("submit", async function(event) {

    event.preventDefault();
    loading.classList.remove("hidden");

    resultSection.classList.add("hidden");

    const data = {

        age:
            document.getElementById("age").value,

        sex:
            document.getElementById("sex").value,

        cp:
            document.getElementById("cp").value,

        trestbps:
            document.getElementById("trestbps").value,

        chol:
            document.getElementById("chol").value,

        fbs:
            document.getElementById("fbs").value,

        restecg:
            document.getElementById("restecg").value,

        thalach:
            document.getElementById("thalach").value,

        exang:
            document.getElementById("exang").value,

        oldpeak:
            document.getElementById("oldpeak").value,

        slope:
            document.getElementById("slope").value,

        ca:
            document.getElementById("ca").value,

        thal:
            document.getElementById("thal").value
    };

    try {

        const response =
            await fetch("/predict", {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify(data)
            });



        const result =
            await response.json();



        loading.classList.add("hidden");

        if (!result.success) {

            alert(
                "Prediction error: "
                + result.error
            );

            return;
        }



        showResult(result);

    }

    catch(error) {

        loading.classList.add("hidden");

        alert(
            "Could not connect to the server."
        );

        console.error(error);
    }

});

function showResult(result) {

    resultSection.classList.remove("hidden");


    document.getElementById(
        "resultText"
    ).textContent =
        result.prediction_text;

    document.getElementById(
        "probability"
    ).textContent =
        result.probability + "%";

    document.getElementById(
        "riskLevel"
    ).textContent =
        result.risk_level;

    const title =
        document.getElementById(
            "resultTitle"
        );

    const icon =
        document.getElementById(
            "resultIcon"
        );

    if(result.prediction === 1) {

        title.textContent =
            "Heart Disease Likely";

        icon.textContent =
            "⚠️";

    }

    else {

        title.textContent =
            "Heart Disease Unlikely";

        icon.textContent =
            "❤️";

    }

    fillList(
        "dietList",
        result.recommendations.diet
    );


    fillList(
        "exerciseList",
        result.recommendations.exercise
    );


    fillList(
        "lifestyleList",
        result.recommendations.lifestyle
    );


    fillList(
        "followupList",
        result.recommendations.followup
    );



    resultSection.scrollIntoView({
        behavior: "smooth"
    });

}

function fillList(elementId, items) {

    const list =
        document.getElementById(
            elementId
        );

    list.innerHTML = "";

    items.forEach(function(item) {

        const li =
            document.createElement("li");

        li.textContent = item;

        list.appendChild(li);

    });

}

function newPrediction() {

    form.reset();

    resultSection.classList.add("hidden");

    document
        .getElementById("prediction")
        .scrollIntoView({
            behavior: "smooth"
        });

}