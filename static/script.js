const coins = ["bitcoin", "ethereum", "binancecoin", "cardano"];
    let predictionChart = null;

    function fetchLivePrices() {
      coins.forEach(coin => {
        fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${coin}&vs_currencies=usd&include_24hr_change=true`)
          .then(res => res.json())
          .then(data => {
            const price = data[coin].usd;
            const change = data[coin].usd_24h_change.toFixed(2);
            const card = document.getElementById(coin);
            card.querySelector(".price").innerText = price;
            card.querySelector(".change").innerText = `${change}%`;
            card.querySelector(".change").style.color = change < 0 ? "red" : "green";
          }).catch(err => console.error(`Failed to fetch price for ${coin}:`, err));
      });
    }

    function updateChart(historicalPrices, predictedDate, predictedPrice, crypto) {
  try {
    console.log("updateChart called with:", { historicalPrices, predictedDate, predictedPrice, crypto });
    let canvas = document.getElementById('prediction-chart');
    const chartError = document.getElementById('chart-error');
    chartError.style.display = 'none';

    if (!canvas) {
      console.error("Canvas element not found for prediction-chart");
      chartError.textContent = "Error: Canvas not found. Please try again.";
      chartError.style.display = 'block';
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error("Canvas context not found for prediction-chart");
      chartError.textContent = "Error: Canvas context not available.";
      chartError.style.display = 'block';
      return;
    }

    if (!historicalPrices || !Array.isArray(historicalPrices) || historicalPrices.length === 0) {
      console.error("Invalid historicalPrices:", historicalPrices);
      chartError.textContent = "Error: No historical price data available.";
      chartError.style.display = 'block';
      return;
    }

    // Validate dates
    const isValidDate = (dateStr) => {
      const regex = /^\d{4}-\d{2}-\d{2}$/;
      if (!regex.test(dateStr)) return false;
      const date = new Date(dateStr);
      return !isNaN(date.getTime());
    };

    const invalidDates = historicalPrices.filter(data => !isValidDate(data.date));
    if (invalidDates.length > 0) {
      console.error("Invalid dates in historicalPrices:", invalidDates);
      chartError.textContent = "Error: Invalid date format in historical data.";
      chartError.style.display = 'block';
      return;
    }

    if (!isValidDate(predictedDate)) {
      console.error("Invalid predictedDate:", predictedDate);
      chartError.textContent = "Error: Invalid predicted date format.";
      chartError.style.display = 'block';
      return;
    }

    // Destroy all Chart.js instances on canvas
    if (typeof Chart !== 'undefined') {
      Object.keys(Chart.instances).forEach(id => {
        const chart = Chart.instances[id];
        if (chart.canvas.id === 'prediction-chart') {
          console.log("Destroying chart instance with ID:", id);
          chart.destroy();
        }
      });
    }
    predictionChart = null; // Clear global variable
    // Fully reset canvas by replacing it
    const newCanvas = canvas.cloneNode(true);
    canvas.parentNode.replaceChild(newCanvas, canvas);
    newCanvas.id = 'prediction-chart'; // Restore ID
    console.log("Canvas fully replaced and reset for prediction-chart");

    const isDarkMode = document.body.classList.contains('dark-mode');

    const historicalData = historicalPrices.map(data => ({
      x: data.date,
      y: data.price
    }));
    const predictedData = [{
      x: predictedDate,
      y: predictedPrice
    }];

    console.log("Chart data:", { historicalData, predictedData });

    predictionChart = new Chart(newCanvas.getContext('2d'), {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Historical Prices',
            data: historicalData,
            borderColor: isDarkMode ? '#90caf9' : '#007bff',
            backgroundColor: isDarkMode ? 'rgba(144, 202, 249, 0.1)' : 'rgba(0, 123, 255, 0.1)',
            fill: true,
            pointRadius: 3
          },
          {
            label: 'Predicted Price',
            data: predictedData,
            borderColor: '#ff4444',
            backgroundColor: 'rgba(255, 68, 68, 0.5)',
            pointRadius: 6,
            pointStyle: 'circle',
            showLine: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: {
              color: isDarkMode ? '#f1f1f1' : '#333'
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const label = context.dataset.label || '';
                const date = context.raw.x;
                const price = context.parsed.y.toFixed(2);
                return `${label}: Date: ${date}, Price: $${price}`;
              }
            }
          }
        },
        scales: {
          x: {
            type: 'time',
            time: {
              unit: 'day',
              displayFormats: {
                day: 'yyyy-MM-dd' // Changed from 'yyyy-MM-DD' to 'yyyy-MM-dd'
              }
            },
            title: {
              display: true,
              text: 'Date',
              color: isDarkMode ? '#f1f1f1' : '#333'
            },
            ticks: {
              color: isDarkMode ? '#f1f1f1' : '#333'
            },
            grid: {
              color: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Price (USD)',
              color: isDarkMode ? '#f1f1f1' : '#333'
            },
            ticks: {
              color: isDarkMode ? '#f1f1f1' : '#333',
              callback: function(value) {
                return `$${value.toFixed(2)}`;
              }
            },
            grid: {
              color: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
            }
          }
        }
      }
    });
    console.log("Chart rendered successfully for", crypto);
  } catch (err) {
    console.error("Error updating chart:", err, err.stack);
    document.getElementById('chart-error').textContent = `Error rendering graph: ${err.message}`;
    document.getElementById('chart-error').style.display = 'block';
  }
}
    document.addEventListener("DOMContentLoaded", () => {
      fetchLivePrices();

      const body = document.body;
      const toggle = document.getElementById("theme-toggle");

      if (localStorage.getItem("darkMode") === "enabled") {
        body.classList.add("dark-mode");
      }

      toggle.addEventListener("click", function () {
        body.classList.toggle("dark-mode");
        const mode = body.classList.contains("dark-mode") ? "enabled" : "disabled";
        localStorage.setItem("darkMode", mode);
        if (predictionChart) {
          const isDarkMode = body.classList.contains('dark-mode');
          predictionChart.data.datasets[0].borderColor = isDarkMode ? '#90caf9' : '#007bff';
          predictionChart.data.datasets[0].backgroundColor = isDarkMode ? 'rgba(144, 202, 249, 0.1)' : 'rgba(0, 123, 255, 0.1)';
          predictionChart.options.plugins.legend.labels.color = isDarkMode ? '#f1f1f1' : '#333';
          predictionChart.options.scales.x.title.color = isDarkMode ? '#f1f1f1' : '#333';
          predictionChart.options.scales.x.ticks.color = isDarkMode ? '#f1f1f1' : '#333';
          predictionChart.options.scales.x.grid.color = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
          predictionChart.options.scales.y.title.color = isDarkMode ? '#f1f1f1' : '#333';
          predictionChart.options.scales.y.ticks.color = isDarkMode ? '#f1f1f1' : '#333';
          predictionChart.options.scales.y.grid.color = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
          predictionChart.update();
        }
      });

      document.getElementById("predictBtn").addEventListener("click", sendPrediction);
      document.getElementById("days").addEventListener("keypress", function (e) {
        if (e.key === "Enter") sendPrediction();
      });

      document.getElementById("clearBtn").addEventListener("click", clearPrediction);
      document.getElementById("sendBtn").addEventListener("click", sendMessage);
      document.getElementById("userInput").addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendMessage();
        }
      });
    });

    function toggleChatbot() {
      const chatbox = document.getElementById("chatbox");
      chatbox.style.display = chatbox.style.display === "none" ? "block" : "none";
    }

    function sendMessage() {
      const input = document.getElementById("userInput");
      const userMsg = input.value.trim();
      if (!userMsg) {
        console.log("Empty message, ignoring");
        return;
      }

      console.log("Sending message:", userMsg);
      const messagesDiv = document.getElementById("messages");
      messagesDiv.innerHTML += `<div style="text-align: right; margin: 8px 0;"><span style="background: #007bff; color: white; padding: 8px 12px; border-radius: 18px; display: inline-block; max-width: 80%;">${userMsg}</span></div>`;
      messagesDiv.scrollTop = messagesDiv.scrollHeight;

      fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg })
      })
        .then(response => {
          console.log("Fetch response status:", response.status);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.json();
        })
        .then(data => {
          console.log("Received response:", data);
          const botMsg = data.response || "Sorry, I didn't get a response. Try again!";
          messagesDiv.innerHTML += `<div style="text-align: left; margin: 8px 0;"><span style="background: #e0e0e0; color: black; padding: 8px 12px; border-radius: 18px; display: inline-block; max-width: 80%;">${botMsg}</span></div>`;
          messagesDiv.scrollTop = messagesDiv.scrollHeight;
        })
        .catch(error => {
          console.error("Error sending message:", error);
          messagesDiv.innerHTML += `<div style="text-align: left; margin: 8px 0;"><span style="background: #e0e0e0; color: red; padding: 8px 12px; border-radius: 18px; display: inline-block; max-width: 80%;">Error: Could not reach the server. Please try again.</span></div>`;
          messagesDiv.scrollTop = messagesDiv.scrollHeight;
        });

      input.value = "";
    }

    function clearPrediction() {
      const resultDiv = document.getElementById("result");
      const alertsDiv = document.getElementById("alerts");
      const chartError = document.getElementById("chart-error");
      const canvas = document.getElementById("prediction-chart");

      resultDiv.style.display = "none";
      alertsDiv.innerHTML = "";
      chartError.style.display = "none";

      if (predictionChart) {
        console.log("Clearing chart instance in clearPrediction");
        predictionChart.destroy();
        predictionChart = null;
      }

      if (canvas) {
        // Reset canvas
        canvas.width = canvas.width;
        console.log("Canvas reset in clearPrediction");
      }

      document.getElementById("crypto").value = "bitcoin";
      document.getElementById("days").value = "1";
      console.log("Prediction cleared");
    }

    function sendPrediction() {
      const crypto = document.getElementById("crypto").value;
      const days = document.getElementById("days").value;
      const resultDiv = document.getElementById("result");
      const alertsDiv = document.getElementById("alerts");
      const chartError = document.getElementById("chart-error");

      console.log(`Fetching prediction for ${crypto}, ${days} days`);
      fetch(`/predict?crypto=${crypto}&days=${days}`, { cache: 'no-store' })
        .then(res => {
          console.log("Fetch response status:", res.status);
          if (!res.ok) {
            return res.json().then(err => { throw new Error(err.error || `HTTP error! Status: ${res.status}`); });
          }
          return res.json();
        })
        .then(data => {
          console.log("Received prediction data:", JSON.stringify(data, null, 2));
          chartError.style.display = "none";
          if (data.predicted_price !== undefined && !data.error) {
            document.getElementById("predicted-price").innerText = `$${data.predicted_price.toFixed(2)}`;
            document.getElementById("predicted-date").innerText = data.predicted_date;
            document.getElementById("actual-price").innerText = `$${data.actual_price.toFixed(2)}`;
            document.getElementById("actual-date").innerText = data.actual_date;
            document.getElementById("accuracy").innerText = `${data.accuracy_percentage.toFixed(2)}%`;
            
            alertsDiv.innerHTML = "";
            if (data.alerts && Array.isArray(data.alerts)) {
              data.alerts.forEach(alert => {
                alertsDiv.innerHTML += `<div class="alert alert-warning">${alert}</div>`;
              });
            }

            updateChart(data.historical_prices, data.predicted_date, data.predicted_price, crypto);
            resultDiv.style.display = "block";
          } else {
            console.error("Invalid response data:", data);
            let errorMessage = data.error || "Invalid response from server. Please try again.";
            resultDiv.innerHTML = `<h4>Prediction Error</h4><p>${errorMessage}</p>`;
            resultDiv.style.display = "block";
            if (predictionChart) {
              console.log("Clearing chart instance due to prediction error");
              predictionChart.destroy();
              predictionChart = null;
            }
            const canvas = document.getElementById("prediction-chart");
            if (canvas) {
              canvas.width = canvas.width;
              console.log("Canvas reset due to prediction error");
            }
          }
        })
        .catch(err => {
          console.error("Fetch error:", err);
          let errorMessage = `Failed to fetch prediction: ${err.message}. Please try again.`;
          resultDiv.innerHTML = `<h4>Prediction Error</h4><p>${errorMessage}</p>`;
          resultDiv.style.display = "block";
          if (predictionChart) {
            console.log("Clearing chart instance due to fetch error");
            predictionChart.destroy();
            predictionChart = null;
          }
          const canvas = document.getElementById("prediction-chart");
          if (canvas) {
            canvas.width = canvas.width;
            console.log("Canvas reset due to fetch error");
          }
        });
    }
function filterNews() {
    const crypto = document.getElementById("cryptoFilter").value;
    const query = document.getElementById("searchQuery").value;
    let url = "/news";
    if (crypto || query) {
        url += "?";
        if (crypto) url += `crypto=${crypto}`;
        if (query) url += `${crypto ? '&' : ''}q=${encodeURIComponent(query)}`;
    }
    window.location.href = url;
}
