import React, { useState } from "react";
import {
  LineChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Line,
} from "recharts";
import "./App.css";

// SearchComponent definition
function SearchComponent({ setResults, setLoading }) {
  const [query, setQuery] = useState("");

  const searchStocks = async () => {
    if (!query.trim()) return;
    setLoading(true);

    try {
      const response = await fetch(`http://localhost:5000/search?query=${query}`);
      const data = await response.json();
      setLoading(false);

      if (data.error) {
        alert(data.error);
        setResults([]);
      } else {
        const formattedResults = Object.entries(data).map(([symbol, name]) => ({
          symbol,
          name,
        }));
        setResults(formattedResults);
      }
    } catch (error) {
      console.error("Search Error:", error);
      setLoading(false);
    }
  };

  return (
    <div className="search-container">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search stock..."
      />
      <button onClick={searchStocks} className="search-btn" disabled={query.trim() === ""}>
        Search
      </button>
    </div>
  );
}

export default function TradeApp() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedStock, setSelectedStock] = useState(null);
  const [tradeData, setTradeData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [showTradeImage, setShowTradeImage] = useState(false);

  const startPrediction = async (symbol, name) => {
    setLoading(true);
    try {
      // Step 1: Get price data from /trade
      const tradeRes = await fetch('http://localhost:5000/trade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, name }),
      });

      const priceData = await tradeRes.json();

      // Step 2: Send price data to /predict
      const predictRes = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(priceData),
      });

      const prediction = await predictRes.json();

      if (!prediction || prediction.error) {
        alert('Prediction failed');
        setLoading(false);
        return;
      }

      setSelectedStock({ symbol, name });
      setPredictionData(prediction);
      setLoading(false);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Prediction failed');
      setLoading(false);
    }
  };

  const executeTrade = async () => {
    if (!selectedStock) return;

    try {
      const response = await fetch('http://localhost:5000/trade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: selectedStock.symbol,
          name: selectedStock.name,
        }),
      });

      const data = await response.json();

      if (!data || !data.trade_details) {
        alert('Invalid trade data received');
        return;
      }

      setTradeData(data.trade_details);
      setShowTradeImage(true);
    } catch (error) {
      console.error('Trade Error:', error);
    }
  };

  const goBack = () => {
    setSelectedStock(null);
    setPredictionData(null);
    setTradeData(null);
    setShowTradeImage(false);
    setResults([]);
  };

  return (
    <div className="app-container">
      <h1 className="heading">Stock Prediction Bot</h1>

      {selectedStock && predictionData && (
        <div className="trade-details">
          <h2>
            Prediction for {selectedStock.name} ({selectedStock.symbol})
          </h2>
          <p><strong>Prediction:</strong> {predictionData.prediction}</p>
          <p><strong>Confidence:</strong> {predictionData.confidence || 'N/A'}</p>

          {Array.isArray(predictionData.price_data) &&
          predictionData.price_data.length > 0 ? (
            <div>
              <h3>Price History</h3>
              <LineChart width={500} height={300} data={predictionData.price_data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="price" stroke="#82ca9d" />
              </LineChart>
            </div>
          ) : (
            <p>No price data available.</p>
          )}

          {!tradeData ? (
            <button onClick={executeTrade} className="trade-btn">
              Execute Trade
            </button>
          ) : (
            <>
              <h3>Trade Executed</h3>
              <p>Price: ${tradeData.price}</p>
              <p>Status: {tradeData.order_status}</p>
              {tradeData.model_results && (
                <>
                  <p>Accuracy: {tradeData.model_results.accuracy}%</p>
                  <p>F1 Score: {tradeData.model_results.f1_score}</p>
                </>
              )}
              {showTradeImage && (
                <img
                  src={`${process.env.PUBLIC_URL}/trade_decision.png`}
                  alt="Trade Decision"
                  className="rounded-lg shadow-lg"
                />
              )}
            </>
          )}

          <button onClick={goBack} className="back-btn">
            Go Back
          </button>
        </div>
      )}

      {!selectedStock && (
        <div className="trade-app">
          <SearchComponent setResults={setResults} setLoading={setLoading} />
          <div className="trade-btn-container">
            {results.map((stock) => (
              <button
                key={stock.symbol}
                onClick={() => startPrediction(stock.symbol, stock.name)}
                className="trade-btn"
              >
                Trade {stock.name} ({stock.symbol})
              </button>
            ))}
          </div>

          {loading && <div className="spinner"></div>}
        </div>
      )}
    </div>
  );
}