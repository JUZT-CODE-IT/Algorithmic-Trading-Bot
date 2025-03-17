import React, { useState } from "react";
import "./App.css";

// SearchComponent definition
function SearchComponent({ setResults, setLoading }) {
  const [query, setQuery] = useState("");

  const searchStocks = async () => {
    if (!query.trim()) return; // Do not search if the query is empty
    setLoading(true); // Set loading state to true while fetching data
    const response = await fetch(`http://localhost:5000/search?query=${query}`);
    const data = await response.json();
    setLoading(false); // Set loading state to false after fetching data
    if (data.error) {
      alert(data.error); // Alert if the query fails
    } else {
      setResults(data); // Set results if successful
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

// Main TradeApp component
export default function TradeApp() {
  const [results, setResults] = useState({});
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  const startTrade = async (symbol) => {
    try {
        const response = await fetch("http://localhost:5000/trade", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ symbol }),
        });
        const data = await response.json();
        alert(data.status || data.error);
    } catch (error) {
        console.error("Trade Error:", error);
    }
  };

  return (
    <div className="app-container">
      {/* Heading outside the flexbox */}
      <h1 className="heading">Stock Prediction Bot</h1>

      <div className="trade-app">
        {/* Integrate the SearchComponent here */}
        <SearchComponent setResults={setResults} setLoading={setLoading} />

        {/* Display the results as a grid */}
        <div className="trade-btn-container">
          {Object.entries(results).map(([symbol, name]) => (
            <button
              key={symbol}
              onClick={() => startTrade(symbol)}
              className="trade-btn"
            >
              Trade {name} ({symbol})
            </button>
          ))}
        </div>

        {/* Display backtest button alongside trade options */}
        {Object.keys(results).length > 0 && (
          <button onClick={() => {}} className="backtest-btn">
            Backtest Strategy
          </button>
        )}

        {/* Show loading spinner if data is being fetched */}
        {loading && <div className="spinner"></div>}

        <p className="status">{status}</p>
      </div>
    </div>
  );
}
