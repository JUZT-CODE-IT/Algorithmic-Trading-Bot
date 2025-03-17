import React, { useState } from "react";
import "./App.css";

// SearchComponent definition
function SearchComponent({ setResults }) {
  const [query, setQuery] = useState("");

  const searchStocks = async () => {
    if (!query.trim()) return; // Do not search if the query is empty
    const response = await fetch(`http://localhost:5000/search?query=${query}`);
    const data = await response.json();
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
      <button onClick={searchStocks} className="search-btn">
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
    setStatus("Starting trade...");
    const response = await fetch("http://localhost:5000/trade", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol }),
    });
    const data = await response.json();
    setStatus(data.status);
  };

  return (
    <div className="trade-app">
      <h1>Trade Stocks</h1>

      {/* Integrate the SearchComponent here */}
      <SearchComponent setResults={setResults} />

      {/* Show tradable assets */}
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
      <button onClick={() => {}} className="backtest-btn">
        Backtest Strategy
      </button>

      <p className="status">{status}</p>
    </div>
  );
}
