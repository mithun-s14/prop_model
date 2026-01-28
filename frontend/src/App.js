import React, { useState } from 'react';

export default function NBAPredictor() {
  const [playerName, setPlayerName] = useState('');
  const [targetStat, setTargetStat] = useState('points');
  const [spread, setSpread] = useState('');
  const [total, setTotal] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  const handlePredict = async () => {
    if (!playerName.trim()) {
      setError('Please enter a player name');
      return;
    }
    if (!spread || !total) {
      setError('Please enter both spread and total');
      return;
    }

    setLoading(true);
    setError('');

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        player_name: playerName,
        target_stat: targetStat,
        spread: parseFloat(spread),
        total: parseFloat(total)
      })
    });
      
      if (!response.ok) {
        throw new Error('Prediction failed');
      }
      
      const data = await response.json();
      setPredictions(data);

    } catch (err) {
      setError('Failed to get predictions. Make sure your API is running on localhost:5000');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const statOptions = [
    { value: 'points', label: 'Points' },
    { value: 'assists', label: 'Assists' },
    { value: 'rebounds', label: 'Rebounds' },
  ];

  return (
    <div className="min-h-screen bg-zinc-950 p-4 md:p-8">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <svg className="w-10 h-10 md:w-12 md:h-12 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <h1 className="text-4xl md:text-5xl font-bold text-white">Prop Model</h1>
          </div>
          <p className="text-zinc-400 text-base md:text-lg">Predicting players' stats with ensemble models: Bayesian, Gradient Boost, LightGBM, Linear, Random Forest, and XGBoost</p>
        </div>

        <div className="bg-zinc-900 rounded-2xl p-6 md:p-8 shadow-2xl border border-zinc-800 mb-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-white font-semibold mb-2">
                👤 Player Name
              </label>
              <input
                type="text"
                value={playerName}
                onChange={(e) => setPlayerName(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handlePredict()}
                placeholder="e.g., LeBron James"
                className="w-full px-4 py-3 rounded-lg bg-zinc-800 text-white placeholder-zinc-500 border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-white font-semibold mb-2">
                🎯 Target Stat
              </label>
              <select
                value={targetStat}
                onChange={(e) => setTargetStat(e.target.value)}
                className="w-full px-4 py-3 rounded-lg bg-zinc-800 text-white border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
              >
                {statOptions.map(opt => (
                  <option key={opt.value} value={opt.value} className="bg-zinc-800">
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-white font-semibold mb-2">
                📊 Spread
              </label>
              <input
                type="number"
                step="0.5"
                value={spread}
                onChange={(e) => setSpread(e.target.value)}
                placeholder="e.g., -5.5"
                className="w-full px-4 py-3 rounded-lg bg-zinc-800 text-white placeholder-zinc-500 border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-white font-semibold mb-2">
                📈 Total
              </label>
              <input
                type="number"
                step="0.5"
                value={total}
                onChange={(e) => setTotal(e.target.value)}
                placeholder="e.g., 225.5"
                className="w-full px-4 py-3 rounded-lg bg-zinc-800 text-white placeholder-zinc-500 border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
              />
            </div>
          </div>

          {error && (
            <div className="bg-red-900/30 border border-red-500 text-red-200 px-4 py-3 rounded-lg mb-6">
              {error}
            </div>
          )}

          <button
            onClick={handlePredict}
            disabled={loading}
            className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-4 px-6 rounded-lg transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none shadow-lg"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                Running ML Models...
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                📊 Generate Prediction
              </span>
            )}
          </button>
        </div>

        {predictions && (
          <div className="space-y-6 animate-fade-in">
            <div className="bg-zinc-900 rounded-2xl p-6 md:p-8 shadow-2xl border border-zinc-800">
              <h2 className="text-2xl md:text-3xl font-bold text-white mb-6 text-center">
                {predictions.player} - {statOptions.find(s => s.value === predictions.target_stat)?.label || predictions.target_stat}
              </h2>
              
              <div className="bg-gradient-to-br from-green-600/20 to-green-700/20 border-2 border-green-500/50 rounded-xl p-8 text-center mb-6">
                <div className="text-green-400 text-sm font-semibold mb-2">ENSEMBLE PREDICTION</div>
                <div className="text-6xl md:text-7xl font-bold text-white mb-2">{predictions.prediction.toFixed(2)}</div>
                <div className="text-green-300 text-sm">{statOptions.find(s => s.value === predictions.target_stat)?.label || predictions.target_stat}</div>
              </div>

              {predictions.game_context && (
                <div className="bg-zinc-800 rounded-lg p-4 border border-zinc-700 mb-6">
                  <h3 className="text-white font-semibold mb-3">Game Context</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div>
                      <div className="text-zinc-400">Opponent</div>
                      <div className="text-white font-bold">{predictions.game_context.opponent}</div>
                    </div>
                    <div>
                      <div className="text-zinc-400">Location</div>
                      <div className="text-white font-bold">{predictions.game_context.is_home ? '🏠 Home' : '✈️ Away'}</div>
                    </div>
                    <div>
                      <div className="text-zinc-400">Spread</div>
                      <div className="text-white font-bold">{predictions.game_context.spread > 0 ? '+' : ''}{predictions.game_context.spread}</div>
                    </div>
                    <div>
                      <div className="text-zinc-400">Total</div>
                      <div className="text-white font-bold">{predictions.game_context.total}</div>
                    </div>
                  </div>
                </div>
              )}

              {predictions.individual_predictions && (
                <div className="bg-zinc-800 rounded-lg p-4 border border-zinc-700">
                  <h3 className="text-white font-semibold mb-3">Individual Model Predictions</h3>
                  <div className="space-y-2">
                    {Object.entries(predictions.individual_predictions).map(([model, pred]) => (
                      <div key={model} className="flex justify-between items-center">
                        <span className="text-zinc-400 text-sm">{model}</span>
                        <span className="text-white font-bold">{pred.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div className="mt-8 text-center text-zinc-500 text-sm">
          <p>Disclaimer: Predictions are for informational purposes only and not guaranteed.</p>
        </div>
      </div>
    </div>
  );
}