import gradio as gr
import pandas as pd
import numpy as np
import sys
import os

# Add backend to path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from model import create_complete_prediction

STAT_MAP = {"Points": "points", "Assists": "assists", "Rebounds": "rebounds"}

custom_css = """
.gradio-container {
    background-color: #09090b !important;
    max-width: 900px !important;
    margin: auto !important;
}
.main-header {
    text-align: center;
    padding: 32px 0 8px 0;
}
.main-header h1 {
    color: #fff !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
}
.main-header p {
    color: #a1a1aa !important;
    font-size: 1rem !important;
    margin-top: 8px !important;
}
.input-card {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 16px !important;
    padding: 32px !important;
}
.input-card .gr-input, .input-card input, .input-card select,
.input-card .gr-box {
    background: #27272a !important;
    border: 1px solid #3f3f46 !important;
    color: #fff !important;
    border-radius: 8px !important;
}
.input-card label, .input-card .gr-label {
    color: #fff !important;
    font-weight: 600 !important;
}
.predict-btn {
    background: #16a34a !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 14px 24px !important;
    border-radius: 10px !important;
    border: none !important;
    width: 100% !important;
    margin-top: 8px !important;
    transition: background 0.2s !important;
}
.predict-btn:hover {
    background: #15803d !important;
}
.results-card {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 16px !important;
    padding: 32px !important;
}
.results-card h2, .results-card h3 {
    color: #fff !important;
}
.ensemble-box {
    background: linear-gradient(135deg, rgba(22,163,74,0.15), rgba(21,128,61,0.15)) !important;
    border: 2px solid rgba(34,197,94,0.4) !important;
    border-radius: 12px !important;
    padding: 32px !important;
    text-align: center !important;
    margin-bottom: 16px !important;
}
.ensemble-box .ensemble-label {
    color: #4ade80 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
}
.ensemble-box .ensemble-value {
    color: #fff !important;
    font-size: 4rem !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
    margin: 8px 0 !important;
}
.ensemble-box .ensemble-stat {
    color: #86efac !important;
    font-size: 0.85rem !important;
}
.context-box, .models-box {
    background: #27272a !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
}
.context-box h3, .models-box h3 {
    margin-top: 0 !important;
    font-size: 1rem !important;
    margin-bottom: 12px !important;
}
.context-grid {
    display: grid !important;
    grid-template-columns: 1fr 1fr 1fr 1fr !important;
    gap: 12px !important;
}
.context-item .context-label {
    color: #a1a1aa !important;
    font-size: 0.8rem !important;
}
.context-item .context-value {
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}
.model-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    padding: 6px 0 !important;
}
.model-row .model-name {
    color: #a1a1aa !important;
    font-size: 0.9rem !important;
}
.model-row .model-value {
    color: #fff !important;
    font-weight: 700 !important;
}
.disclaimer {
    text-align: center !important;
    color: #71717a !important;
    font-size: 0.85rem !important;
    padding: 24px 0 !important;
}
footer { display: none !important; }
"""


def predict_player_stats(player_name, target_stat, spread, total):
    try:
        if not player_name or not player_name.strip():
            return """<div style="color: #fca5a5; background: rgba(127,29,29,0.3); border: 1px solid #ef4444; border-radius: 8px; padding: 12px;">Please enter a player name</div>"""

        stat_key = STAT_MAP.get(target_stat, target_stat.lower())
        result = create_complete_prediction(
            player_name.strip(),
            stat_key,
            float(spread),
            float(total)
        )

        if result is None:
            return f"""<div style="color: #fca5a5; background: rgba(127,29,29,0.3); border: 1px solid #ef4444; border-radius: 8px; padding: 12px;">Prediction failed for {player_name}. Player may not be found or insufficient data available.</div>"""

        prediction = result.get('prediction', 0)
        game_context = result.get('game_context', {})
        individual_preds = result.get('individual_predictions', {})

        opponent = game_context.get('opponent', 'N/A')
        is_home = game_context.get('is_home', False)
        loc = '🏠 Home' if is_home else '✈️ Away'
        sp = game_context.get('spread', 0)
        spread_str = f"+{sp}" if sp > 0 else str(sp)
        tot = game_context.get('total', 0)

        models_html = ""
        for model_name, pred in individual_preds.items():
            models_html += f"""<div class="model-row"><span class="model-name">{model_name}</span><span class="model-value">{pred:.2f}</span></div>"""

        return f"""
<h2 style="text-align:center; color:#fff; font-size:1.6rem; margin-bottom:20px;">{result['player']} - {target_stat}</h2>

<div class="ensemble-box">
    <div class="ensemble-label">ENSEMBLE PREDICTION</div>
    <div class="ensemble-value">{prediction:.2f}</div>
    <div class="ensemble-stat">{target_stat}</div>
</div>

<div class="context-box">
    <h3 style="color:#fff;">Game Context</h3>
    <div class="context-grid">
        <div class="context-item"><div class="context-label">Opponent</div><div class="context-value">{opponent}</div></div>
        <div class="context-item"><div class="context-label">Location</div><div class="context-value">{loc}</div></div>
        <div class="context-item"><div class="context-label">Spread</div><div class="context-value">{spread_str}</div></div>
        <div class="context-item"><div class="context-label">Total</div><div class="context-value">{tot}</div></div>
    </div>
</div>

<div class="models-box">
    <h3 style="color:#fff;">Individual Model Predictions</h3>
    {models_html}
</div>
"""

    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return f"""<div style="color: #fca5a5; background: rgba(127,29,29,0.3); border: 1px solid #ef4444; border-radius: 8px; padding: 12px;">Prediction unavailable: <strong>{player_name.strip()}</strong> does not have a game today.</div>"""
        return f"""<div style="color: #fca5a5; background: rgba(127,29,29,0.3); border: 1px solid #ef4444; border-radius: 8px; padding: 12px;">Error: {str(e)}</div>"""
    except Exception as e:
        return f"""<div style="color: #fca5a5; background: rgba(127,29,29,0.3); border: 1px solid #ef4444; border-radius: 8px; padding: 12px;">Error: {str(e)}</div>"""


with gr.Blocks(css=custom_css, theme=gr.themes.Base(
    primary_hue="green",
    neutral_hue="zinc",
    font=gr.themes.GoogleFont("Inter"),
)) as demo:

    gr.HTML("""
    <div class="main-header">
        <h1>📊 Prop Model</h1>
        <p>Predicting players' stats with ensemble models: Bayesian, Gradient Boost, LightGBM, Linear, Random Forest, and XGBoost</p>
    </div>
    """)

    with gr.Group(elem_classes="input-card"):
        with gr.Row():
            player_input = gr.Textbox(
                label="👤 Player Name",
                placeholder="e.g., LeBron James",
            )
            stat_input = gr.Dropdown(
                label="🎯 Target Stat",
                choices=["Points", "Assists", "Rebounds"],
                value="Points"
            )
        with gr.Row():
            spread_input = gr.Number(
                label="📊 Spread",
                value=-5.5,
                step=0.5
            )
            total_input = gr.Number(
                label="📈 Total",
                value=225.5,
                step=0.5
            )
        predict_btn = gr.Button("📊 Generate Prediction", variant="primary", elem_classes="predict-btn")

    output = gr.HTML(elem_classes="results-card")

    predict_btn.click(
        fn=predict_player_stats,
        inputs=[player_input, stat_input, spread_input, total_input],
        outputs=output
    )

    gr.HTML('<div class="disclaimer">Disclaimer: Predictions are for informational purposes only and not guaranteed.</div>')


if __name__ == "__main__":
    demo.launch()
