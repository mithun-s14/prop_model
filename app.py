import gradio as gr
import pandas as pd
import numpy as np
from model import create_complete_prediction

def predict_player_stats(player_name, target_stat, spread, total):
    """
    Make prediction using the model
    """
    try:
        # Call your existing model
        result = create_complete_prediction(player_name, target_stat, float(spread), float(total))
        
        if result is None:
            return f"❌ Prediction failed for {player_name}. Player may not be found or insufficient data available."
        
        # Format the output
        output = f"""
## 🏀 Prediction for {result['player']}

### 📊 **{target_stat} Prediction: {result['prediction']:.2f}**

---

### 📈 Model Details
- **Confidence**: {result['confidence']:.1f}%
- **Usage Rate**: {result['usage_rate']:.1f}%

### 🎮 Game Context
- **Opponent**: {result['game_context']['opponent']}
- **Location**: {'🏠 Home' if result['game_context']['is_home'] else '✈️ Away'}
- **Spread**: {result['game_context']['spread']:+.1f}
- **Total**: {result['game_context']['total']:.1f}

### 🤖 Individual Model Predictions
"""
        for model_name, pred in result['individual_predictions'].items():
            output += f"- **{model_name}**: {pred:.2f}\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏀 NBA Player Prop Predictor
    
    Predict player statistics using ensemble machine learning models.
    Powered by: Random Forest, Gradient Boosting, XGBoost, Bayesian Ridge, Linear Regression
    """)
    
    with gr.Row():
        with gr.Column():
            player_input = gr.Textbox(
                label="👤 Player Name",
                placeholder="e.g., LeBron James",
                value="LeBron James"
            )
            
            stat_input = gr.Dropdown(
                label="🎯 Target Stat",
                choices=["PTS", "AST", "REB", "STL", "BLK", "TOV", "FG3M"],
                value="PTS"
            )
            
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
            
            predict_btn = gr.Button("🚀 Generate Prediction", variant="primary", size="lg")
        
        with gr.Column():
            output = gr.Markdown(label="Results")
    
    predict_btn.click(
        fn=predict_player_stats,
        inputs=[player_input, stat_input, spread_input, total_input],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### How it works
    This model uses ensemble machine learning to predict NBA player statistics based on:
    - Recent game performance (rolling averages)
    - Usage rate and team dynamics
    - Opponent defensive statistics
    - Game context (home/away, spread, total)
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()