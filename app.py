import gradio as gr
import pandas as pd
import numpy as np
import sys
import os

# Add backend to path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from model import create_complete_prediction

def predict_player_stats(player_name, target_stat, spread, total):
    """
    Make prediction using the model
    """
    try:
        # Validate inputs
        if not player_name or not player_name.strip():
            return "❌ Please enter a player name"
        
        # Call your existing model
        result = create_complete_prediction(
            player_name.strip(), 
            target_stat, 
            float(spread), 
            float(total)
        )
        
        if result is None:
            return f"❌ Prediction failed for {player_name}. Player may not be found or insufficient data available."
        
        # Format the output
        prediction = result.get('prediction', 0)
        confidence = result.get('confidence', 0)
        usage_rate = result.get('usage_rate', 0)
        game_context = result.get('game_context', {})
        individual_preds = result.get('individual_predictions', {})
        
        output = f"""
## 🏀 Prediction for {result['player']}

### 📊 **{target_stat} Prediction: {prediction:.2f}**

---

### 📈 Model Details
- **Confidence**: {confidence:.1f}%
- **Usage Rate**: {usage_rate:.1f}%

### 🎮 Game Context
- **Opponent**: {game_context.get('opponent', 'N/A')}
- **Location**: {'🏠 Home' if game_context.get('is_home', False) else '✈️ Away'}
- **Spread**: {game_context.get('spread', 0):+.1f}
- **Total**: {game_context.get('total', 0):.1f}

### 🤖 Individual Model Predictions
"""
        for model_name, pred in individual_preds.items():
            output += f"- **{model_name}**: {pred:.2f}\n"
        
        return output
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ Error: {str(e)}\n\nDetails:\n{error_details}"

# Create Gradio interface with dark theme
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as demo:
    gr.Markdown("""
    # 🏀 NBA Player Prop Predictor
    
    Predict player statistics using ensemble machine learning models.
    
    **Powered by**: Random Forest, Gradient Boosting, XGBoost, Bayesian Ridge, Linear Regression
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
            
            predict_btn = gr.Button("🚀 Generate Prediction", variant="primary", size="lg")
        
        with gr.Column():
            output = gr.Markdown(label="Results", value="Results will appear here...")
    
    predict_btn.click(
        fn=predict_player_stats,
        inputs=[player_input, stat_input, spread_input, total_input],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### ℹ️ How it works
    This model uses ensemble machine learning to predict NBA player statistics based on:
    - Recent game performance (rolling averages)
    - Usage rate and team dynamics  
    - Opponent defensive statistics
    - Game context (home/away, spread, total)
    
    ### 📊 Data Sources
    - NBA official stats API
    - Hashtag Basketball defensive stats
    - Real-time usage rate data
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()