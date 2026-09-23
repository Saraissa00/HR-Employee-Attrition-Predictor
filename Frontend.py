import gradio as gr
import joblib
import pandas as pd

# 1. Load your trained "Brain"
model = joblib.load('models/hr_model.pkl')


def predict_attrition(satisfaction, evaluation, projects, hours, tenure, accident, promotion, salary_level):
    # Convert words to numbers for the AI
    salary_map = {"Low": 0, "Medium": 1, "High": 2}

    # Feature order exactly as the AI learned it
    feature_cols = ['satisfaction_level', 'last_evaluation', 'number_project',
                    'average_montly_hours', 'time_spend_company', 'Work_accident',
                    'promotion_last_5years', 'sales', 'salary']

    # Create the data row (Using 7 as a default for 'sales' department)
    input_row = pd.DataFrame([[satisfaction / 100, evaluation / 100, projects, hours, tenure,
                               1 if accident else 0, 1 if promotion else 0, 7, salary_map[salary_level]]],
                             columns=feature_cols)

    # Get the answer
    prediction = model.predict(input_row)

    if prediction[0] == 1:
        return "🚩 RESULT: Employee is likely to LEAVE."
    else:
        return "✅ RESULT: Employee is likely to STAY."


# 2. Create the simple "Cover"
with gr.Blocks(title="AI HR Tool") as demo:
    gr.Markdown("# 🏢 AI HR Prediction Tool")
    gr.Markdown("Adjust the sliders and click the button to see the AI forecast.")

    with gr.Row():
        # Input Section
        with gr.Column():
            s = gr.Slider(0, 100, label="Satisfaction %", value=70)
            e = gr.Slider(0, 100, label="Evaluation %", value=70)
            p = gr.Number(label="Projects", value=3)
            h = gr.Number(label="Monthly Hours", value=180)
            t = gr.Number(label="Years at Company", value=3)
            a = gr.Checkbox(label="Work Accident?")
            pr = gr.Checkbox(label="Promoted in last 5 years?")
            sal = gr.Dropdown(["Low", "Medium", "High"], label="Salary Level", value="Medium")

            btn = gr.Button("RUN AI FORECAST", variant="primary")

        # Output Section
        with gr.Column():
            output_text = gr.Textbox(label="AI Analysis", interactive=False)

    btn.click(predict_attrition, [s, e, p, h, t, a, pr, sal], output_text)

# 3. Launch
demo.launch()