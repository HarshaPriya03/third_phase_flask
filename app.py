import io
import sys
import pickle
from flask import Flask, render_template, request, jsonify
from datetime import date
import pandas as pd
import numpy as np

# Assuming your functions are in a separate module called 'leave_functions.py'
from functions import fetch_leave_data, predict_sick_leave, rejected_and_absent, same_diff_leaves,check_casual_leave_exceeded, check_casual_leave_exceeded_one, high_leave_frequency, can_apply_leave, get_weekdays,on_submit

app = Flask(__name__)

# Load the pre-trained vectorizer and model
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/log_reg_model.pkl', 'rb') as f:
    log_reg_model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        email = request.form.get('email')
        leave_type = request.form.get('leave_type')
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
        reason = request.form.get('reason')

        # Convert dates from string to date objects
        from_date = date.fromisoformat(from_date)
        to_date = date.fromisoformat(to_date)



        output = {}
        output = on_submit(email, leave_type, from_date, to_date, reason, output)
        return render_template('index.html', 
                           output=output, 
                           today=date.today(), 
                           email=email, 
                           leave_type=leave_type, 
                           from_date=from_date, 
                           to_date=to_date, 
                           reason=reason)

    return render_template('index.html', output=[], today=date.today())
# Add the actual on_submit function if needed (or import it from a different file)


if __name__ == '__main__':
    app.run(debug=True)
