import ipywidgets as widgets
from IPython.display import display, clear_output
import mysql.connector
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
from datetime import timedelta
from datetime import date, timedelta
import pickle
# Load the pre-trained vectorizer and model

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/log_reg_model.pkl', 'rb') as f:
    log_reg_model = pickle.load(f)



def connect_to_db():
    return mysql.connector.connect(
       host="localhost",
       user="root",
       password="",
       database="leave_data"
    )

def fetch_leave_data(email):
    try:
        conn=connect_to_db()
        cursor=conn.cursor(dictionary=True)
        query="""
        SELECT l.id, lb.empname, lb.cl, lb.sl, lb.co, lb.empemail, l.hrremark, l.mgrremark, l.aprremark, l.from, l.to, l.desg
        FROM leavebalance lb
        JOIN leaves l ON lb.empemail = l.empemail
        WHERE lb.empemail = %s
        """
        cursor.execute(query,(email,))
        result=cursor.fetchall()
        conn.close()
        return pd.DataFrame(result) if result else None
    except mysql.connector.Error as err:
        print(f"Error:{err}")
        return None
    
#leave eligibility

def can_apply_leave(data):
    if data is not None and not data.empty:
        # Convert columns to numeric, handling non-numeric values
        data["cl"] = pd.to_numeric(data["cl"], errors='coerce').fillna(0)
        data["sl"] = pd.to_numeric(data["sl"], errors='coerce').fillna(0)
        data["co"] = pd.to_numeric(data["co"], errors='coerce').fillna(0)
        
        # Calculate total leave balance (lb = cl + sl + co)
        data["lb"] = data["cl"] + data["sl"] + data["co"]
        
        # Check if leave can be applied (i.e., if total balance > 0)
        data["can_apply_leave"] = data["lb"] > 0
        return data
    else:
        return "not having enough leave balance"

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the dataset for leave classification
df = pd.read_csv(r"C:\Users\it\Downloads\leave_app\leave_app\type_of_leave.csv")  # Adjust path to your dataset

# Preprocessing function
def preprocess(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    words = text.split()  # Tokenization
    unwanted_keywords = ["feeling"]
    words = [word for word in words if word not in unwanted_keywords]
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing to the 'text' column in the dataset
df['processed_text'] = df['text'].apply(preprocess)

def check_casual_leave_exceeded(email):
    try:
        conn = connect_to_db()
        cursor = conn.cursor(dictionary=True)
        
        # Query to count the casual leave applications for the given email
        query = """
            SELECT empemail, COUNT(*) AS record_count
            FROM leaves
            WHERE leavetype = 'CASUAL LEAVE'
            AND DATE(applied) = DATE(`from`)
            AND MONTH(applied) = MONTH(CURRENT_DATE) 
            AND YEAR(applied) = YEAR(CURRENT_DATE)
            AND empemail = %s
            GROUP BY empemail
            HAVING COUNT(*) >= 2;

        """
        
        cursor.execute(query, (email,))
        result = cursor.fetchall()
        conn.close()

        # Return whether the employee has exceeded casual leave applications more than 2 times
        if result:
            return True
        return False
    
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return False
    
    
    
def get_weekdays(from_date, to_date):
    # Ensure dates are in datetime format
    from_date = pd.to_datetime(from_date)
    to_date = pd.to_datetime(to_date)
    
    # Adjust the range using timedelta
    date_range = pd.date_range(start=from_date - timedelta(days=1), end=to_date + timedelta(days=1))
    
    # Create a DataFrame with dates and weekdays
    weekdays_df = pd.DataFrame({
        "Date": date_range,
        "Weekday": date_range.day_name()  # Get the weekday name
    })
    
    return weekdays_df



def high_leave_frequency(email):
    try:
        conn = connect_to_db()
        cursor = conn.cursor(dictionary=True)

        # Get current month and year for filtering
        current_month = datetime.now().month  # Hardcoded month, update as needed
        current_year = datetime.now().year
        
        # Query to fetch leaves for the provided email
        query = """
                SELECT 
                    `from`, 
                    `to`
                FROM leaves
                WHERE empemail = %s AND YEAR(`from`) = %s AND MONTH(`from`) = %s
                """
        cursor.execute(query, (email, current_year, current_month))
        leaves = cursor.fetchall()
        
        total_leave_days = 0

        # Loop through the fetched leave data
        for leave in leaves:
            from_date = leave['from']  # 'from' is the column name for the start date
            to_date = leave['to']      # 'to' is the column name for the end date

            # Calculate number of leave days between from_date and to_date
            leave_days = (to_date - from_date).days + 1  # Including the last day

            # Check if the day before the from_date is a Sunday
            if (from_date - timedelta(days=1)).weekday() == 6:
                leave_days += 1  # Add Sunday before leave

            # Check if the day after the to_date is a Sunday
            if (to_date + timedelta(days=1)).weekday() == 6:
                leave_days += 1  # Add Sunday after leave

            # Add this leave period's total days to the overall leave count
            total_leave_days += leave_days

        # Close the cursor and connection
        conn.close()

        # If the total leave days exceed 6, return True
        if total_leave_days > 6:
            return True

        return False

    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return False
    
def predict_sick_leave(text, model, vectorizer):
    # Preprocess the input text
    processed_text = preprocess(text)
    # Convert the text to the feature vector
    text_vector = vectorizer.transform([processed_text])
    # Predict using the trained model
    prediction = model.predict(text_vector)
    # Return result
    if prediction[0] == 1:
        return "Sick Leave"
    else:
        return "Not Sick Leave"

#short time leave application
def same_diff_leaves(email):
    try:
        conn = connect_to_db()
        cursor = conn.cursor(dictionary=True)
        
        # Corrected SQL query
        query = """
        SELECT 
            l.empemail
        FROM 
            leaves l
        WHERE 
            l.empemail = %s  -- Filter for specific employee email
            AND MONTH(l.from) = MONTH(CURRENT_DATE)  -- Current month
            AND YEAR(l.from) = YEAR(CURRENT_DATE)  -- Current year
            AND (DATEDIFF(l.to, l.from) = 0 OR DATEDIFF(l.to, l.from) = 1)  -- Leave duration of 0 or 1 day
        GROUP BY 
            l.empemail  -- Group by employee email
        HAVING 
            COUNT(*) > 2  -- Employees who applied for leave more than twice in the same month
        """       
        cursor.execute(query, (email,))
        result = cursor.fetchall()
        conn.close()
        if result:
            return True
        return False
    
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return False

def rejected_and_absent(empemail):
    try:
        # Connect to the database
        conn = connect_to_db()
        cursor = conn.cursor(dictionary=True)

        # First, check if the status2 count is greater than 2 in the leaves table
        query_leaves = """
        SELECT COUNT(*) AS status2_count
        FROM leaves
        WHERE empemail = %s AND status = 2;
        """
        cursor.execute(query_leaves, (empemail,))
        status2_result = cursor.fetchone()

        # If the count of status2=1 is not greater than 2, return False
        if status2_result['status2_count'] < 2:
            return False

        # If status2=1 count > 2, proceed to check the absent table
        query_absent = """
        SELECT COUNT(*) AS absent_count
        FROM absent
        WHERE empname = (SELECT empname FROM leaves WHERE empemail = %s LIMIT 1)
        AND YEAR(AttendanceTime) = YEAR(CURRENT_DATE);
        """
        cursor.execute(query_absent, (empemail,))
        absent_result = cursor.fetchone()

        # If the count of absences in the current year is greater than 2, return True
        if absent_result['absent_count'] > 2:
            return True
        return False

    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return False


# def rejected_and_absent(empemail):
#     try:
#         # Get current date and calculate the date range (March of the current year to April of the next year)
#         today = date.today()
#         current_year = today.year
#         next_year = current_year + 1
        
#         # Start date: March 1st of the current year
#         start_date = f"{current_year}-03-01"
        
#         # End date: April 30th of the next year
#         end_date = f"{next_year}-04-30"

#         # Connect to the database
#         conn = connect_to_db()
#         cursor = conn.cursor(dictionary=True)

#         # First, check if the status2 count is greater than 2 in the leaves table
#         query_leaves = """
#         SELECT COUNT(*) AS status2_count
#         FROM leaves
#         WHERE empemail = %s AND status = 2;
#         """
#         cursor.execute(query_leaves, (empemail,))
#         status2_result = cursor.fetchone()

#         # If the count of status2=1 is not greater than 2, return False
#         if status2_result['status2_count'] < 2:
#             return False

#         # If status2=1 count > 2, proceed to check the absent table
#         query_absent = """
#         SELECT COUNT(*) AS absent_count
#         FROM absent
#         WHERE empname = (SELECT empname FROM leaves WHERE empemail = %s LIMIT 1)
#         AND AttendanceTime BETWEEN %s AND %s;
#         """
#         cursor.execute(query_absent, (empemail, start_date, end_date))
#         absent_result = cursor.fetchone()

#         # If the count of absences in the range is greater than 2, return True
#         if absent_result['absent_count'] > 2:
#             return True
#         return False

#     except mysql.connector.Error as err:
#         print(f"Database error: {err}")
#         return False
    
def check_casual_leave_exceeded_one(email):
    try:
        conn = connect_to_db()
        cursor = conn.cursor(dictionary=True)
        
        # Query to count the casual leave applications for the given email
        query = """
            SELECT empemail, COUNT(*) AS record_count
            FROM leaves
            WHERE leavetype = 'CASUAL LEAVE'
            AND DATE(applied) = DATE(`from`)
            AND MONTH(applied) = MONTH(CURRENT_DATE) 
            AND YEAR(applied) = YEAR(CURRENT_DATE)
            AND empemail = %s
            GROUP BY empemail
            HAVING COUNT(*) = 1;

        """
        
        cursor.execute(query, (email,))
        result = cursor.fetchall()
        conn.close()

        # Return whether the employee has exceeded casual leave applications more than 2 times
        if result:
            return True
        return False
    
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return False
    
def on_submit(email_value,leave_type_value,from_date_value,to_date_value,reason_value,ls):
    with output:
        clear_output()  # Clear previous outputs
        
        email = email_value
        data = fetch_leave_data(email)
        from_date = from_date_value
        to_date = to_date_value
        selected_leave_type = leave_type_value
         # Check for rejection if the leave reason is "personal problem" or "personal issue"
        leave_reason = reason_value
        if leave_reason.lower() in ["personal problem", "personal issue","personal"]:
            print("Decision : Rejected \n Detailed Feedback : Leave Rejected: Your leave request has been rejected due to the reason being a personal problem/issue \n Final Decision : Leave cannot be approved.")
            ls= {
            "Decision": "Rejected",
            "Detailed Feedback": "Your leave request has been rejected due to the reason being a personal problem/issue",
            "Final Decision" : "Leave cannot be approved"
            }
            return ls
        leave_status = predict_sick_leave(leave_reason, log_reg_model, vectorizer)  # Assume model and vectorizer are available

        if not from_date or not to_date:
            print("Decision : Rejected \n Detailed Feedback : Please select both 'From Date' and 'To Date' \n Final Decision : Incomplete data provided.")
            ls= {
            "Decision": "Rejected",
            "Detailed Feedback": "Please select both 'From Date' and 'To Date'",
            "Final Decision" : "Incomplete data provided."
            }
            return ls

        elif from_date > to_date:
            print("'Decision: Rejected \n Detailed feedback: From Date' cannot be later than 'To Date'. Please correct the dates. \n Final Decision: Invalid date range.")
            ls= {
            "Decision": "Rejected",
            "Detailed Feedback": "From Date' cannot be later than 'To Date'",
            "Final Decision" : "Invalid date range"
            }
            return ls
        
        employee_leave_rejection=rejected_and_absent(email)
        a="You are rejected more than 2 times and you are absent for more than 2 days in this current year." if employee_leave_rejection else ""

        same_diff = same_diff_leaves(email)
        l= "You are applying the leaves of same frequency" if same_diff else ""
        
        check_casual_leave=check_casual_leave_exceeded(email)
        k="Leave Rejected: You have already applied for Casual Leave more than 2 times where applied == from dates" if check_casual_leave else ""
        
        check_casual_leave_one=check_casual_leave_exceeded_one(email)
        b="You have already applied for leave where applied == from . If you attempt to apply for leave again on any other day, your request will be rejected."
        # Check high leave frequency
        is_high_frequency = high_leave_frequency(email)
        h = "Warning: Your leave frequency is high. You have already taken more than 6 days of leave this month." if is_high_frequency else ""

        
        # Extend the date range and check for Sundays
        delta = (to_date - from_date).days + 1  # Number of days for the leave
        extended_weekdays = get_weekdays(from_date, to_date)
        sunday_count = extended_weekdays[extended_weekdays['Weekday'] == 'Sunday'].shape[0]

        # Adjust delta by adding the number of Sundays in the range
        delta += sunday_count

        # Only create the 'e' message when Sundays are present
        e = ""
        if sunday_count > 0:
            e = f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."

        if data is not None and not data.empty:
            # Check leave eligibility (whether user has enough balance)
            data = can_apply_leave(data)
            email = check_casual_leave_exceeded(email)
            # Check if the user is eligible to apply for leave
            if data["can_apply_leave"].iloc[0] == True:
                # Casual Leave case
                if selected_leave_type == "Casual Leave" and leave_status != "Sick Leave":
                    if leave_status != "Sick Leave" and today < from_date and employee_leave_rejection == False:
                        if delta <= data["lb"].iloc[0] and delta < 4:
                            if is_high_frequency: 
                                print(h)
                            if same_diff:
                                print(l)
                            if sunday_count > 0: 
                                print(e)  # Print only if Sundays exist
                            print("Leave Granted")
                            print("Detailed Feedback : Leave balance is sufficient")
                            print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")
                            ls= {
                                "Decision": "Accepted",
                                "Detailed Feedback": "Leave balance is sufficient",
                                "Final Balance" : f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."
                                }
                            if is_high_frequency: 
                                ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"]="You are applying the leaves of same frequency"
                            if sunday_count > 0: 
                                ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                            return ls
                        elif delta > data["lb"].iloc[0]:
                            if is_high_frequency: 
                                print(h)
                            if same_diff:
                                print(l)
                            if sunday_count > 0:
                                print(e)  # Print only if Sundays exist
                            
                            print("Requested leaves are exceeding the leave balance")
                            print(f"So it needs HR review & there will be LOP for {delta - data['lb'].iloc[0]} days")
                            print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")
                            ls={
                            "Note":"Requested leaves are exceeding the leave balance",
                            "Detailed Feedback":f"So it needs HR review & there will be LOP for {delta - data['lb'].iloc[0]} days",
                            "Final Balance":f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."
                            }
                            if is_high_frequency: 
                                ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"]="You are applying the leaves of same frequency"
                            if sunday_count > 0: 
                                ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                            return ls
                        elif delta > 3:
                            if is_high_frequency: 
                                print(h)
                            if same_diff:
                                print(l)
                            if sunday_count > 0: 
                                print(e)  # Print only if Sundays exist
                            print("As you are applying for more than 3 days, it needs HR review")
                            print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")
                            ls={
                                 "Request Pending" :"As you are applying for more than 3 days, it needs  HR review",
                                 "Final Balance":f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."
                            }
                            if is_high_frequency: 
                                ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"]="You are applying the leaves of same frequency"
                            if sunday_count > 0: 
                                ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                            return ls
                    elif employee_leave_rejection:
                        print(a)
                        return {
                            "Error" : "You are rejected more than 2 times and you are absent for more than 2 days in this current year."
                        }
                    elif today == from_date:
                        # Check if the employee has exceeded casual leave more than 2 times
                        if selected_leave_type == "Casual Leave" and check_casual_leave and employee_leave_rejection == False:
                            print(k)
                            return{
                                "Decision": "Rejected",
                                "Reason" : "You have already applied for Casual Leave more than 2 times where applied == from dates"
                            }
                        elif employee_leave_rejection:
                            print(a)
                            return {
                                "Error" : "You are rejected more than 2 times and you are absent for more than 2 days in this current year."
                            }
                        else:
                            if delta <= data["lb"].iloc[0] and delta < 4:
                                ls={
                                    "Decision": "Granted",
                                    "Detailed Feedback" : "Leave balance is sufficient"
                                }
                                if is_high_frequency: 
                                    print(h)
                                if same_diff:
                                    print(l)
                                if sunday_count > 0: 
                                    print(e)  # Print only if Sundays exist
                                print("Leave Granted")
                                if check_casual_leave_one:
                                    print(b)
                                    ls["Note"]="You have already applied for leave where applied == from . If you attempt to apply for leave again on any other day, your request will be rejected."
                                
                                print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")
                                ls["Final Decision"] = f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."
                                
                                if is_high_frequency: 
                                    ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                                if same_diff:
                                    ls["Please Check"]="You are applying the leaves of same frequency"
                                if sunday_count > 0: 
                                    ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                                return ls
                            elif delta > data["lb"].iloc[0]:
                                if is_high_frequency: 
                                    print(h)
                                if same_diff:
                                    print(l)
                                if sunday_count > 0: 
                                    print(e)  # Print only if Sundays exist
                                print("Requested leaves are exceeding the leave balance")
                                print(f"So it needs HR review & there will be LOP for {delta - data['lb'].iloc[0]} days")

                                ls["request pending"]="Requested leaves are exceeding the leave balance"
                                ls["Please Kindly"]=f" meet HR for review & there will be LOP for {delta - data['lb'].iloc[0]} days"

                                if check_casual_leave_one:
                                    print(b)
                                    ls["Note"]="You have already applied for leave where applied == from . If you attempt to apply for leave again on any other day, your request will be rejected."
                                print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")
                                
                                ls["Final Decision"] = f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."
                        
                                if is_high_frequency: 
                                    ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                                if same_diff:
                                    ls["Please Check"]="You are applying the leaves of same frequency"
                                if sunday_count > 0: 
                                    ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                                return ls
                            elif employee_leave_rejection:
                                print(a)
                                return {
                                "Error" : "You are rejected more than 2 times and you are absent for more than 2 days in this current year."
                                }
                            elif delta > 3:
                                if is_high_frequency: 
                                    print(h)
                                if same_diff:
                                    print(l)
                                if sunday_count > 0: 
                                    print(e)  # Print only if Sundays exist
                                print("As you are applying for more than 3 days, it needs HR review")
                                ls["Decision"]="Request Pending"
                                ls["Detailed Reason"]="As you are applying for more than 3 days, it needs HR review"
                                if check_casual_leave_one:
                                    print(b)
                                    ls["Note"]="You have already applied for leave where applied == from . If you attempt to apply for leave again on any other day, your request will be rejected."

                                print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")
                                ls["Total Balance"]=f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."


                                if is_high_frequency: 
                                    ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                                if same_diff:
                                    ls["Please Check"]="You are applying the leaves of same frequency"
                                if sunday_count > 0: 
                                    ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                                return ls
                    elif today > from_date:
                        print("Today's date should be less than the from date")
                        return {
                            "Warning":"Today's date should be less than the from date"
                        }
                
                # Sick Leave case
                elif selected_leave_type == "Sick Leave" and leave_status == "Sick Leave":
                    if today <= from_date and employee_leave_rejection == False:
                        if delta > data["lb"].iloc[0]:
                            if is_high_frequency: 
                                print(h)
                            if same_diff:
                                print(l)
                            if sunday_count > 0: 
                                print(e)  # Print only if Sundays exist
                            print("Submit medical certificates after coming to office as you requested leaves are more than your leave balance")
                            print(f"LOP for {delta - data['lb'].iloc[0]} days")
                            print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")
                            
                            ls["Important"]="Submit medical certificates after coming to office as you requested leaves are more than your leave balance"
                            ls["LOP for"]=f"{delta - data['lb'].iloc[0]} days"
                            ls["Final Balance"] = f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."
                        
                            if is_high_frequency: 
                                ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"]="You are applying the leaves of same frequency"
                            if sunday_count > 0: 
                                ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                            return ls
                        elif delta <= data["lb"].iloc[0] and delta < 4:
                            if is_high_frequency: 
                                print(h)
                            if same_diff:
                                print(l)
                            if sunday_count > 0: 
                                print(e)  # Print only if Sundays exist
                            print("Leave granted")
                            print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")
                            ls= {
                                "Decision": "Accepted",
                                "Final Balance" : f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."
                            }
                            if is_high_frequency: 
                                ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"]="You are applying the leaves of same frequency"
                            if sunday_count > 0: 
                                ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                            return ls
                        
                        elif delta <= data["lb"].iloc[0] and delta >= 4:
                            if is_high_frequency: 
                                print(h)
                            if same_diff:
                                print(l)
                            if sunday_count > 0: 
                                print(e)  # Print only if Sundays exist
                            print("Exceeding more than 3 days needs HR review. Submit the medical certificates after coming to office.")
                            print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")

                            ls["Note"]="Exceeding more than 3 days needs HR review. Submit the medical certificates after coming to office."
                            ls["Final Balance"] = f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."
                            if is_high_frequency: 
                                ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"]="You are applying the leaves of same frequency"
                            if sunday_count > 0: 
                                ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                            return ls
                    elif today > from_date and employee_leave_rejection == False:
                        if today >= to_date:
                            if is_high_frequency: 
                                print(h)
                            if same_diff:
                                print(l)
                            if sunday_count > 0: 
                                print(e)  # Print only if Sundays exist
                            print("Submit medical certificates")
                            ls["Important"]="Submit medical certificates"
                            if is_high_frequency: 
                                ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"]="You are applying the leaves of same frequency"
                            if sunday_count > 0: 
                                ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                            return ls
                        elif employee_leave_rejection:
                            print(a)
                            return {
                                "Error" : "You are rejected more than 2 times and you are absent for more than 2 days in this current year."
                            }
                        else:
                            if is_high_frequency: 
                                print(h)
                            if same_diff:
                                print(l)
                            if sunday_count > 0: 
                                print(e)  # Print only if Sundays exist
                            print("Submit medical certificates after coming")
                            print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")
                            
                            ls["Important"]="Submit medical certificates after coming"
                            ls["Final Balance"] = f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."
                        
                            if is_high_frequency: 
                                ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"]="You are applying the leaves of same frequency"
                            if sunday_count > 0: 
                                ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                            return ls
                    elif employee_leave_rejection:
                        print(a)
                        return {
                                "Error" : "You are rejected more than 2 times and you are absent for more than 2 days in this current year."
                        }
                else:
                    print("Leave type isn't matching with the leave status")
                    return {
                        "Error":"Leave type isn't matching with the leave status"
                    }
            
            elif data["can_apply_leave"].iloc[0] == False:
                if selected_leave_type == "Casual Leave" and leave_status != "Sick Leave" and employee_leave_rejection == False:
                    if leave_status != "Sick Leave" and today < from_date:
                        if delta >= 3:
                            if is_high_frequency: 
                                print(h)
                            if same_diff:
                                print(l)
                            if sunday_count > 0: 
                                print(e)  # Print only if Sundays exist
                            print("Approving percentage is less")
                            print(f"So it needs HR review & there will be LOP for {delta} days")
                            print("You can't apply the leave, it can be applied through only HR")
                            print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")

                            ls={
                                "Decision":"Pending",
                                "Detailed Review":f" Approving percentage is less So it needs HR review & there will be LOP for {delta} days",
                                "Caution":"You can't apply the leave, it can be applied through only HR"
                            }
                            ls["Final Balance"] = f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."
                        

                            if is_high_frequency: 
                                ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"]="You are applying the leaves of same frequency"
                            if sunday_count > 0: 
                                ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                            return ls
                        elif delta < 3:
                            if sunday_count > 0: 
                                print(e)  # Print only if Sundays exist
                            if same_diff:
                                print(l)
                            print("You have chance of getting leave approved and it depends on the HR")
                            print(f"So it needs HR review & there will be LOP for {delta} days")
                            print("You can't apply the leave, it can be applied through only HR")
                            print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")
                            ls={
                                "Decision":"Pending",
                                "Detailed Reason" : f"You have chance of getting leave approved and it depends on the HR So it needs HR review & there will be LOP for {delta} days ",
                                "Caution":"You can't apply the leave, it can be applied through only HR"
                                }
                            ls["Final Balance"] = f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."
                            if same_diff:
                                ls["Please Check"]="You are applying the leaves of same frequency"
                            if sunday_count > 0: 
                                ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                            return ls
                    elif today >= from_date:
                        print("You cant apply on the same date as cls leave type ")
                        return {
                            "Error":"You cant apply on the same date as cls leave type"
                        }
                elif selected_leave_type == "Sick Leave" and leave_status == "Sick Leave" and employee_leave_rejection == False:
                    if today <= from_date and today < to_date:
                        if is_high_frequency: 
                            print(h)
                        if same_diff:
                            print(l)
                        if sunday_count > 0: 
                            print(e)  # Print only if Sundays exist
                        print("Submit medical certificates after coming to office")
                        print(f"LOP for {delta} days because you don't have leave balance")
                        print("You can't apply the leave, it can be applied through only HR")
                        print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")
                        ls={
                                "Decision":"Pending",
                                "Detailed Reason" : f"Submit medical certificates after coming to office LOP for {delta} days because you don't have leave balance ",
                                "Caution":"You can't apply the leave, it can be applied through only HR"
                            }
                        ls["Final Balance"] = f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."

                        if is_high_frequency: 
                            ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                        if same_diff:
                            ls["Please Check"]="You are applying the leaves of same frequency"
                        if sunday_count > 0: 
                            ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                        return ls
                    elif today > from_date:
                        if today >= to_date:
                            if is_high_frequency: 
                                print(h)
                            if same_diff:
                                print(l)
                            if sunday_count > 0: 
                                print(e)  # Print only if Sundays exist
                            print("Submit medical certificates")
                            print(f"and your LOP will be {delta} days")
                            print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")
                            ls={
                                "Decision":"Pending",
                                "Detailed Reason" : f"Submit medical certificates and your LOP will be for {delta} days ",
                            }
                            ls["Final Balance"] = f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."

                            if is_high_frequency: 
                                ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"]="You are applying the leaves of same frequency"
                            if sunday_count > 0: 
                                ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                            return ls
                        else:
                            if is_high_frequency: 
                                print(h)
                            if same_diff:
                                print(l)
                            if sunday_count > 0: 
                                print(e)  # Print only if Sundays exist
                            
                            print("Submit medical certificates after coming")
                            print("You can't apply the leave, it can be applied through only HR")
                            print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")
                            ls={
                                "Decision":"Pending",
                                "Detailed Reason" : f"Submit medical certificates after coming.",
                                "Caution":"You can't apply the leave, it can be applied through only HR"
                            }
                            ls["Final Balance"] = f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}."

                            if is_high_frequency: 
                                ls["Warning"]="Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"]="You are applying the leaves of same frequency"
                            if sunday_count > 0: 
                                ls["Sunday Count"]=f"Sundays are included in the leave duration. {sunday_count} Sunday(s) were counted."
                            return ls
                            
                elif employee_leave_rejection:
                    print(a)
                    return {
                        "Error" : "You are rejected more than 2 times and you are absent for more than 2 days in this current year."
                    }
                else:
                    print("Leave type isn't matching with the leave status")
                    return {
                        "Error":"Leave type isn't matching with the leave status"
                    }
            
            else:
                print("You don't have enough leave balance")
                return {
                    "Reason":"You don't have enough leave balance"
                }
        else:
            print("No data found for that email")
            print("Unable to process leave request.")
            return {
                "Reason":"No data found for that email",
                "Process":"Unable to process leave Request"
            }

# Create widgets for the form
leave_type_input = widgets.Dropdown(
    options=['Casual Leave', 'Sick Leave'],
    description='Leave Type:',
    disabled=False
)
email_input = widgets.Text(description="Email:")

# Create date picker widgets for "From" and "To" inputs with min date set to today
today = date.today()
from_date_input = widgets.DatePicker(
    description="From Date",
    disabled=False,
    min=today  # Prevent selection of dates before today
)
to_date_input = widgets.DatePicker(
    description="To Date",
    disabled=False,
    min=today  # Prevent selection of dates before today
)

reason_input = widgets.Text(description="Reason:")

submit_button = widgets.Button(description="Fetch Data:")
output = widgets.Output()

# Attach event listener to the button

submit_button.on_click(on_submit)

# Display the form
display(widgets.VBox([ 
    leave_type_input, 
    email_input, 
    from_date_input, 
    to_date_input, 
    reason_input, 
    submit_button, 
    output 
]))



def fetch_leave_data_for_previous_month():
    try:
        # Establish the database connection
        conn = connect_to_db()
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT 
            l.empemail
        FROM  
            leaves l
        WHERE 
            MONTH(l.from) = MONTH(CURRENT_DATE)  -- Current month
            AND YEAR(l.from) = YEAR(CURRENT_DATE)  -- Current year
            AND (DATEDIFF(l.to, l.from) = 0 OR DATEDIFF(l.to, l.from) = 1)  -- Leave duration of 0 or 1 day
        GROUP BY 
            l.empemail  -- Group by employee email
        HAVING 
            COUNT(*) > 2  -- Employees who applied for leave more than twice in the same month
        """
        
        # Execute the query
        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()
        
        # If results are found, convert them into a DataFrame
        if result:
            df = pd.DataFrame(result)
            return df
        else:
            # If no employees satisfy the criteria, return a message or an empty DataFrame
            print("No employees found with repeated leave requests.")
            return None
    except mysql.connector.Error as err:
        # Handle database connection errors
        print(f"Error: {err}")
        return None
