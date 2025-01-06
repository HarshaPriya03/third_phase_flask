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
import calendar
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
df = pd.read_csv(r"C:\HarshaPriya\ML\leave_app\leave_app\type_of_leave.csv")  # Adjust path to your dataset

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
            AND DATE(applied) = DATE(from)
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
                    from, 
                    to
                FROM leaves
                WHERE empemail = %s AND YEAR(from) = %s AND MONTH(from) = %s
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

# def rejected_and_absent(empemail):
#     try:
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
#         AND YEAR(AttendanceTime) = YEAR(CURRENT_DATE);
#         """
#         cursor.execute(query_absent, (empemail,))
#         absent_result = cursor.fetchone()

#         # If the count of absences in the current year is greater than 2, return True
#         if absent_result['absent_count'] > 2:
#             return True
#         return False

#     except mysql.connector.Error as err:
#         print(f"Database error: {err}")
#         return False


def rejected_and_absent(empemail):
    try:
        # Connect to the database
        conn = connect_to_db()
        cursor = conn.cursor(dictionary=True)

        # Get the current year and the date range from March 1st this year to March 1st next year
        current_year = datetime.now().year
        start_date = datetime(current_year, 3, 1).date()  # March 1 of the current year
        end_date = datetime(current_year + 1, 3, 1).date()  # March 1 of the next year

        # First, check if the status2 count is greater than 2 in the leaves table within the date range
        query_leaves = """
        SELECT COUNT(*) AS status2_count
        FROM leaves
        WHERE empemail = %s
        AND status = 2
        AND DATE(from) >= %s  -- Leave date from this year March 1st onwards
        AND DATE(from) < %s  -- Leave date until next year March 1st
        """
        cursor.execute(query_leaves, (empemail, start_date, end_date))
        status2_result = cursor.fetchone()

        # Debugging: Print the status2_count
        # print(f"Status2 Count: {status2_result['status2_count']}")

        # If the count of status2 is not greater than or equal to 3, return False
        if status2_result['status2_count'] < 3 :
            return False

        # If status2 count >= 3, proceed to check the absent table within the same date range
        query_absent = """
        SELECT COUNT(*) AS absent_count
        FROM absent
        WHERE empname = (SELECT empname FROM leaves WHERE empemail = %s LIMIT 1)
        AND DATE(AttendanceTime) >= %s  -- Absence after March 1st this year
        AND DATE(AttendanceTime) < %s  -- Absence until March 1st next year
        """
        cursor.execute(query_absent, (empemail, start_date, end_date))
        absent_result = cursor.fetchone()

        # Debugging: Print the absent_count
        # print(f"Absent Count: {absent_result['absent_count']}")

        # If the count of absences in the specified date range is greater than 2, return True
        if absent_result['absent_count'] > 2:
            return True

        return False

    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return False

    
def check_casual_leave_exceeded_one(email):
    try:
        conn = connect_to_db()
        cursor = conn.cursor(dictionary=True)
        
        # Query to count the casual leave applications for the given email
        query = """
            SELECT empemail, COUNT(*) AS record_count
            FROM leaves
            WHERE leavetype = 'CASUAL LEAVE'
            AND DATE(applied) = DATE(from)
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
    


def manager(empemail):
    try:
        conn=connect_to_db()
        cursor=conn.cursor(dictionary=True)
        query="""
        SELECT
            m.email
        FROM
            manager m
        WHERE
            m.email = %s
        
         """
        cursor.execute(query,(empemail,))
        result=cursor.fetchall()
        conn.close()
    
        if result:
           return True
        return False


    except mysql.connector.Error as err:
        print(f"Database error: {err}") 
        return False



def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="leave_data"
    )

def salary(empemail):
    try:
        conn = connect_to_db()
        cursor = conn.cursor(dictionary=True)
        query= """
        SELECT empname 
        FROM leaves 
        WHERE empemail = %s LIMIT 1
        """
        cursor.execute(query, (empemail,))
        result = cursor.fetchone() 

        if result:
            empname = result['empname']
            query_ctc = """
            SELECT ctc
            FROM payroll_msalarystruc
            WHERE empname = %s LIMIT 1
            """
            cursor.execute(query_ctc, (empname,))
            ctc_result = cursor.fetchone()

            if ctc_result:
                return ctc_result['ctc'] 
            else:
                print("No ctc data found for the employee.")
                return False
        else:
            print(f"No employee found with the email: {empemail}")
            return False

    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return False
    

def calculate_lop(ctc, delta):
    today = date.today()
    current_month_days = calendar.monthrange(today.year, today.month)[1]  
    per_day_pay = int(ctc) / int(current_month_days)  
    return per_day_pay


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
                   "Feedback": "The leave request has been declined as it falls under a personal problem/issue category.",
                   "Final Decision": "Leave cannot be approved"
            }
            return ls
        leave_status = predict_sick_leave(leave_reason, log_reg_model, vectorizer)  # Assume model and vectorizer are available

        if not from_date or not to_date:
            print("Decision : Rejected \n Detailed Feedback : Please select both 'From Date' and 'To Date' \n Final Decision : Incomplete data provided.")
            ls= {
                "Decision": "Rejected",
                "Feedback": "Both 'From Date' and 'To Date' must be provided.",
                "Final Decision": "Incomplete data provided."
            }
            return ls

        elif from_date > to_date:
            print("'Decision: Rejected \n Detailed feedback: From Date' cannot be later than 'To Date'. Please correct the dates. \n Final Decision: Invalid date range.")
            ls= {
                  "Decision": "Rejected",
                    "Feedback": "The 'From Date' cannot be later than the 'To Date'. Please correct the date range.",
                    "Final Decision": "Invalid date range"

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

        manager_leave=manager(email)
        m="Your leave can be approved only by HR." if manager_leave else ""

        # Extend the date range and check for Sundays
        delta = (to_date - from_date).days + 1  # Number of days for the leave
        extended_weekdays = get_weekdays(from_date, to_date)
        sunday_count = extended_weekdays[extended_weekdays['Weekday'] == 'Sunday'].shape[0]

        # Adjust delta by adding the number of Sundays in the range
        delta += sunday_count
        
        #LOP
        ctc = salary(email)
        if not ctc:
            print("Failed to calculate LOP: Unable to fetch employee's CTC.")
            return
        per_day_pay = calculate_lop(ctc, delta)


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
                if selected_leave_type == "Casual Leave" and leave_status != "Sick Leave" and selected_leave_type != "Comp Off":
                    if leave_status != "Sick Leave" and today < from_date and selected_leave_type != "Comp Off":
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
                                "Feedback": "Your leave request has been approved as the available balance is sufficient.",
                                "Final Balance": f"Your final leave balance, after deducting the requested leave days, is {data['lb'].iloc[0] - delta}."
                                }
                            if is_high_frequency:
                                ls["Warning"] = "Your leave frequency appears to be high. You have already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"] = "You are applying for leave at the same frequency. Kindly review your request."
                            if sunday_count > 0:
                                ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                            if employee_leave_rejection:
                                ls["Alert"] = "A pattern of leave requests with a high likelihood of rejection and absenteeism has been detected. This may reduce the chances of approval, and HR intervention may be necessary to resolve this matter."
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
                            ls = {
                                "Note": "The requested leave duration exceeds your available leave balance.",
                                "Detailed Feedback": f"HR review is required, and there will be a Leave Without Pay (LOP) for {delta - data['lb'].iloc[0]} day(s).",
                                "Final Balance": f"Your final leave balance, after deducting the requested leave, is {data['lb'].iloc[0] - delta}.",
                                "LOP Deducted" : f"Thus, ₹{(delta - data['lb'].iloc[0])*per_day_pay:.2f} would be deducted from  your salary as LOP."
                            }
                            
                            if is_high_frequency:
                                ls["Warning"] = "Your leave frequency appears to be high, having already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"] = "You are applying for leave at the same frequency. Please review your request."
                            if sunday_count > 0:
                                ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                            if employee_leave_rejection:
                                ls["Alert"] = "A pattern of leave requests with a high likelihood of rejection and absenteeism has been detected. This may reduce the likelihood of approval, and HR intervention is necessary to address this issue."
                                
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
                            ls = {
                                "Request Pending": "Since your leave request exceeds 3 days, it requires HR review.",
                                "Final Balance": f"Your final leave balance, after deducting the requested leave, is {data['lb'].iloc[0] - delta}."
                            }
                            
                            if is_high_frequency:
                                ls["Warning"] = "Your leave frequency is high, having already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"] = "You are applying for leave at the same frequency. Kindly review your request."
                            if sunday_count > 0:
                                ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                            if employee_leave_rejection:
                                ls["Alert"] = "A pattern of leave requests with a high likelihood of rejection and absenteeism has been detected. This may reduce the chances of approval, and HR intervention is required to address this issue."
                            
                            return ls

                    elif today == from_date:
                        # Check if the employee has exceeded casual leave more than 2 times
                        if selected_leave_type == "Casual Leave" and check_casual_leave:
                            print(k)
                            return {
                                "Decision": "Rejected",
                                "Reason": "Your leave request has been declined as you have already applied for Casual Leave more than twice with the same 'From Date' for previous requests."
                            }

                        
                        else:
                            if delta <= data["lb"].iloc[0] and delta < 4:
                                ls = {
                                    "Decision": "Granted",
                                    "Detailed Feedback": "Your leave request has been approved as the leave balance is sufficient."
                                }
                                
                                if is_high_frequency:
                                    print(h)  # Assuming 'h' contains a relevant message
                                if same_diff:
                                    print(l)  # Assuming 'l' contains a relevant message
                                if sunday_count > 0:
                                    print(e)  # Print only if Sundays are included in the leave duration
                                
                                print("Leave Granted")
                                
                                if check_casual_leave_one:
                                    print(b)  # Assuming 'b' contains a relevant message
                                    ls["Note"] = "You have previously applied for leave with the same 'From Date'. Any future requests for leave on the same day will be rejected."
                                
                                print(f"Your final leave balance, after deducting the requested leave, is {data['lb'].iloc[0] - delta}.")
                                ls["Final Decision"] = f"Your final leave balance, after deducting the requested leave, is {data['lb'].iloc[0] - delta}."
                                
                                if is_high_frequency:
                                    ls["Warning"] = "Your leave frequency is high, having already taken more than 6 days of leave this month."
                                if same_diff:
                                    ls["Please Check"] = "You are applying for leave at the same frequency. Kindly review your request."
                                if sunday_count > 0:
                                    ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                                if employee_leave_rejection:
                                    ls["Alert"] = "A pattern of leave requests with a high likelihood of rejection and absenteeism has been detected. This may reduce the chances of approval, and HR intervention is necessary."
                                
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
                                ls = {
                                    "Request Pending": "The requested leave duration exceeds your available leave balance.",
                                    "Action Required": f"Please meet with HR for review. Additionally, there will be Leave Without Pay (LOP) for {delta - data['lb'].iloc[0]} day(s).",
                                    "LOP Deducted" : f"Thus, ₹{(delta - data['lb'].iloc[0])*per_day_pay:.2f} would be deducted from  your salary as LOP."
                                }
                                
                                if check_casual_leave_one:
                                    print(b)  # Assuming 'b' contains a relevant message
                                    ls["Note"] = "You have previously applied for leave with the same 'From Date'. Future leave requests for the same day will be rejected."
                                
                                print(f"Your final leave balance, after deducting the requested leave, is {data['lb'].iloc[0] - delta}.")
                                ls["Final Decision"] = f"Your final leave balance, after deducting the requested leave, is {data['lb'].iloc[0] - delta}."
                                
                                if is_high_frequency:
                                    ls["Warning"] = "Your leave frequency is high, as you have already taken more than 6 days of leave this month."
                                if same_diff:
                                    ls["Please Check"] = "You are applying for leave at the same frequency. Kindly review your request."
                                if sunday_count > 0:
                                    ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                                if employee_leave_rejection:
                                    ls["Alert"] = "A pattern of leave requests with a high likelihood of rejection and absenteeism has been detected. This may reduce the chances of approval, and HR intervention is necessary to address the issue."
                                
                                return ls

                            elif delta > 3:
                                if is_high_frequency: 
                                    print(h)
                                if same_diff:
                                    print(l)
                                if sunday_count > 0: 
                                    print(e)  # Print only if Sundays exist
                                print("As you are applying for more than 3 days, it needs HR review")
                                ls = {
                                    "Decision": "Request Pending",
                                    "Detailed Reason": "As your leave request exceeds 3 days, it requires HR review."
                                }
                                
                                if check_casual_leave_one:
                                    print(b)  # Assuming 'b' contains a relevant message
                                    ls["Note"] = "You have previously applied for leave with the same 'From Date'. Future leave requests for the same day will be rejected."
                                
                                print(f"Your final leave balance, after deducting the requested leave, is {data['lb'].iloc[0] - delta}.")
                                ls["Total Balance"] = f"Your final leave balance, after deducting the requested leave, is {data['lb'].iloc[0] - delta}."
                                
                                if is_high_frequency:
                                    ls["Warning"] = "Your leave frequency is high. You have already taken more than 6 days of leave this month."
                                if same_diff:
                                    ls["Please Check"] = "You are applying for leave at the same frequency. Please review your request."
                                if sunday_count > 0:
                                    ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                                if employee_leave_rejection:
                                    ls["Alert"] = "A pattern of leave requests with a high likelihood of rejection and absenteeism has been detected. This may reduce the chances of approval, and HR intervention is necessary to address the issue."
                                
                                return ls

                    elif today > from_date:
                        print("Today's date should be less than the from date")
                        return {
                            "Warning": "The selected 'From Date' must be greater than today's date."
                        }
                
                # Sick Leave case
                elif selected_leave_type == "Sick Leave" and leave_status == "Sick Leave":
                    if today <= from_date:
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
                            
                            ls = {
                                "Important": "Please submit medical certificates upon returning to the office, as your requested leave exceeds your available balance.",
                                "LOP Days": f"{delta - data['lb'].iloc[0]} day(s) will be treated as Leave Without Pay (LOP).",
                                "Final Balance": f"Your final leave balance, after deducting the requested leave, is {data['lb'].iloc[0] - delta}.",
                                "LOP Deducted" : f"Thus, ₹{(delta - data['lb'].iloc[0])*per_day_pay:.2f} would be deducted from  your salary as LOP."
                            }
                            
                            if is_high_frequency:
                                ls["Warning"] = "Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            
                            if same_diff:
                                ls["Please Check"] = "You are applying for leave with the same frequency. Please review your request."
                            
                            if sunday_count > 0:
                                ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                            
                            if employee_leave_rejection:
                                ls["Alert"] = "A pattern of leave requests with a high likelihood of rejection and absenteeism has been detected. This may reduce the chances of approval, and HR intervention is necessary to address the issue."
                            
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
                            ls = {
                                "Decision": "Accepted",
                                "Final Balance": f"Your final leave balance, after deducting the requested leave, is {data['lb'].iloc[0] - delta}."
                            }
                            
                            if is_high_frequency:
                                ls["Warning"] = "Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            
                            if same_diff:
                                ls["Please Check"] = "You are applying for leave with the same frequency as before. Please review your request."
                            
                            if sunday_count > 0:
                                ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                            
                            if employee_leave_rejection:
                                ls["Alert"] = "A pattern of leave requests with a high likelihood of rejection and absenteeism has been detected. This may reduce the chances of approval, and HR intervention is necessary to address the issue."
                            
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
                            ls = {
                                "Note": "Leave requests exceeding 3 days require HR review. Please submit medical certificates upon returning to the office.",
                                "Final Balance": f"Your final leave balance, after deducting the requested leave, is {data['lb'].iloc[0] - delta}."
                            }
                            
                            if is_high_frequency:
                                ls["Warning"] = "Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            
                            if same_diff:
                                ls["Please Check"] = "Your leave request matches the frequency of a previous request. Please review it carefully."
                            
                            if sunday_count > 0:
                                ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                            
                            if employee_leave_rejection:
                                ls["Alert"] = "A pattern of leave requests with a high likelihood of rejection and absenteeism has been detected. This may reduce the chances of approval, and HR intervention is necessary to address the issue."
                            
                            return ls

                            
                    elif today > from_date :
                        if today >= to_date:
                            if is_high_frequency: 
                                print(h)
                            if same_diff:
                                print(l)
                            if sunday_count > 0: 
                                print(e)  # Print only if Sundays exist
                            print("Submit medical certificates")
                            ls = {
                                "Important": "Please submit medical certificates as requested leave exceeds your available balance."
                            }
                            
                            if is_high_frequency:
                                ls["Warning"] = "Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            
                            if same_diff:
                                ls["Please Check"] = "You are applying for leave at the same frequency as previous requests. Please review your request."
                            
                            if sunday_count > 0:
                                ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                            
                            return ls

                        
                        else:
                            if is_high_frequency: 
                                print(h)
                            if same_diff:
                                print(l)
                            if sunday_count > 0: 
                                print(e)  # Print only if Sundays exist
                            print("Submit medical certificates after coming")
                            print(f"This is your final leave balance after deducting your requested leaves {data['lb'].iloc[0] - delta}.")
                            ls = {
                                "Important": "Please submit medical certificates upon your return to the office.",
                                "Final Balance": f"Your final leave balance, after deducting the requested leave, is {data['lb'].iloc[0] - delta}."
                            }
                            
                            if is_high_frequency:
                                ls["Warning"] = "Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            
                            if same_diff:
                                ls["Please Check"] = "You are applying for leave at the same frequency as previous requests. Please review your request carefully."
                            
                            if sunday_count > 0:
                                ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                            
                            if employee_leave_rejection:
                                ls["Alert"] = "A pattern of leave requests with a high likelihood of rejection and absenteeism has been detected. This may reduce the chances of approval, and HR intervention is necessary to address the issue."
                            
                            return ls

                elif selected_leave_type == "Comp Off":
                    comp_off_balance = pd.to_numeric(data["co"].iloc[0], errors='coerce')
                    if comp_off_balance > 0:
                        ls["Comp Off"] = "Your leave request will be processed as you have a remaining comp off balance."
                    else:
                        ls["Comp Off"] = "You are not eligible to apply for comp off, as you have no remaining balance."
                    
                    return ls

                    
                else:
                    print("Leave type isn't matching with the leave status")
                    return {
                        "Error": "The leave type does not match the leave status."
                    }

            
            elif data["can_apply_leave"].iloc[0] == False:
                if selected_leave_type == "Casual Leave" and leave_status != "Sick Leave" :
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
                            ls = {
                                "Decision": "Pending",
                                "Detailed Review": f"Approval percentage is insufficient. HR review is required, and there will be a LOP for {delta} day(s).",
                                "Caution": "Leave cannot be applied directly. It must go through HR for processing.",
                                "LOP Deducted": f"Thus, ₹{(delta)*per_day_pay:.2f} would be deducted from your salary as LOP"
                            }
                            
                            ls["Final Balance"] = f"Your final leave balance, after deducting the requested leave, is {data['lb'].iloc[0] - delta}."
                            
                            if is_high_frequency:
                                ls["Warning"] = "Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            
                            if same_diff:
                                ls["Please Check"] = "You are applying for leave at the same frequency as previous requests. Please review your request carefully."
                            
                            if sunday_count > 0:
                                ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                            
                            if employee_leave_rejection:
                                ls["Alert"] = "A pattern of leave requests with a high likelihood of rejection and absenteeism has been detected. This may reduce the chances of approval, and HR intervention is necessary to address the issue."
                            
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
                            ls = {
                                "Decision": "Pending",
                                "Detailed Reason": f"Your leave approval is contingent upon HR review. There will be a LOP for {delta} day(s) pending approval.",
                                "Caution": "Leave cannot be applied directly. It must be processed through HR.",
                                "LOP Deducted": f"Thus, ₹{(delta)*per_day_pay:.2f} would be deducted from your salary as LOP"
                            }
                            
                            ls["Final Balance"] = f"Your final leave balance, after deducting the requested leave, is {data['lb'].iloc[0] - delta}."
                            
                            if same_diff:
                                ls["Please Check"] = "You are applying for leave at the same frequency as previous requests. Please review your request carefully."
                            
                            if sunday_count > 0:
                                ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                            
                            if employee_leave_rejection:
                                ls["Alert"] = "A pattern of leave requests with a high likelihood of rejection and absenteeism has been detected. This may reduce the chances of approval, and HR intervention is necessary to address the issue."
                            
                            return ls

                    elif today >= from_date:
                        print("You cant apply on the same date as cls leave type ")
                        return {
                            "Error": "Leave cannot be applied on the same date as the 'CLS' leave type."
                        }

                elif selected_leave_type == "Sick Leave" and leave_status == "Sick Leave":
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
                        ls = {
                            "Decision": "Pending",
                            "Detailed Reason": f"Medical certificates must be submitted after returning to the office. There will be LOP for {delta} days as you currently do not have sufficient leave balance.",
                            "Caution": "Leave applications must be processed through HR due to the insufficient leave balance.",
                            "LOP Deducted": f"Thus, ₹{(delta)*per_day_pay:.2f} would be deducted from your salary as LOP"

                        }
                        
                        ls["Final Balance"] = f"Your final leave balance, after deducting the requested leaves, is {data['lb'].iloc[0] - delta}."
                        
                        if is_high_frequency: 
                            ls["Warning"] = "Your leave frequency is high. You have already taken more than 6 days of leave this month."
                        if same_diff:
                            ls["Please Check"] = "You are applying for leaves of the same frequency. Please review."
                        if sunday_count > 0: 
                            ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                        if employee_leave_rejection:
                            ls["Alert"] = "A high frequency of leave requests with a likelihood of rejection and absenteeism has been identified. HR intervention may be required to address this issue."
                        
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
                            ls = {
                                "Decision": "Pending",
                                "Detailed Reason": f"Submit medical certificates. Your LOP will be for {delta} days due to insufficient leave balance.",
                                "LOP Deducted": f"Thus, ₹{(delta)*per_day_pay:.2f} would be deducted from your salary as LOP"

                            }
                            
                            ls["Final Balance"] = f"Your final leave balance, after deducting the requested leaves, is {data['lb'].iloc[0] - delta}."
                            
                            if is_high_frequency: 
                                ls["Warning"] = "Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"] = "You are applying for leaves of the same frequency. Please review your request."
                            if sunday_count > 0: 
                                ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                            
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
                            
                            ls = {
                                "Decision": "Pending",
                                "Detailed Reason": "Submit medical certificates after coming to the office.",
                                "Caution": "You cannot apply for this leave directly; it must be processed through HR.",
                                "LOP Deducted": f"Thus, ₹{(delta)*per_day_pay:.2f} would be deducted from your salary as LOP"
                            }
                            
                            ls["Final Balance"] = f"Your final leave balance, after deducting the requested leaves, is {data['lb'].iloc[0] - delta}."
                            
                            if is_high_frequency: 
                                ls["Warning"] = "Your leave frequency is high. You have already taken more than 6 days of leave this month."
                            if same_diff:
                                ls["Please Check"] = "You are applying for leaves of the same frequency. Please review your request."
                            if sunday_count > 0: 
                                ls["Sunday Count"] = f"Sundays are included in the leave duration. A total of {sunday_count} Sunday(s) were counted."
                            if employee_leave_rejection:
                                ls["Alert"] = "The frequency of leave requests suggests a high likelihood of rejection and absences. HR intervention is required."
                            
                            return ls

                elif selected_leave_type == "Comp Off":
                    comp_off_balance = pd.to_numeric(data["co"].iloc[0], errors='coerce')
                    if comp_off_balance > 0:
                        ls["Comp Off"] = "Your leave request is being processed."
                    else:
                        ls["Comp Off"] = "You are not eligible to apply for Comp Off due to insufficient remaining balance."

                    
                    return ls
           
                else:
                    print("Leave type isn't matching with the leave status")
                    return {
                        "Error": "The leave type does not align with the current leave status"
                    }

            
            else:
                print("You don't have enough leave balance")
                return {
                    "Reason":"Insufficient leave balance to process the request."
                }
        else:
            print("No data found for that email")
            print("Unable to process leave request.")
            return {
                "Reason": "No records found associated with the provided email address.",
                "Process": "Unable to process the leave request due to the absence of relevant data."
            }


# Create widgets for the form
leave_type_input = widgets.Dropdown(
    options=['Casual Leave', 'Sick Leave','Comp Off'],
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
            AND (DATEDIFF(l.to, l.from) = 0 OR DA  TEDIFF(l.to, l.from) = 1)  -- Leave duration of 0 or 1 day
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