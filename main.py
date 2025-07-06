# Import necessary libraries
import speech_recognition as sr
import pyttsx3
import re
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import sympy as sp

# --- INITIALIZATION ---
engine = pyttsx3.init()

# main.py

import json # Add this import at the top with your other imports

# --- API KEYS ---

# This new block of code opens and reads your secret config.json file
try:
    # 'with open...' safely opens the file
    with open('config.json') as config_file:
        # 'json.load' reads the data from the file
        config = json.load(config_file)
    
    # Get each key from the file and store it in a variable
    GEMINI_API_KEY = config.get('GEMINI_API_KEY')
    ALPHA_VANTAGE_API_KEY = config.get('ALPHA_VANTAGE_API_KEY')
    EXCHANGERATE_API_KEY = config.get('EXCHANGERATE_API_KEY')

# This part runs if the file is not found
except FileNotFoundError:
    print("Error: The 'config.json' file was not found. Please create it.")
    exit() # Stop the script if keys are missing

# ... the rest of your code continues from here as normal ...# Configure the Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

# --- GLOBAL STATE ---
last_result = None
last_equation_coeffs = None
variables = {}
HISTORY_FILE = 'calculation_history.json'

# --- DATA MAPPINGS ---
CURRENCY_MAP = {
    'dollars': 'USD', 'dollar': 'USD', '$': 'USD',
    'euros': 'EUR', 'euro': 'EUR', '€': 'EUR',
    'pounds': 'GBP', 'pound': 'GBP', '£': 'GBP',
    'taka': 'BDT', 'bd': 'BDT',
    'rupees': 'INR', 'rupee': 'INR',
    'yen': 'JPY',
    'yuan': 'CNY',
}

# --- CORE ASSISTANT FUNCTIONS ---
def speak(text):
    """Converts text to speech and prints to console."""
    print(f"Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    """Listens for voice commands and converts them to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nListening...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        try:
            command = r.recognize_google(audio)
            print(f"You said: {command}")
            return command.lower()
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            speak("Sorry, my speech service is down.")
            return None

# --- HISTORY AND VARIABLE MANAGEMENT ---
def save_to_history(entry):
    """Saves a calculation entry to the history file."""
    try:
        with open(HISTORY_FILE, 'r+') as f:
            history = json.load(f)
            history.append(entry)
            f.seek(0)
            json.dump(history, f, indent=4)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(HISTORY_FILE, 'w') as f:
            json.dump([entry], f, indent=4)

def read_history():
    """Reads and speaks the last 5 history entries."""
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
            if not history: return "Your history is empty."
            speak("Here are your last few calculations:")
            for entry in history[-5:]: speak(entry)
            return "End of recent history."
    except FileNotFoundError:
        return "You have no calculation history yet."

# --- TOOL FUNCTIONS ---
def _format_for_sympy(expr):
    """Helper function to format a string for sympy parsing."""
    # Add multiplication signs between numbers and variables (e.g., "3x" -> "3*x")
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    # Add power signs for sympy (e.g., "x^2" -> "x**2")
    expr = expr.replace('^', '**')
    return expr

def calculate(expression):
    """Evaluates a simple mathematical expression string."""
    global last_result
    try:
        result = eval(expression, {"__builtins__": None}, {})
        last_result = result
        history_entry = f"Calculation: {expression}, Result: {result}"
        save_to_history(history_entry)
        return f"The result is {result}"
    except Exception as e:
        return f"I couldn't calculate that. Error: {e}"

def solve_equation(a=0, b=0, c=0):
    """Solves a single-variable linear or quadratic equation."""
    global last_equation_coeffs
    a, b, c = float(a), float(b), float(c)
    if a != 0: # Quadratic
        last_equation_coeffs = [a, b, c]
        discriminant = (b**2) - 4*(a*c)
        if discriminant >= 0:
            sol1 = (-b - np.sqrt(discriminant)) / (2*a)
            sol2 = (-b + np.sqrt(discriminant)) / (2*a)
            return f"The solutions are: {sol1:.2f} and {sol2:.2f}"
        else:
            return "The solutions are complex numbers."
    elif b != 0: # Linear
        last_equation_coeffs = [b, c]
        return f"The solution is: x = {-c / b:.2f}"
    return "This doesn't seem to be a valid single-variable equation."

def solve_system_of_equations(eq1, eq2):
    """Solves a system of two linear equations."""
    try:
        x, y = sp.symbols('x y')
        eq1_formatted = _format_for_sympy(eq1)
        eq2_formatted = _format_for_sympy(eq2)
        
        lhs1, rhs1 = map(sp.sympify, eq1_formatted.split('='))
        lhs2, rhs2 = map(sp.sympify, eq2_formatted.split('='))
        
        solution = sp.solve([sp.Eq(lhs1, rhs1), sp.Eq(lhs2, rhs2)], (x, y))
        if solution and isinstance(solution, dict):
            return f"The solution is: x = {solution.get(x, 'N/A')}, y = {solution.get(y, 'N/A')}"
        else:
            return "No unique solution found for the system."
    except Exception as e:
        return f"I couldn't solve that system. Please check the equations. Error: {e}"

def calculate_integral(expression, variable='x'):
    """Calculates the indefinite integral of an expression."""
    try:
        var = sp.symbols(variable)
        formatted_expr = _format_for_sympy(expression)
        expr = sp.sympify(formatted_expr)
        integral = sp.integrate(expr, var)
        return f"The integral of {expression} is {integral} + C"
    except Exception as e:
        return f"I couldn't perform the integration. Error: {e}"

def calculate_derivative(expression, variable='x'):
    """Calculates the derivative of an expression."""
    try:
        var = sp.symbols(variable)
        formatted_expr = _format_for_sympy(expression)
        expr = sp.sympify(formatted_expr)
        derivative = sp.diff(expr, var)
        return f"The derivative of {expression} is {derivative}"
    except Exception as e:
        return f"I couldn't perform the differentiation. Error: {e}"

def convert_currency(amount, from_currency, to_currency):
    """Converts an amount from one currency to another using a reliable API."""
    if EXCHANGERATE_API_KEY == 'YOUR_EXCHANGERATE_API_KEY':
        return "Currency conversion API key is not configured."
    from_currency = CURRENCY_MAP.get(from_currency.lower(), from_currency).upper()
    to_currency = CURRENCY_MAP.get(to_currency.lower(), to_currency).upper()
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGERATE_API_KEY}/pair/{from_currency}/{to_currency}/{amount}"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("result") == "success":
            converted_amount = data.get("conversion_result")
            return f"{amount} {from_currency} is equal to {converted_amount:.2f} {to_currency}."
        else:
            return f"Sorry, I couldn't get the exchange rate. Reason: {data.get('error-type', 'unknown error')}"
    except Exception as e:
        return f"An error occurred during currency conversion: {e}"

def calculate_loan_payment(principal, years, rate):
    """Calculates monthly loan payments."""
    p, t, r = float(principal), float(years), float(rate)
    r_monthly = (r / 100) / 12
    n_payments = t * 12
    if r_monthly == 0: return "Interest rate cannot be zero."
    payment = p * (r_monthly * (1 + r_monthly)**n_payments) / ((1 + r_monthly)**n_payments - 1)
    return f"The monthly payment is {payment:.2f}."

def get_stock_price(ticker):
    """Fetches the stock price for a given ticker."""
    if ALPHA_VANTAGE_API_KEY == 'YOUR_API_KEY':
        return "Alpha Vantage API key is not configured."
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}'
    try:
        data = requests.get(url).json()
        price = data.get('Global Quote', {}).get('05. price')
        if price: return f"The current price of {ticker} is ${float(price):.2f}."
        return f"Could not find the stock price for {ticker}."
    except Exception:
        return "Failed to retrieve stock data."

def plot_last_equation():
    """Generates and displays a graph of the last solved equation."""
    if not last_equation_coeffs: return "Solve an equation first."
    coeffs = last_equation_coeffs
    x = np.linspace(-10, 10, 400)
    y = np.polyval(coeffs, x)
    title = f'Graph of {np.poly1d(coeffs)}'
    try:
        plt.figure()
        plt.plot(x, y)
        plt.title(title), plt.xlabel('x'), plt.ylabel('y'), plt.grid(True)
        plt.axhline(0, color='black', lw=0.5), plt.axvline(0, color='black', lw=0.5)
        plt.show(block=False), plt.pause(1)
        return "Here is the graph."
    except Exception as e:
        return f"Could not display graph: {e}"

def close_graphs():
    """Closes all open matplotlib graph windows."""
    plt.close('all')
    return "Graph windows closed."

# --- LANGUAGE MODEL INTERACTION ---
def get_intent(command):
    """Uses a language model to determine the user's intent and extract parameters."""
    if not model:
        return {"error": "Language model not initialized. Check your API key."}

    # System prompt to define the capabilities and desired JSON output format.
    system_prompt = """
    You are an expert at understanding a user's mathematical or financial command and determining which function to call and what parameters to use.
    Your only output should be a single, valid JSON object. Do not add any other text or explanations.

    Available functions:
    - calculate(expression: str): For simple math like "5*10" or "100/4".
    - solve_equation(a: float, b: float, c: float): For SINGLE-VARIABLE equations like "2x^2 + 3x - 5=0". For "4x - 8=0", a=0, b=4, c=-8.
    - solve_system_of_equations(eq1: str, eq2: str): For systems of two linear equations. The equations must be provided as strings.
    - calculate_integral(expression: str, variable: str): For indefinite integrals.
    - calculate_derivative(expression: str, variable: str): For derivatives.
    - convert_currency(amount: float, from_currency: str, to_currency: str): For currency conversions.
    - calculate_loan_payment(principal: float, years: float, rate: float): For loan calculations.
    - get_stock_price(ticker: str): For stock prices.
    - plot_last_equation(): To graph the previously solved single-variable equation.
    - close_graphs(): To close graph windows.
    - read_history(): To read calculation history.
    - general_conversation(response: str): For greetings or simple questions.
    - exit(): To quit the program.

    Examples:
    User: "what is 5 times 12" -> {"function": "calculate", "params": {"expression": "5 * 12"}}
    User: "solve 3x squared plus 2x minus 8" -> {"function": "solve_equation", "params": {"a": 3, "b": 2, "c": -8}}
    User: "solve the system 3*y + 15 = 0 and 7*y + 3*x + 9 = 0" -> {"function": "solve_system_of_equations", "params": {"eq1": "3*y + 15 = 0", "eq2": "7*y + 3*x + 9 = 0"}}
    User: "what's the integration of 3*x**2" -> {"function": "calculate_integral", "params": {"expression": "3*x**2", "variable": "x"}}
    User: "derivative of 6x square" -> {"function": "calculate_derivative", "params": {"expression": "6*x**2", "variable": "x"}}
    User: "what's the dollar to taka rate" -> {"function": "convert_currency", "params": {"amount": 1, "from_currency": "USD", "to_currency": "BDT"}}
    User: "what's up" -> {"function": "general_conversation", "params": {"response": "Not much, I'm ready for your command!"}}
    User: "goodbye" -> {"function": "exit", "params": {}}
    """
    
    try:
        response = model.generate_content(system_prompt + "\nUser: \"" + command + "\"")
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_text)
    except Exception as e:
        print(f"Gemini parsing error: {e}")
        return {"function": "general_conversation", "params": {"response": "Sorry, I had a little trouble understanding that."}}

# --- MAIN APPLICATION LOOP ---
def main():
    """Main function to run the voice assistant."""
    speak("Hello! Your Gemini-powered financial assistant is online.")
    
    # Maps function names from the AI model to the actual Python functions.
    function_map = {
        "calculate": calculate,
        "solve_equation": solve_equation,
        "solve_system_of_equations": solve_system_of_equations,
        "calculate_integral": calculate_integral,
        "calculate_derivative": calculate_derivative,
        "convert_currency": convert_currency,
        "calculate_loan_payment": calculate_loan_payment,
        "get_stock_price": get_stock_price,
        "plot_last_equation": plot_last_equation,
        "close_graphs": close_graphs,
        "read_history": read_history,
    }

    while True:
        command = listen()
        if command:
            intent = get_intent(command)
            
            if "error" in intent:
                speak(intent["error"])
                continue

            func_name = intent.get("function")
            params = intent.get("params", {})
            
            if func_name in function_map:
                func_to_call = function_map[func_name]
                # The ** operator unpacks the dictionary into keyword arguments.
                response = func_to_call(**params)
                speak(response)
            elif func_name == "general_conversation":
                speak(params.get("response", "I'm not sure how to respond to that."))
            elif func_name == "exit":
                speak("Goodbye! Have a great day.")
                break
            else:
                speak("I'm not sure what you mean. Please try again.")

if __name__ == "__main__":
    main()
