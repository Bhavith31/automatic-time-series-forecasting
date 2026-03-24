# ============================================================
# views.py - Main Web App Logic
# ============================================================
# This file handles ALL pages and forecasting logic.
#
# UPGRADED: Linear Regression → Prophet (by Facebook)
#
# WHY PROPHET IS BETTER THAN LINEAR REGRESSION:
# Linear Regression:
#   → Could only draw a straight line
#   → No ups and downs in forecast
#   → Ignored weekly and monthly patterns
#
# Prophet:
#   → Shows realistic ups and downs 
#   → Learns weekly patterns (Mon high, Sun low) 
#   → Learns monthly patterns 
#   → Gives honest confidence bands 
#   → Designed specifically for business time series 
#   → Still easy to use 
# ============================================================


# ── IMPORTING LIBRARIES ───────────────────────────────────

from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse, FileResponse
import pandas as pd
import numpy as np
import os, uuid, json
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .models import ForecastHistory
from django.conf import settings

# ── PROPHET IMPORT ────────────────────────────────────────
from prophet import Prophet
# Prophet is Facebook's open source forecasting tool.
# It was built specifically for business time series data
# like sales, revenue, website traffic etc.
#
# Prophet automatically learns:
# → Overall trend (going up or down?)
# → Weekly seasonality (which day of week has highest sales?)
# → Yearly seasonality (which month of year has highest sales?)
# → Holiday effects (optional)

import logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
# Suppress Prophet's verbose logging output
# We don't want it printing to our server console


# ── FOLDER SETUP ─────────────────────────────────────────
RESULTS_DIR = os.path.join(settings.MEDIA_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# STEP 1: DEFAULT SALES DATASET
# ============================================================
# Generates 2 years of realistic sample sales data
# Used when user clicks "Use Default Data"

def generate_default_sales_data():
    """
    Creates 2 years of realistic fake sales data.
    Has upward trend + weekly patterns + monthly patterns + noise.
    This gives Prophet enough data to learn all the patterns!
    """
    np.random.seed(42)
    dates   = pd.date_range(start='2022-01-01', periods=730, freq='D')
    n       = len(dates)

    # Base upward trend
    trend   = np.linspace(1000, 2000, n)

    # Weekly pattern: weekdays higher than weekends
    # sin wave repeating every 7 days
    weekly  = 150 * np.sin(2 * np.pi * np.arange(n) / 7)

    # Monthly pattern: mid-month peak (payday effect)
    monthly = 100 * np.sin(2 * np.pi * np.arange(n) / 30)

    # Random noise to make it look real
    noise   = np.random.normal(0, 80, n)

    sales   = np.maximum(trend + weekly + monthly + noise, 100)

    return pd.DataFrame({'date': dates, 'sales': np.round(sales, 2)})


# ============================================================
# STEP 2: CSV FILE PROCESSOR
# ============================================================
# Reads uploaded CSV and auto-detects date + value columns

def process_uploaded_csv(file):
    """
    Reads user's CSV file.
    Auto-detects which column has dates and which has values.
    Returns cleaned DataFrame with 'date' and 'sales' columns.
    """
    try:
        df = pd.read_csv(file)

        if len(df) < 10:
            return None, "CSV needs at least 10 rows of data"

        # ── AUTO DETECT DATE COLUMN ──
        date_col  = None
        date_names = ['date','time','timestamp','period','month','week','day','datetime']

        for col in df.columns:
            if col.lower() in date_names:
                date_col = col; break

        if date_col is None:
            for col in df.columns:
                try:
                    pd.to_datetime(df[col].head(5))
                    date_col = col; break
                except: continue

        if date_col is None:
            return None, "Could not find a date column. Please name it 'date'."

        # ── AUTO DETECT VALUE COLUMN ──
        value_col   = None
        value_names = ['sales','revenue','value','amount','price',
                       'quantity','count','total','profit','income']

        for col in df.columns:
            if col != date_col and col.lower() in value_names:
                value_col = col; break

        if value_col is None:
            for col in df.columns:
                if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                    value_col = col; break

        if value_col is None:
            return None, "Could not find a numeric value column to forecast."

        # ── CLEAN AND PREPARE ──
        result            = df[[date_col, value_col]].copy()
        result.columns    = ['date', 'sales']
        result['date']    = pd.to_datetime(result['date'])
        result            = result.dropna().sort_values('date').reset_index(drop=True)

        return result, None

    except Exception as e:
        return None, f"Error reading CSV: {str(e)}"


# ============================================================
# STEP 3: PROPHET FORECASTING ENGINE
# ============================================================
# This is the HEART of our upgraded project!
# Takes historical data → trains Prophet → predicts future
#
# WHAT PROPHET DOES DIFFERENTLY from Linear Regression:
#
# Linear Regression:
#   → Found ONE straight line through all data
#   → Extended that line into the future
#   → Result: boring straight line forecast
#
# Prophet:
#   → Decomposes data into: Trend + Weekly + Yearly + Noise
#   → Trend: overall direction (up/down)
#   → Weekly seasonality: which days are busiest?
#   → Yearly seasonality: which months are busiest?
#   → Reconstructs these patterns for future dates
#   → Result: realistic fluctuating forecast! 

def run_prophet_forecast(df, forecast_days=30):
    """
    Runs Prophet forecasting on time series data.

    HOW IT WORKS (Simple Version):
    1. Prophet looks at 2 years of sales data
    2. It figures out:
       - "Sales are generally going UP over time" (trend)
       - "Sales are higher on Thursdays and Fridays" (weekly pattern)
       - "Sales peak in December" (yearly pattern)
    3. For future dates, it applies all these learned patterns
    4. Result: forecast that wiggles realistically just like real sales!

    PARAMETERS:
    - df           : DataFrame with 'date' and 'sales' columns
    - forecast_days: how many days into the future to predict

    RETURNS:
    - Dictionary with all results for chart and summary
    """

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # ── STEP A: PREPARE DATA FOR PROPHET ─────────────────
    # Prophet REQUIRES column names 'ds' (date) and 'y' (value)
    # This is a strict requirement — cannot use other names!
    prophet_df = df.rename(columns={'date': 'ds', 'sales': 'y'})

    # ── STEP B: CREATE AND CONFIGURE PROPHET MODEL ────────
    model = Prophet(
        yearly_seasonality=True,
        # Learn yearly patterns (higher sales in Dec, lower in Jan?)
        # Only useful if data spans at least 1 full year

        weekly_seasonality=True,
        # Learn weekly patterns (Mon-Sun variations)
        # This is what creates the UP and DOWN in the forecast!

        daily_seasonality=False,
        # Daily patterns (hourly within a day)
        # We have daily data so this is not applicable

        changepoint_prior_scale=0.05,
        # How flexible the trend line can be
        # 0.05 = moderate flexibility (not too rigid, not too flexible)
        # Higher = model adapts more to recent changes
        # Lower  = model follows long-term trend more

        seasonality_prior_scale=10,
        # How strong the seasonal patterns are
        # Higher = model trusts the seasonal patterns more

        interval_width=0.80,
        # Confidence interval width
        # 0.80 = 80% confidence band (realistic range)
        # This is wider than our old ±10% — more honest!
    )

    # ── STEP C: TRAIN THE MODEL ────────────────────────────
    model.fit(prophet_df)
    # This is where Prophet does all the learning!
    # It uses a statistical method called Stan (Bayesian inference)
    # to find the best trend, weekly pattern, and yearly pattern
    # that fits our historical data.
    # Takes about 3-10 seconds depending on data size.

    # ── STEP D: CREATE FUTURE DATES ───────────────────────
    future = model.make_future_dataframe(
        periods=forecast_days,
        freq='D'
        # freq='D' = daily frequency
        # This creates a DataFrame with dates:
        # historical dates + forecast_days new future dates
    )
    # Example: if history has 730 days and forecast_days=30,
    # future DataFrame has 760 rows total

    # ── STEP E: GENERATE FORECAST ─────────────────────────
    forecast = model.predict(future)
    # Prophet predicts for ALL dates (historical + future)
    # Key columns in forecast:
    # ds          = date
    # yhat        = predicted value (best estimate)
    # yhat_lower  = lower confidence bound
    # yhat_upper  = upper confidence bound
    # trend       = just the trend component
    # weekly      = just the weekly pattern component
    # yearly      = just the yearly pattern component

    # ── STEP F: SPLIT INTO HISTORICAL AND FUTURE ──────────
    # We only need the FUTURE part for the forecast section
    history_len      = len(df)
    historical_part  = forecast.iloc[:history_len]
    # First part = historical dates (model's fit on past data)

    future_part      = forecast.iloc[history_len:]
    # Last part = future dates (the actual forecast)

    # ── STEP G: CALCULATE ACCURACY METRICS ────────────────
    # Compare Prophet's predictions on historical data vs actual
    # This tells us how well the model learned the patterns

    # Use last 20% of historical data as test set
    test_start = int(history_len * 0.8)
    actual     = df['sales'].values[test_start:]
    predicted  = historical_part['yhat'].values[test_start:]

    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # Lower MAE and RMSE = better model fit

    # ── STEP H: DETECT TREND ──────────────────────────────
    # Compare trend at start vs trend at end of forecast
    trend_start = float(forecast['trend'].iloc[0])
    trend_end   = float(forecast['trend'].iloc[-1])
    trend_change = trend_end - trend_start

    if trend_change > 50:
        trend = '📈 Upward'
    elif trend_change < -50:
        trend = '📉 Downward'
    else:
        trend = '➡️ Stable'

    # ── STEP I: CALCULATE 7-DAY MOVING AVERAGE ────────────
    df['moving_avg'] = df['sales'].rolling(window=7, min_periods=1).mean()

    # ── STEP J: BUILD RESULT DICTIONARY ───────────────────
    # Package everything the chart and summary need

    # Show only last 90 days of history on chart (readable)
    show_last     = min(90, history_len)
    hist_slice    = df.tail(show_last)
    hist_forecast = historical_part.tail(show_last)

    future_values = np.round(future_part['yhat'].values, 2)
    upper_values  = np.round(future_part['yhat_upper'].values, 2)
    lower_values  = np.round(future_part['yhat_lower'].values, 2)

    # Make sure no negative forecasts
    future_values = np.maximum(future_values, 0)
    upper_values  = np.maximum(upper_values, 0)
    lower_values  = np.maximum(lower_values, 0)

    future_dates  = future_part['ds'].dt.strftime('%Y-%m-%d').tolist()

    result = {
        # ── HISTORICAL DATA ──
        'historical_dates'  : hist_slice['date'].dt.strftime('%Y-%m-%d').tolist(),
        'historical_values' : np.round(hist_slice['sales'].values, 2).tolist(),
        'moving_avg'        : np.round(hist_slice['moving_avg'].values, 2).tolist(),

        # ── PROPHET FIT ON HISTORICAL (how well model fits past) ──
        'prophet_fit'       : np.round(hist_forecast['yhat'].values, 2).tolist(),

        # ── FORECAST DATA (the future predictions) ──
        'forecast_dates'    : future_dates,
        'forecast_values'   : future_values.tolist(),
        'forecast_upper'    : upper_values.tolist(),
        'forecast_lower'    : lower_values.tolist(),

        # ── ACCURACY METRICS ──
        'mae'               : round(mae, 2),
        'rmse'              : round(rmse, 2),

        # ── SUMMARY INFO ──
        'trend'             : trend,
        'total_records'     : history_len,
        'forecast_start'    : future_dates[0] if future_dates else '',
        'forecast_end'      : future_dates[-1] if future_dates else '',
        'last_actual_value' : round(float(df['sales'].iloc[-1]), 2),
        'avg_forecast_value': round(float(np.mean(future_values)), 2),
        'max_forecast_value': round(float(np.max(future_values)), 2),
        'min_forecast_value': round(float(np.min(future_values)), 2),
        'model_used'        : 'Prophet (Facebook)',
    }

    return result


# ============================================================
# STEP 4: SAVE RESULT TO CSV
# ============================================================

def save_result_csv(result):
    """Saves forecast results to a downloadable CSV file."""
    uid      = str(uuid.uuid4())[:8]
    filename = f'forecast_{uid}.csv'
    filepath = os.path.join(RESULTS_DIR, filename)

    pd.DataFrame({
        'date'             : result['forecast_dates'],
        'forecasted_value' : result['forecast_values'],
        'lower_bound'      : result['forecast_lower'],
        'upper_bound'      : result['forecast_upper'],
    }).to_csv(filepath, index=False)

    return filename


# ============================================================
# DJANGO VIEW FUNCTIONS (Pages)
# ============================================================

def index(request):
    """HOME PAGE — shows the main forecasting form."""
    period_options = [
        (7,   '1 Week'),
        (14,  '2 Weeks'),
        (30,  '1 Month'),
        (60,  '2 Months'),
        (90,  '3 Months'),
        (180, '6 Months'),
    ]
    how_it_works = [
        {'icon': '📂', 'title': 'Load Data',
         'desc': 'Upload your CSV or use our built-in 2-year sales dataset'},
        {'icon': '🔍', 'title': 'Prophet Learns',
         'desc': 'AI learns trend, weekly patterns, and monthly patterns from your data'},
        {'icon': '🔮', 'title': 'Decompose',
         'desc': 'Splits data into: Trend + Weekly + Yearly + Noise components'},
        {'icon': '🚀', 'title': 'Forecast',
         'desc': 'Reconstructs all patterns for future dates — realistic ups and downs!'},
    ]
    return render(request, 'forecaster/index.html', {
        'period_options': period_options,
        'how_it_works'  : how_it_works,
    })


def run_forecast(request):
    """
    FORECAST RUNNER — handles form submission.
    Loads data → runs Prophet → saves to DB → redirects to result.
    """
    if request.method != 'POST':
        return redirect('index')

    forecast_days = int(request.POST.get('forecast_days', 30))
    use_default   = request.POST.get('use_default', 'false') == 'true'

    period_options = [
        (7,'1 Week'),(14,'2 Weeks'),(30,'1 Month'),
        (60,'2 Months'),(90,'3 Months'),(180,'6 Months'),
    ]
    how_it_works = [
        {'icon':'📂','title':'Load Data','desc':'Upload CSV or use default'},
        {'icon':'🔍','title':'Prophet Learns','desc':'AI learns all patterns'},
        {'icon':'🔮','title':'Decompose','desc':'Trend + Weekly + Yearly'},
        {'icon':'🚀','title':'Forecast','desc':'Realistic future prediction'},
    ]

    # ── LOAD DATA ─────────────────────────────────────────
    if use_default or 'csv_file' not in request.FILES:
        df           = generate_default_sales_data()
        dataset_name = 'Default Sales Data (2022-2023)'
    else:
        csv_file = request.FILES['csv_file']
        if not csv_file.name.endswith('.csv'):
            return render(request, 'forecaster/index.html', {
                'error': 'Please upload a .csv file only',
                'period_options': period_options,
                'how_it_works'  : how_it_works,
            })
        df, error = process_uploaded_csv(csv_file)
        if error:
            return render(request, 'forecaster/index.html', {
                'error': error,
                'period_options': period_options,
                'how_it_works'  : how_it_works,
            })
        dataset_name = csv_file.name

    # ── RUN PROPHET FORECAST ──────────────────────────────
    try:
        result = run_prophet_forecast(df, forecast_days)
    except Exception as e:
        return render(request, 'forecaster/index.html', {
            'error': f'Forecasting failed: {str(e)}',
            'period_options': period_options,
            'how_it_works'  : how_it_works,
        })

    # ── SAVE RESULT ───────────────────────────────────────
    result_file = save_result_csv(result)
    entry = ForecastHistory.objects.create(
        dataset_name   = dataset_name,
        forecast_days  = forecast_days,
        model_used     = 'Prophet (Facebook)',
        mae            = result['mae'],
        rmse           = result['rmse'],
        trend          = result['trend'],
        total_records  = result['total_records'],
        forecast_start = result['forecast_start'],
        forecast_end   = result['forecast_end'],
        result_file    = result_file,
    )

    request.session['chart_result'] = result
    return redirect('show_result', forecast_id=entry.id)


def show_result(request, forecast_id):
    """RESULTS PAGE — shows chart and metrics."""
    forecast = get_object_or_404(ForecastHistory, id=forecast_id)
    result   = request.session.get('chart_result', {})

    # Regenerate if session expired
    if not result:
        df     = generate_default_sales_data()
        result = run_prophet_forecast(df, forecast.forecast_days)

    forecast_summary = [
        ('Dataset',            forecast.dataset_name),
        ('Model Used',         'Prophet by Facebook ✨'),
        ('Total Records',      f"{forecast.total_records} rows"),
        ('Forecast Period',    f"{forecast.forecast_days} days"),
        ('Forecast Start',     forecast.forecast_start),
        ('Forecast End',       forecast.forecast_end),
        ('Detected Trend',     forecast.trend),
        ('MAE (Avg Error)',    f"{forecast.mae}"),
        ('RMSE',               f"{forecast.rmse}"),
    ]

    dates  = result.get('forecast_dates',  [])[:10]
    values = result.get('forecast_values', [])[:10]
    uppers = result.get('forecast_upper',  [])[:10]
    lowers = result.get('forecast_lower',  [])[:10]

    first_ten = []
    for i in range(len(dates)):
        first_ten.append({
            'date' : dates[i],
            'value': f"{values[i]:,.2f}" if i < len(values) else '-',
            'upper': f"{uppers[i]:,.2f}" if i < len(uppers) else '-',
            'lower': f"{lowers[i]:,.2f}" if i < len(lowers) else '-',
        })

    return render(request, 'forecaster/result.html', {
        'forecast'        : forecast,
        'result_json'     : json.dumps(result),
        'forecast_summary': forecast_summary,
        'first_ten'       : first_ten,
    })


def get_chart_data(request, forecast_id):
    """API — returns chart data as JSON."""
    forecast = get_object_or_404(ForecastHistory, id=forecast_id)
    df       = generate_default_sales_data()
    result   = run_prophet_forecast(df, forecast.forecast_days)
    return JsonResponse(result)


def history(request):
    """HISTORY PAGE — all past forecasts."""
    return render(request, 'forecaster/history.html', {
        'forecasts': ForecastHistory.objects.all()
    })


def dashboard(request):
    """DASHBOARD PAGE — summary stats."""
    forecasts = ForecastHistory.objects.all()
    total     = forecasts.count()
    mae_vals  = list(forecasts.filter(mae__isnull=False).values_list('mae', flat=True))
    avg_mae   = round(sum(mae_vals)/len(mae_vals), 2) if mae_vals else 0
    trends    = list(forecasts.values_list('trend', flat=True))
    return render(request, 'forecaster/dashboard.html', {
        'forecasts'      : forecasts[:5],
        'total_forecasts': total,
        'avg_mae'        : avg_mae,
        'upward'  : sum(1 for t in trends if 'Upward'   in str(t)),
        'downward': sum(1 for t in trends if 'Downward' in str(t)),
        'stable'  : sum(1 for t in trends if 'Stable'   in str(t)),
    })


def about(request):
    """ABOUT PAGE — project information."""
    hero_stats = [
        {'num': '730',     'lbl': 'Default Training Days'},
        {'num': 'Prophet', 'lbl': 'ML Algorithm'},
        {'num': '365',     'lbl': 'Max Forecast Days'},
        {'num': '100%',    'lbl': 'CSV Flexible'},
    ]
    steps = [
        {
            'title': 'Load & Validate Data',
            'desc' : 'User uploads a CSV or uses default sales data. We auto-detect the date and value columns.',
            'code' : 'df = pd.read_csv(file) → auto-detect date + value columns'
        },
        {
            'title': 'Prepare for Prophet',
            'desc' : 'Prophet requires columns named exactly "ds" (date) and "y" (value). We rename our columns accordingly.',
            'code' : 'prophet_df = df.rename(columns={"date":"ds", "sales":"y"})'
        },
        {
            'title': 'Train Prophet Model',
            'desc' : 'Prophet learns 3 things from your data: (1) Overall trend direction, (2) Weekly patterns (which day of week is busiest?), (3) Yearly patterns (which month is busiest?)',
            'code' : 'model = Prophet(weekly_seasonality=True, yearly_seasonality=True)'
        },
        {
            'title': 'Generate Future Dates',
            'desc' : 'We create a DataFrame with future dates — one row per day for the forecast period.',
            'code' : 'future = model.make_future_dataframe(periods=30)'
        },
        {
            'title': 'Predict Future Values',
            'desc' : 'Prophet applies the learned trend + weekly pattern + yearly pattern to future dates. This creates a realistic fluctuating forecast!',
            'code' : 'forecast = model.predict(future) → yhat, yhat_lower, yhat_upper'
        },
        {
            'title': 'Show Results',
            'desc' : 'We display the forecast on an interactive chart with confidence bands showing the realistic range of future values.',
            'code' : 'Chart shows: history + Prophet fit + forecast + confidence band'
        },
    ]
    tech_stack = [
        {'icon': '🐍', 'name': 'Python 3.8+',        'role': 'Main programming language'},
        {'icon': '🌐', 'name': 'Django',              'role': 'Web framework — handles URLs, views, database'},
        {'icon': '🔮', 'name': 'Prophet (Facebook)',  'role': 'Core forecasting model — trend + seasonality + confidence'},
        {'icon': '🐼', 'name': 'pandas',              'role': 'CSV reading, data tables, date handling'},
        {'icon': '🔢', 'name': 'numpy',               'role': 'Mathematical operations and array handling'},
        {'icon': '📊', 'name': 'Chart.js',            'role': 'Interactive forecast chart in the browser'},
        {'icon': '🗄️', 'name': 'SQLite + Django ORM', 'role': 'Stores forecast history without any setup'},
        {'icon': '🎨', 'name': 'Cormorant + DM Sans', 'role': 'Premium font pairing for the forest green aesthetic'},
    ]
    faqs = [
        {
            'q': 'Why did we upgrade from Linear Regression to Prophet?',
            'a': 'Linear Regression could only draw a perfectly straight line into the future — no ups and downs. Prophet is specifically designed for business time series data. It learns weekly patterns (Monday vs Sunday sales), yearly patterns (December vs January), and produces a realistic fluctuating forecast that looks like real business data.'
        },
        {
            'q': 'What does the confidence band (shaded area) mean?',
            'a': 'The shaded band shows the 80% confidence interval. This means Prophet is 80% sure the actual future value will fall somewhere inside this band. Unlike our old ±10% band, Prophet calculates this intelligently based on the actual uncertainty in the data. Wider band = more uncertainty. Narrower band = Prophet is more confident.'
        },
        {
            'q': 'What are ds and y in Prophet?',
            'a': 'Prophet requires specific column names. "ds" stands for "datestamp" — the date column. "y" is the value we want to forecast. Prophet was designed this way by Facebook and it is a strict requirement. We rename our date and sales columns to ds and y before feeding data to Prophet.'
        },
        {
            'q': 'Why does the forecast wiggle now (unlike before)?',
            'a': 'Because Prophet learned the weekly seasonality from your data! If your sales are consistently higher on Thursdays and lower on Sundays, Prophet applies this pattern to future weeks too. This is what makes Prophet so much better than Linear Regression for real business forecasting.'
        },
        {
            'q': 'How many days of data does Prophet need?',
            'a': 'Minimum recommended is about 2 full cycles of your seasonality. For weekly patterns, you need at least 2 weeks. For yearly patterns, you need at least 2 years. Our default dataset has 2 years which is perfect for learning all patterns!'
        },
    ]
    return render(request, 'forecaster/about.html', {
        'hero_stats': hero_stats,
        'steps'     : steps,
        'tech_stack': tech_stack,
        'faqs'      : faqs,
    })


def download_result(request, forecast_id):
    """DOWNLOAD — sends result CSV to browser."""
    forecast = get_object_or_404(ForecastHistory, id=forecast_id)
    filepath = os.path.join(RESULTS_DIR, forecast.result_file)
    if not os.path.exists(filepath):
        return HttpResponse('File not found', status=404)
    return FileResponse(open(filepath, 'rb'), as_attachment=True,
        filename=f'prophet_forecast_{forecast.forecast_start}.csv')


def delete_forecast(request, forecast_id):
    """DELETE — removes forecast from history."""
    forecast = get_object_or_404(ForecastHistory, id=forecast_id)
    if forecast.result_file:
        fp = os.path.join(RESULTS_DIR, forecast.result_file)
        if os.path.exists(fp): os.remove(fp)
    forecast.delete()
    return redirect('history')
