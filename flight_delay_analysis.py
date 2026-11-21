# ==============================================================================
# AIRLINE OPERATIONS & FLIGHT DELAY ANALYSIS REPORT
# Dataset: 14_flight_delays.csv
# Environment: Python (pandas, numpy, matplotlib, seaborn)
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define directories
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(PROJECT_DIR, "14_flight_delays.csv")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set global plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

print("="*80)
print("AIRLINE OPERATIONS & FLIGHT DELAY ANALYSIS PIPELINE")
print("="*80)

# ==============================================================================
# TASK 1: DATA INGESTION & INITIAL INSPECTION
# ==============================================================================
print("--- TASK 1: DATA INGESTION & INITIAL INSPECTION ---")
# Rule: Do not assume column names; inspect first
df_raw = pd.read_csv(DATA_FILE)

print("1.1 Dataset Shape (Rows, Columns):", df_raw.shape)
print("1.2 Real Column Names:", list(df_raw.columns))
print("1.3 Column Data Types:", df_raw.dtypes)
print("1.4 First 5 Rows:", df_raw.head())
print("1.5 Summary Statistics:", df_raw.describe())

# ==============================================================================
# TASK 2: DATA INTEGRITY, MISSING VALUE AUDIT & DEDUPLICATION
# ==============================================================================
print("--- TASK 2: MISSING VALUE AUDIT & DEDUPLICATION ---")
missing_count = df_raw.isnull().sum()
missing_pct = (missing_count / len(df_raw)) * 100
missing_df = pd.DataFrame({"Missing_Count": missing_count, "Missing_Pct (%)": missing_pct})
print("Missing values per column:", missing_df[missing_df["Missing_Count"] > 0])

# Check duplicate Flight_ID entries
dup_mask = df_raw.duplicated(subset=['Flight_ID'], keep=False)
duplicates = df_raw[dup_mask].sort_values(by='Flight_ID')
num_redundant = df_raw.duplicated(subset=['Flight_ID']).sum()
print(f"Duplicate Flight_ID instances found: {num_redundant} duplicate rows across {len(duplicates)} records.")
print("Sample duplicates:", duplicates[['Flight_ID', 'Flight_Date', 'Airline', 'Flight_Number', 'Origin', 'Destination']].head(6))

# Working on copy df_clean, keeping first occurrence
df_clean = df_raw.drop_duplicates(subset=['Flight_ID'], keep='first').copy()
print(f"Rows removed: {num_redundant} | Remaining clean rows: {len(df_clean)}")

# ==============================================================================
# TASK 3: LOGICAL IMPUTATION OF ARRIVAL_DELAY_MINS
# ==============================================================================
print("--- TASK 3: LOGICAL IMPUTATION OF ARRIVAL_DELAY_MINS ---")
completed_mask = (df_clean['Cancelled'] == 0) & (df_clean['Diverted'] == 0)
cancelled_count = df_clean['Cancelled'].sum()
diverted_count = df_clean['Diverted'].sum()
missing_completed = df_clean[completed_mask & df_clean['Arrival_Delay_Mins'].isnull()]

print(f"Cancelled Flights: {cancelled_count} (Legitimately missing arrival delay)")
print(f"Diverted Flights: {diverted_count} (Legitimately missing arrival delay)")
print(f"Completed Flights Missing Arrival Delay: {len(missing_completed)}")

# Evaluation of options:
# Cancelled and diverted flights are NOT imputed (they never arrived at scheduled destination).
# Completed flights are imputed using Group-wise Median by Route (Origin -> Destination),
# falling back to Airline Median, then Overall Median.
df_clean['Route'] = df_clean['Origin'] + " -> " + df_clean['Destination']
route_medians = df_clean[completed_mask].groupby('Route')['Arrival_Delay_Mins'].median()
airline_medians = df_clean[completed_mask].groupby('Airline')['Arrival_Delay_Mins'].median()
overall_median = df_clean.loc[completed_mask, 'Arrival_Delay_Mins'].median()

imputed_values = df_clean.loc[completed_mask & df_clean['Arrival_Delay_Mins'].isnull()].apply(
    lambda row: route_medians.get(row['Route'], airline_medians.get(row['Airline'], overall_median)), axis=1
)
df_clean.loc[completed_mask & df_clean['Arrival_Delay_Mins'].isnull(), 'Arrival_Delay_Mins'] = imputed_values

print("Distribution Before Imputation (Completed Flights, N=5,761):")
print(f"Mean: {df_raw.loc[(df_raw['Cancelled']==0)&(df_raw['Diverted']==0), 'Arrival_Delay_Mins'].mean():.2f}m | Median: {overall_median:.2f}m")
print("Distribution After Imputation (Completed Flights, N=5,831):")
print(f"Mean: {df_clean.loc[completed_mask, 'Arrival_Delay_Mins'].mean():.2f}m | Median: {df_clean.loc[completed_mask, 'Arrival_Delay_Mins'].median():.2f}m")

# ==============================================================================
# TASK 4: TEMPORAL FEATURE ENGINEERING & PARSING
# ==============================================================================
print("--- TASK 4: TEMPORAL FEATURE ENGINEERING ---")
df_clean['Flight_Date'] = pd.to_datetime(df_clean['Flight_Date'], errors='coerce')
unparseable_dates = df_clean['Flight_Date'].isnull().sum()
print(f"Unparseable flight dates: {unparseable_dates}")

df_clean['Month'] = df_clean['Flight_Date'].dt.month
df_clean['Month_Name'] = df_clean['Flight_Date'].dt.month_name()
df_clean['Day_of_Week'] = df_clean['Flight_Date'].dt.day_name()
print("Sample engineered date features:", df_clean[['Flight_Date', 'Month', 'Month_Name', 'Day_of_Week']].head(3))

# ==============================================================================
# TASK 5: HIGH-LEVEL OPERATIONAL METRICS & ON-TIME PERFORMANCE
# ==============================================================================
print("--- TASK 5: HIGH-LEVEL OPERATIONAL METRICS ---")

total_flights = len(df_clean)
valid_dep_flights = df_clean['Departure_Delay_Mins'].notnull().sum()
avg_dep_delay = df_clean['Departure_Delay_Mins'].mean()
dep_ontime_pct = (df_clean['Departure_Delay_Mins'] <= 15).sum() / valid_dep_flights * 100
arr_ontime_pct = (df_clean.loc[completed_mask, 'Arrival_Delay_Mins'] <= 15).sum() / completed_mask.sum() * 100

print(f"Total Network Scheduled Flights: {total_flights}")
print(f"Average Departure Delay (N={valid_dep_flights}): {avg_dep_delay:.2f} minutes")
print(f"Overall Departure On-Time Performance (Delay <= 15m): {dep_ontime_pct:.2f}%")
print(f"Overall Arrival On-Time Performance (Completed Flights, Delay <= 15m): {arr_ontime_pct:.2f}%")

# ==============================================================================
# TASK 6: ORIGIN AIRPORT BOTTLENECK RANKING
# ==============================================================================
print("--- TASK 6: ORIGIN AIRPORT BOTTLENECK RANKING ---")

airport_stats = df_clean.groupby('Origin').agg(
    Flight_Count=('Flight_ID', 'count'),
    Avg_Dep_Delay=('Departure_Delay_Mins', 'mean'),
    On_Time_Flights=('Departure_Delay_Mins', lambda x: (x <= 15).sum()),
    Valid_Dep_Flights=('Departure_Delay_Mins', 'count')
)
airport_stats['On_Time_Pct (%)'] = (airport_stats['On_Time_Flights'] / airport_stats['Valid_Dep_Flights']) * 100
airport_stats['Reliable_Sample'] = airport_stats['Flight_Count'] >= 30
airport_stats = airport_stats.sort_values(by='Avg_Dep_Delay', ascending=False)
print("Top 10 Origin Airports by Departure Delay:", airport_stats.head(10))

# Plot Top 10 Airports
top10_airports = airport_stats.head(10)
plt.figure(figsize=(10, 5))
colors = ['#d95f02' if x >= 10 else '#386cb0' for x in top10_airports['Avg_Dep_Delay']]
bars = plt.bar(top10_airports.index, top10_airports['Avg_Dep_Delay'], color=colors, edgecolor='black', alpha=0.85)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.25, f"{yval:.1f}m", ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.title("Top 10 Origin Airports by Average Departure Delay (minutes)All top 10 airports exceed reliability threshold (N >= 30 flights)", fontsize=12, pad=12)
plt.xlabel("Origin Airport (IATA Code)", fontsize=11)
plt.ylabel("Average Departure Delay (minutes)", fontsize=11)
plt.axhline(avg_dep_delay, color='red', linestyle='--', linewidth=1.2, label=f"Network Avg ({avg_dep_delay:.1f}m)")
plt.ylim(0, max(top10_airports['Avg_Dep_Delay']) * 1.15)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
fig6_path = os.path.join(FIGURES_DIR, "task06_top10_origin_airports.png")
plt.savefig(fig6_path, dpi=200)
plt.close()
print("Saved Task 6 chart to:", fig6_path)

# ==============================================================================
# TASK 7: WORST ROUTE DELAY ANALYSIS (N >= 10 FLIGHTS)
# ==============================================================================
print("--- TASK 7: WORST ROUTE DELAY ANALYSIS ---")
route_stats = df_clean[completed_mask].groupby('Route').agg(
    Flight_Count=('Flight_ID', 'count'),
    Avg_Arr_Delay=('Arrival_Delay_Mins', 'mean'),
    Avg_Dep_Delay=('Departure_Delay_Mins', 'mean')
)
top10_routes = route_stats[route_stats['Flight_Count'] >= 10].sort_values(by='Avg_Arr_Delay', ascending=False).head(10)
print("Top 10 Worst Routes (Min. 10 Flights):", top10_routes)

# Plot Worst Routes
plt.figure(figsize=(10, 6))
top10_routes_plot = top10_routes.sort_values(by='Avg_Arr_Delay', ascending=True)
bars = plt.barh(top10_routes_plot.index, top10_routes_plot['Avg_Arr_Delay'], color='#7570b3', edgecolor='black', alpha=0.85)
for bar in bars:
    xval = bar.get_width()
    plt.text(xval + 0.6, bar.get_y() + bar.get_height()/2.0, f"{xval:.1f}m", ha='left', va='center', fontsize=9, fontweight='bold')
plt.title("Top 10 Worst Routes by Average Arrival Delay (min. volume >= 10 flights)", fontsize=12, pad=12)
plt.xlabel("Average Arrival Delay (minutes)", fontsize=11)
plt.ylabel("Route (Origin -> Destination)", fontsize=11)
plt.xlim(0, max(top10_routes['Avg_Arr_Delay']) * 1.15)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
fig7_path = os.path.join(FIGURES_DIR, "task07_top10_worst_routes.png")
plt.savefig(fig7_path, dpi=200)
plt.close()
print("Saved Task 7 chart to:", fig7_path)

# ==============================================================================
# TASK 8: TIME-OF-DAY DELAY PROPAGATION ANALYSIS
# ==============================================================================
print("--- TASK 8: TIME-OF-DAY DELAY PROPAGATION ANALYSIS ---")

tod_order = ['Morning', 'Afternoon', 'Evening', 'Night']
tod_stats = df_clean.groupby('Time_of_Day').agg(
    Flight_Count=('Flight_ID', 'count'),
    Avg_Dep_Delay=('Departure_Delay_Mins', 'mean'),
    Avg_Arr_Delay=('Arrival_Delay_Mins', 'mean'),
    Dep_Delayed_15_Count=('Departure_Delay_Mins', lambda x: (x > 15).sum()),
    Valid_Dep_Flights=('Departure_Delay_Mins', 'count'),
    Arr_Delayed_15_Count=('Arrival_Delay_Mins', lambda x: (x > 15).sum()),
    Valid_Arr_Flights=('Arrival_Delay_Mins', 'count')
)
tod_stats['Pct_Dep_Delayed_Over_15 (%)'] = (tod_stats['Dep_Delayed_15_Count'] / tod_stats['Valid_Dep_Flights']) * 100
tod_stats['Pct_Arr_Delayed_Over_15 (%)'] = (tod_stats['Arr_Delayed_15_Count'] / tod_stats['Valid_Arr_Flights']) * 100
tod_stats = tod_stats.reindex(tod_order)
print("Time-of-Day Performance Comparison:", tod_stats[['Flight_Count', 'Avg_Dep_Delay', 'Avg_Arr_Delay', 'Pct_Dep_Delayed_Over_15 (%)', 'Pct_Arr_Delayed_Over_15 (%)']])

# Plot Time of Day
fig, ax1 = plt.subplots(figsize=(9, 5))
x = np.arange(len(tod_order))
width = 0.35
rects1 = ax1.bar(x - width/2, tod_stats['Avg_Dep_Delay'], width, label='Avg Dep Delay (mins)', color='#1b9e77', edgecolor='black', alpha=0.85)
rects2 = ax1.bar(x + width/2, tod_stats['Avg_Arr_Delay'], width, label='Avg Arr Delay (mins)', color='#d95f02', edgecolor='black', alpha=0.85)
ax1.set_ylabel('Average Delay (minutes)', fontsize=11)
ax1.set_xlabel('Scheduled Time of Day', fontsize=11)
ax1.set_title('Delay Propagation Across Time of Day: Cascading Operational Impact', fontsize=12, pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(tod_order, fontsize=10, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(x, tod_stats['Pct_Dep_Delayed_Over_15 (%)'], color='#7570b3', marker='o', linewidth=2.5, label='% Dep Delayed > 15m')
ax2.set_ylabel('% Flights Delayed > 15 Mins', fontsize=11, color='#7570b3')
ax2.tick_params(axis='y', labelcolor='#7570b3')
ax2.set_ylim(0, 30)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout()
fig8_path = os.path.join(FIGURES_DIR, "task08_time_of_day_propagation.png")
plt.savefig(fig8_path, dpi=200)
plt.close()
print("Saved Task 8 chart to:", fig8_path)

# ==============================================================================
# TASK 9: DELAY-CAUSE BREAKDOWN & TOP-5 AIRPORT CAUSE ANALYSIS
# ==============================================================================
print("--- TASK 9: DELAY-CAUSE BREAKDOWN ---")
delay_cols = ['Delay_Late_Aircraft', 'Delay_Air_Traffic', 'Delay_Carrier', 'Delay_Weather', 'Delay_Security']
clean_labels = ['Late Aircraft', 'Air Traffic (NAS)', 'Carrier', 'Weather', 'Security']
total_delay_mins = df_clean[delay_cols].sum()
pct_delay_share = (total_delay_mins / total_delay_mins.sum()) * 100
cause_df = pd.DataFrame({'Total_Minutes': total_delay_mins, 'Share_Pct (%)': pct_delay_share})
print("Delay Breakdown by Primary Root Cause:", cause_df)

top5_airports = airport_stats.head(5).index.tolist()
airport_causes = df_clean[df_clean['Origin'].isin(top5_airports)].groupby('Origin')[delay_cols].sum().reindex(top5_airports)
print("Delay Cause Minutes Across Top 5 Worst Origin Airports:", airport_causes)

# Plot Task 9: Overall Donut & Top 5 Stacked Bar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#ff7f00', '#4daf4a', '#377eb8', '#e41a1c', '#984ea3']
wedges, texts, autotexts = ax1.pie(total_delay_mins, labels=clean_labels, autopct='%1.1f%%',
                                    startangle=140, colors=colors, pctdistance=0.75,
                                    wedgeprops=dict(width=0.45, edgecolor='w'))
for autotext in autotexts:
    autotext.set_fontsize(9)
    autotext.set_weight('bold')
ax1.set_title(f"Overall Delay Breakdown by Cause (% Share)Total Tracked Delay: {total_delay_mins.sum():,.0f} mins", fontsize=11, pad=10)

bottom = np.zeros(len(top5_airports))
for i, col in enumerate(delay_cols):
    ax2.bar(top5_airports, airport_causes[col], bottom=bottom, label=clean_labels[i], color=colors[i], edgecolor='black', alpha=0.85)
    bottom += airport_causes[col].values

ax2.set_title("Delay Cause Minutes Across Top 5 Worst Origin Airports", fontsize=11, pad=10)
ax2.set_xlabel("Origin Airport", fontsize=10)
ax2.set_ylabel("Total Delay Minutes", fontsize=10)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
fig9_path = os.path.join(FIGURES_DIR, "task09_delay_causes_breakdown.png")
plt.savefig(fig9_path, dpi=200)
plt.close()
print("Saved Task 9 chart to:", fig9_path)

# ==============================================================================
# TASK 10: AIRLINE CANCELLATION & DIVERSION BENCHMARKING
# ==============================================================================
print("--- TASK 10: AIRLINE CANCELLATION & DIVERSION BENCHMARKING ---")

airline_ops = df_clean.groupby('Airline').agg(
    Total_Flights=('Flight_ID', 'count'),
    Cancelled_Flights=('Cancelled', 'sum'),
    Diverted_Flights=('Diverted', 'sum')
)
airline_ops['Cancellation_Rate (%)'] = (airline_ops['Cancelled_Flights'] / airline_ops['Total_Flights']) * 100
airline_ops['Diversion_Rate (%)'] = (airline_ops['Diverted_Flights'] / airline_ops['Total_Flights']) * 100
airline_ops = airline_ops.sort_values(by='Cancellation_Rate (%)', ascending=False)
print("Airline Operational Reliability Ranking:", airline_ops)

# Plot Airline Reliability
fig, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(airline_ops))
width = 0.35
rects1 = ax1.bar(x - width/2, airline_ops['Cancellation_Rate (%)'], width, label='Cancellation %', color='#e41a1c', edgecolor='black', alpha=0.85)
rects2 = ax1.bar(x + width/2, airline_ops['Diversion_Rate (%)'], width, label='Diversion %', color='#377eb8', edgecolor='black', alpha=0.85)

for rect in rects1:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., h + 0.08, f"{h:.2f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
for rect in rects2:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., h + 0.08, f"{h:.2f}%", ha='center', va='bottom', fontsize=8)

ax1.set_ylabel('Rate (%)', fontsize=11)
ax1.set_title('Cancellation and Diversion Rates by Airline/Carrier', fontsize=12, pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(airline_ops.index, fontsize=10, rotation=15, ha='right')
ax1.set_ylim(0, max(airline_ops['Cancellation_Rate (%)']) * 1.25)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend()
plt.tight_layout()
fig10_path = os.path.join(FIGURES_DIR, "task10_airline_cancellations_diversions.png")
plt.savefig(fig10_path, dpi=200)
plt.close()
print("Saved Task 10 chart to:", fig10_path)

# ==============================================================================
# TASK 11: CORRELATION ANALYSIS (PEARSON & SPEARMAN)
# ==============================================================================
print("--- TASK 11: CORRELATION ANALYSIS ---")

valid_delays = df_clean[['Departure_Delay_Mins', 'Arrival_Delay_Mins']].dropna()
pearson_corr = valid_delays['Departure_Delay_Mins'].corr(valid_delays['Arrival_Delay_Mins'], method='pearson')
spearman_corr = valid_delays['Departure_Delay_Mins'].rank().corr(valid_delays['Arrival_Delay_Mins'].rank(), method='pearson')
m, b = np.polyfit(valid_delays['Departure_Delay_Mins'], valid_delays['Arrival_Delay_Mins'], 1)

print(f"Sample Size N: {len(valid_delays)}")
print(f"Pearson Correlation (r): {pearson_corr:.4f}")
print(f"Spearman Rank Correlation (rho): {spearman_corr:.4f}")
print(f"Fitted Linear Model: Arrival_Delay = {m:.4f} * Departure_Delay + ({b:.4f})")

# Plot Correlation Scatter
plt.figure(figsize=(8, 6))
plt.scatter(valid_delays['Departure_Delay_Mins'], valid_delays['Arrival_Delay_Mins'], alpha=0.3, color='#1f77b4', edgecolors='none', s=20)
x_vals = np.array([valid_delays['Departure_Delay_Mins'].min(), valid_delays['Departure_Delay_Mins'].max()])
plt.plot(x_vals, m * x_vals + b, color='#d62728', linewidth=2, label=f'Fit: Arr = {m:.2f}*Dep + {b:.2f}')
plt.title(f"Departure vs Arrival Delay (Pearson r = {pearson_corr:.2f}, Spearman rho = {spearman_corr:.2f})\nN = {len(valid_delays)} completed flights", fontsize=12, pad=10)
plt.xlabel("Departure Delay (minutes)", fontsize=11)
plt.ylabel("Arrival Delay (minutes)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
fig11_path = os.path.join(FIGURES_DIR, "task11_correlation_scatter.png")
plt.savefig(fig11_path, dpi=200)
plt.close()
print("Saved Task 11 chart to:", fig11_path)

# ==============================================================================
# TASK 12: NUMPY METRICS & SKEWNESS EVALUATION
# ==============================================================================
print("--- TASK 12: NUMPY METRICS & SKEWNESS EVALUATION ---")
dep_delay_arr = df_clean['Departure_Delay_Mins'].dropna().to_numpy()
np_median = np.median(dep_delay_arr)
np_std = np.std(dep_delay_arr, ddof=1)
np_p90 = np.percentile(dep_delay_arr, 90)
np_mean = np.mean(dep_delay_arr)

print(f"NumPy Evaluation (Departure Delays, N={len(dep_delay_arr)}):")
print(f"Mean: {np_mean:.2f} mins")
print(f"Median: {np_median:.2f} mins")
print(f"Standard Deviation: {np_std:.2f} mins")
print(f"90th Percentile: {np_p90:.2f} mins")
print(f"Mean - Median Gap: {np_mean - np_median:.2f} mins (Positive right skew)")

# ==============================================================================
# TASK 13: OUTLIER DETECTION (IQR METHOD)
# ==============================================================================
print("--- TASK 13: OUTLIER DETECTION (IQR METHOD) ---")
q1 = np.percentile(dep_delay_arr, 25)
q3 = np.percentile(dep_delay_arr, 75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
lower_fence = q1 - 1.5 * iqr
outliers = df_clean[df_clean['Departure_Delay_Mins'] > upper_fence]
outlier_pct = (len(outliers) / len(dep_delay_arr)) * 100

print(f"Q1: {q1:.2f}m | Q3: {q3:.2f}m | IQR: {iqr:.2f}m")
print(f"Upper Fence: {upper_fence:.2f} mins | Lower Fence: {lower_fence:.2f} mins")
print(f"Identified Outliers: {len(outliers)} flights ({outlier_pct:.2f}% of flights)")
print("Top 5 Extreme Outliers:", outliers[['Flight_ID', 'Flight_Date', 'Airline', 'Origin', 'Destination', 'Departure_Delay_Mins', 'Arrival_Delay_Mins']].sort_values(by='Departure_Delay_Mins', ascending=False).head(5))

# Plot Boxplot
plt.figure(figsize=(9, 3.5))
plt.boxplot(dep_delay_arr, vert=False, patch_artist=True, boxprops=dict(facecolor='#aec7e8', color='#1f77b4'),
            medianprops=dict(color='red', linewidth=1.5), flierprops=dict(marker='o', markerfacecolor='#1f77b4', markersize=3, alpha=0.5))
plt.axvline(upper_fence, color='red', linestyle='--', linewidth=1.5, label=f'Upper Fence ({upper_fence:.1f} mins)')
plt.title(f"Boxplot of Departure Delays (IQR Method: Upper Fence = {upper_fence:.1f} mins)Outliers: N={len(outliers)} ({outlier_pct:.1f}%)", fontsize=11)
plt.xlabel("Departure Delay (minutes)", fontsize=10)
plt.legend()
plt.tight_layout()
fig13_path = os.path.join(FIGURES_DIR, "task13_departure_delay_boxplot.png")
plt.savefig(fig13_path, dpi=200)
plt.close()
print("Saved Task 13 chart to:", fig13_path)

# ==============================================================================
# TASK 14: DELAY SEVERITY CATEGORIZATION
# ==============================================================================
print("--- TASK 14: DELAY SEVERITY CATEGORIZATION ---")

conditions = [
    (df_clean['Cancelled'] == 1) | (df_clean['Diverted'] == 1),
    (df_clean['Departure_Delay_Mins'] <= 15),
    (df_clean['Departure_Delay_Mins'] > 15) & (df_clean['Departure_Delay_Mins'] <= 60),
    (df_clean['Departure_Delay_Mins'] > 60) & (df_clean['Departure_Delay_Mins'] <= 120),
    (df_clean['Departure_Delay_Mins'] > 120)
]
choices = ['Cancelled/Diverted', 'On-Time (<=15m)', 'Minor (16-60m)', 'Moderate (61-120m)', 'Severe (>120m)']
df_clean['Delay_Severity'] = np.select(conditions, choices, default='Unknown')

severity_counts = df_clean['Delay_Severity'].value_counts()
severity_pct = (severity_counts / len(df_clean)) * 100
severity_table = pd.DataFrame({'Flight_Count': severity_counts, 'Share_Pct (%)': severity_pct}).reindex(choices)
print("Delay Severity Classification Breakdown:", severity_table)

# ==============================================================================
# EXTRA ANALYSES: HEATMAP & AIRLINE RANKING
# ==============================================================================
print("--- EXTRA ANALYSES ---")

dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
short_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

heatmap_matrix = df_clean.pivot_table(index='Day_of_Week', columns='Month_Name', values='Departure_Delay_Mins', aggfunc='mean')
heatmap_matrix = heatmap_matrix.reindex(index=dow_order, columns=month_order)

plt.figure(figsize=(11, 5))
im = plt.imshow(heatmap_matrix.values, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, label='Avg Departure Delay (minutes)')
plt.xticks(np.arange(12), short_months, fontsize=10)
plt.yticks(np.arange(7), dow_order, fontsize=10)
for i in range(7):
    for j in range(12):
        val = heatmap_matrix.values[i, j]
        plt.text(j, i, f"{val:.1f}", ha='center', va='center', color='black' if val < 12 else 'white', fontsize=8, fontweight='bold')
plt.title("Seasonal & Weekly Heatmap: Average Departure Delay (minutes)", fontsize=12, pad=12)
plt.tight_layout()
fig_hm_path = os.path.join(FIGURES_DIR, "extra_heatmap_dow_month.png")
plt.savefig(fig_hm_path, dpi=200)
plt.close()
print("Saved Extra Heatmap to:", fig_hm_path)

airline_delays = df_clean.groupby('Airline').agg(
    Flight_Count=('Flight_ID', 'count'),
    Avg_Dep_Delay=('Departure_Delay_Mins', 'mean'),
    Avg_Arr_Delay=('Arrival_Delay_Mins', 'mean'),
    Median_Dep_Delay=('Departure_Delay_Mins', 'median'),
    On_Time_Pct=('Departure_Delay_Mins', lambda x: (x <= 15).sum() / x.count() * 100)
).sort_values(by='Avg_Dep_Delay', ascending=False)
print("Airline Overall Delay Performance Ranking:", airline_delays)

print("" + "="*80)
print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY.")
print("="*80)
import panda as pd
df= pd.read_csv(r"C:\Users\singh\Downloads\14_flight_delays.csv")
