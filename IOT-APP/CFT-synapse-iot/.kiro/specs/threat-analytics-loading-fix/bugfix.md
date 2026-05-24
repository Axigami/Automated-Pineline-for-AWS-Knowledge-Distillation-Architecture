# Bugfix Requirements Document

## Introduction

The Threat Analytics page (`src/modules/threat-analytics`) is currently non-functional. When users navigate to this page, they encounter an infinite loading spinner with no data displayed in any of the analytics sections (Attack Distribution, Attack Timeline, Top Attacker IPs, and Raw Network Logs). This prevents users from accessing critical threat intelligence and performing human-in-the-loop verification of network flow predictions.

The bug affects the core functionality of the threat analytics module, making it completely unusable. Users cannot view network flow logs, analyze attack patterns, or provide feedback on AI predictions.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the Threat Analytics page is loaded THEN the system displays an infinite loading spinner without ever showing data

1.2 WHEN the `queryLogs()` function is called on component mount THEN the system sets `isLoading` to true but fails to properly reset it to false after the query completes

1.3 WHEN the database query returns successfully with data THEN the system fails to populate the UI components (Attack Distribution chart, Attack Timeline chart, Top Attacker IPs list, and Raw Network Logs table remain empty)

1.4 WHEN the database query returns with no data THEN the system fails to display an appropriate empty state and continues showing the loading spinner indefinitely

1.5 WHEN an error occurs during data fetching THEN the system may not properly handle the error state, leaving the loading spinner active

### Expected Behavior (Correct)

2.1 WHEN the Threat Analytics page is loaded THEN the system SHALL fetch data from the `network_flows_feedback_all` table and display it within a reasonable timeframe (< 5 seconds)

2.2 WHEN the `queryLogs()` function completes successfully with data THEN the system SHALL set `isLoading` to false and populate all analytics components (Attack Distribution, Attack Timeline, Top Attacker IPs, Raw Network Logs)

2.3 WHEN the database query returns with no data THEN the system SHALL set `isLoading` to false and display an appropriate empty state message in each analytics section

2.4 WHEN an error occurs during data fetching THEN the system SHALL set `isLoading` to false, display the error message to the user, and allow retry attempts

2.5 WHEN the initial data load completes THEN the system SHALL enable all interactive features (search, filtering, time range selection, feedback submission)

### Unchanged Behavior (Regression Prevention)

3.1 WHEN a user performs a search query with label filters (e.g., "DDoS") THEN the system SHALL CONTINUE TO filter results by the predicted_label field

3.2 WHEN a user performs a search query with home filters (e.g., "@HomeName") THEN the system SHALL CONTINUE TO resolve the home name to ID and filter by flow_home_id

3.3 WHEN a user selects a time range preset (24h, 7d, custom) THEN the system SHALL CONTINUE TO filter results by the flow_ts timestamp field

3.4 WHEN new network flow data arrives via Supabase Realtime THEN the system SHALL CONTINUE TO prepend the new flow to the existing data and update all analytics charts

3.5 WHEN a user submits feedback on a flow label THEN the system SHALL CONTINUE TO perform optimistic UI updates and persist the feedback to the database

3.6 WHEN a user exports data to CSV THEN the system SHALL CONTINUE TO generate a properly formatted CSV file with all flow records

3.7 WHEN the Attack Distribution pie chart is rendered THEN the system SHALL CONTINUE TO use the correct color mapping for each attack label

3.8 WHEN the Attack Timeline bar chart is rendered THEN the system SHALL CONTINUE TO group flows by hour and stack them by attack label

3.9 WHEN the Top Attacker IPs list is rendered THEN the system SHALL CONTINUE TO aggregate flows by source IP and sort by count descending

3.10 WHEN pagination controls are used in the Raw Network Logs table THEN the system SHALL CONTINUE TO display 10 rows per page and allow navigation between pages
