"""Conservative operating assumptions, not limits learned from the CSV.

Single-cell doorbell; revisit if hardware specifications are supplied.
"""
LIMITS = {
    "battery_v": (0.0, 5.0),
    "temp_c": (-40.0, 85.0),
    "rings_per_day": (0.0, 86400.0),  # at most one recorded event per second
    "wifi_rssi_dbm": (-127.0, 0.0),
    "age_months": (0.0, 600.0),  # up to 50 years
}
