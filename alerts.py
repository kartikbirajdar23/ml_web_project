def trigger_traffic_alert(location, predicted_count, traffic_level):
    if traffic_level == "HIGH TRAFFIC":
        alert_msg = f"🚨 EMERGENCY TRAFFIC ALERT! High congestion ({predicted_count} vehicles) detected at {location}."
        print(f"[SYSTEM LOG]: Email/SMS Sent to Traffic Authorities -> {alert_msg}")
        return True, alert_msg
    return False, "Traffic flow is normal."