#Click run then use right arrow to advance count, left arrow to decrease count

import requests
from pynput import keyboard
import ntplib
from datetime import datetime, timezone, timedelta

# === CONFIGURATION ===
DATABASE_URL = "https://autonomous-hvac-default-rtdb.firebaseio.com"  
DB_SECRET = "D7UBQ1TtLE70Qhjy6TCRsVkiS3sPz6rEi5kbnsI2"                   
LOG_PATH = "/keyboard_count.json"                        

# get ntp time then adjust to timezone
def get_ntp_time():
    try:
        client = ntplib.NTPClient()
        response = client.request('pool.ntp.org')
        utc_time = datetime.fromtimestamp(response.tx_time, tz=timezone.utc)
        adjusted_time = utc_time - timedelta(hours=4)
        return adjusted_time.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print("NTP error, using local time:", e)
        return (datetime.now(timezone.utc) - timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S')


# === LOG COUNT + TIMESTAMP TO FIREBASE ===
def log_count_to_firebase():
    global count
    timestamp = get_ntp_time()
    payload = {
        "timestamp": timestamp,
        "count": count
    }
    url = f"{DATABASE_URL}{LOG_PATH}?auth={DB_SECRET}"
    response = requests.post(url, json=payload)
    print("POST response:", response.status_code, response.text)

# === UPDATE COUNT LOCALLY + LOG ===
def update_count(change):
    global count
    count += change
    log_count_to_firebase()
    print(f"Logged count: {count}")

# === KEY PRESS HANDLER ===
def on_press(key):
    try:
        if key == keyboard.Key.right:
            update_count(1)
        elif key == keyboard.Key.left:
            update_count(-1)
        elif key == keyboard.Key.esc:
            print("Exiting...")
            return False
    except Exception as e:
        print(f"Exception caught: {e}")

# === START ===
count = 0
print(f"Starting count: {count}")

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()