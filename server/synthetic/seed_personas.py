import json, urllib.request
from .personas import reader, skimmer, rager, form_lost

URL="http://127.0.0.1:8123/ingest"

def post_one(ev):
    data = json.dumps(ev).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req) as r:
        r.read()

def post_batch(evlist):
    # send as list to save round-trips
    data = json.dumps(evlist).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req) as r:
        r.read()

def main():
    sessions = []
    sessions += reader()
    sessions += skimmer()
    sessions += rager()
    sessions += form_lost()
    # send in chunks of ~500
    CH=500
    for i in range(0, len(sessions), CH):
        post_batch(sessions[i:i+CH])
    print(f"Seeded {len(sessions)} events across 4 personas.")

if __name__ == "__main__":
    main()
