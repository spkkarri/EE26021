import requests
import time

def download_raw_pgn(username, max_games=50000, output_file="lichess_raw.pgn"):
    url = f"https://lichess.org/api/games/user/{username}"
    
    headers = {
        "Accept": "application/x-chess-pgn"
    }

    downloaded = 0

    with open(output_file, "w", encoding="utf-8") as f:
        while downloaded < max_games:
            batch_size = min(300, max_games - downloaded)

            params = {
                "max": batch_size,
                "moves": True
            }

            print(f"Downloading {downloaded} → {downloaded + batch_size}...")

            response = requests.get(url, headers=headers, params=params, stream=True)

            if response.status_code != 200:
                print("Error:", response.status_code)
                break

            text = response.text
            if not text.strip():
                print("No more games available.")
                break

            f.write(text + "\n\n")
            downloaded += batch_size

            time.sleep(1)  # avoid rate limit

    print(f"✅ Done! Saved {downloaded} games to {output_file}")


# Example
download_raw_pgn("MagnusCarlsen", 50000)