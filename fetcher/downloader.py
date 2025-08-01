import csv, requests, os, time
from PIL import Image
from io import BytesIO
import re
from bs4 import BeautifulSoup

# Discogs API credentials
# Use a Personal Access Token (recommended for personal scripts)
DISCOGS_TOKEN = "KofDHjxFdsaoOGCvTbztGerlmrGyZJfCKBVWRCYX" # Replace with your actual token

# A proper user agent is REQUIRED by Discogs API
USER_AGENT = "KosmischeCovers/1.0 +https://github.com/evemcgivern/KosmischeCovers"

# Build headers with token authentication
HEADERS = {
    "User-Agent": USER_AGENT,
    "Authorization": f"Discogs token={DISCOGS_TOKEN}"
}

# Enhance the rate_limited_request function to better handle auth errors
def rate_limited_request(url, params=None, headers=None, retries=3, delay=1):
    """Make a request with rate limiting and retries"""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers)

            # Handle authentication error
            if response.status_code == 401:
                print("❌ Authentication Error (401): Invalid or missing API credentials")
                print(f"Response: {response.text}")
                print("\nPlease check your Discogs token and make sure it's valid.")
                return None

            # If we get rate limited (429) or server error (5xx), wait and retry
            if response.status_code == 429 or response.status_code >= 500:
                wait_time = int(response.headers.get('Retry-After', delay * (attempt + 1)))
                print(f"Rate limited, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue

            # For successful responses, parse JSON and return
            if response.status_code == 200:
                return response
            else:
                print(f"API error: {response.status_code} - {response.text[:100]}")
        except Exception as e:
            print(f"Request error: {e}")

        # Only sleep between retries, not after the last one
        if attempt < retries - 1:
            time.sleep(delay)

    # If we get here, all retries failed
    return None


def select_best_image(images, min_primary=1200, min_fallback=800):
    """Given a list of (img_data, width, height), select best image by size."""
    best_primary = None
    best_fallback = None
    max_fallback_size = 0
    for img_data, width, height in images:
        min_dim = min(width, height)
        if min_dim >= min_primary:
            # Return immediately if primary found
            return img_data, width, height
        elif min_dim >= min_fallback and min_dim > max_fallback_size:
            best_fallback = (img_data, width, height)
            max_fallback_size = min_dim
    return best_fallback


def fetch_cover_musicbrainz(artist, album, out_dir="covers", min_size=1200, debug=False):
    """Search MusicBrainz for release and fetch cover from Cover Art Archive"""
    print(f"🔍 [MusicBrainz] Searching for: {artist} - {album}")
    # Search for release in MusicBrainz
    mb_url = "https://musicbrainz.org/ws/2/release/"
    params = {
        "query": f'artist:"{artist}" AND release:"{album}"',
        "fmt": "json",
        "limit": 5
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(mb_url, params=params, headers=headers)
        if r.status_code != 200:
            print(f"❌ MusicBrainz search failed: {r.status_code}")
            return
        data = r.json()
        releases = data.get("releases", [])
        images = []
        if not releases:
            print("⚠️ No MusicBrainz releases found")
            return
        # Try each release for cover art
        for rel in releases:
            mbid = rel.get("id")
            if not mbid:
                continue
            caa_url = f"https://coverartarchive.org/release/{mbid}/front"
            caa_r = requests.get(caa_url, headers=headers)
            if caa_r.status_code == 200:
                img_data = caa_r.content
                try:
                    with Image.open(BytesIO(img_data)) as img:
                        images.append((img_data, img.width, img.height))
                except Exception as e:
                    print(f"⚠️ [MusicBrainz] Error checking image size: {e}")
        best = select_best_image(images)
        if best:
            img_data, width, height = best
            fname = f"{artist} - {album}.jpg".replace("/", "_") if min(width, height) >= 1200 else f"{artist} - {album}_fallback.jpg".replace("/", "_")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, fname), "wb") as f:
                f.write(img_data)
            print(f"✓ [MusicBrainz] Downloaded {fname} ({width}×{height})")
            return
        print("⚠️ [MusicBrainz] No suitable cover found")
    except Exception as e:
        print(f"❌ [MusicBrainz] Error: {e}")

def fetch_cover(artist, album, out_dir="covers", debug=False):
    query = f"{artist} {album}"
    print(f"🔍 Searching for: {query}")
    r = rate_limited_request(
        "https://api.discogs.com/database/search",
        params={"q": query, "type": "release", "per_page": 10},
        headers=HEADERS,
    )
    # Check if the request was successful
    if not r:
        print(f"❌ Request failed for {query}")
        return

    try:
        data = r.json()
        if debug:
            print(f"API response status: {r.status_code}")
            if "error" in data:
                print(f"API error: {data.get('error', 'Unknown error')}")
                print(f"API message: {data.get('message', 'No message')}")
            elif "message" in data:
                print(f"API message: {data.get('message')}")
            if "results" in data:
                print(f"Found {len(data['results'])} results")
                for i, result in enumerate(data["results"][:5]):  # Up to 5 results
                    artist_val = result.get('artist', 'N/A')
                    title_val = result.get('title', 'N/A')
                    print(f"  {i+1}. Artist: {artist_val}, Title: {title_val}")
            else:
                print(f"No 'results' key in response. Keys: {list(data.keys())}")
    except Exception as e:
        print(f"❌ Failed to parse response for {query}: {e}")
        return
    results = data.get("results", [])
    # Try to find an exact match for artist and album
    match = None
    artist_norm = (
        artist.lower()
        .replace(" i", "")
        .replace(" ii", "")
        .replace(" iii", "")
        .replace(" iv", "")
        .replace(" v", "")
        .replace(".", "")
        .replace("-", "")
        .replace("ü", "u")
        .strip()
    )
    album_norm = (
        album.lower().replace(".", "").replace("-", "").replace("ü", "u").strip()
    )
    # First, try exact match
    for result in results:
        result_artist = result.get("artist", "")
        result_title = result.get("title", "")
        if isinstance(result_artist, list):
            result_artist = " ".join(result_artist)
        result_artist_norm = (
            result_artist.lower()
            .replace(" i", "")
            .replace(" ii", "")
            .replace(" iii", "")
            .replace(" iv", "")
            .replace(" v", "")
            .replace(".", "")
            .replace("-", "")
            .replace("ü", "u")
            .strip()
        )
        result_title_norm = (
            result_title.lower()
            .replace(".", "")
            .replace("-", "")
            .replace("ü", "u")
            .strip()
        )
        if result_artist_norm == artist_norm and result_title_norm == album_norm:
            match = result
            break
    # If no exact match, try partial match (artist or album contained in result)
    if not match:
        for result in results:
            result_artist = result.get("artist", "")
            result_title = result.get("title", "")
            if isinstance(result_artist, list):
                result_artist = " ".join(result_artist)
            result_artist_norm = (
                result_artist.lower()
                .replace(" i", "")
                .replace(" ii", "")
                .replace(" iii", "")
                .replace(" iv", "")
                .replace(" v", "")
                .replace(".", "")
                .replace("-", "")
                .replace("ü", "u")
                .strip()
            )
            result_title_norm = (
                result_title.lower()
                .replace(".", "")
                .replace("-", "")
                .replace("ü", "u")
                .strip()
            )
            # More flexible partial matching - check if parts of artist and album match
            if (artist_norm in result_artist_norm or
                result_artist_norm in artist_norm or
                any(word in result_artist_norm for word in artist_norm.split() if len(word) > 2)):

                if (album_norm in result_title_norm or
                    result_title_norm in album_norm or
                    any(word in result_title_norm for word in album_norm.split() if len(word) > 2)):
                    match = result
                    break
    # If still no match, fallback to first result
    if not match and results:
        match = results[0]
    if match:
        img_url = match.get("cover_image")
        if img_url:
            try:
                time.sleep(1)
                img_headers = {"User-Agent": USER_AGENT}
                img_response = requests.get(img_url, headers=img_headers)
                if img_response.status_code == 200:
                    img_data = img_response.content
                    try:
                        with Image.open(BytesIO(img_data)) as img:
                            min_dim = min(img.width, img.height)
                            if min_dim >= 1200:
                                fname = f"{artist} - {album}.jpg".replace("/", "_")
                                os.makedirs(out_dir, exist_ok=True)
                                with open(os.path.join(out_dir, fname), "wb") as f:
                                    f.write(img_data)
                                print(f"✓ Downloaded {fname} ({img.width}×{img.height})")
                                return True
                            elif min_dim >= 800:
                                fname = f"{artist} - {album}_fallback.jpg".replace("/", "_")
                                os.makedirs(out_dir, exist_ok=True)
                                with open(os.path.join(out_dir, fname), "wb") as f:
                                    f.write(img_data)
                                print(f"✓ Fallback downloaded {fname} ({img.width}×{img.height})")
                                return True
                            else:
                                print(f"⚠️ Skipped {artist} - {album}: image too small ({img.width}×{img.height})")
                                return False
                    except Exception as e:
                        print(f"⚠️ Error checking image size: {e}")
                        return False
                else:
                    print(f"⚠️ Image download failed: status {img_response.status_code}")
                    if img_response.status_code == 403:
                        print("⚠️ Forbidden (403): The server denied access to the image URL.")
                        print("Try setting a User-Agent header for the image request.")
                        print(f"Image URL: {img_url}")
                    return False
            except Exception as e:
                print(f"⚠️ Error downloading image: {e}")
                return False
        else:
            print(f"⚠️ No image for {query}")
            return False
    else:
        print(f"❌ No release found for {query}")
        return False


# Database of known release IDs for specific albums
KNOWN_RELEASE_IDS = {
    # Format: ("Artist", "Album"): release_id
    ("Amon Düül I", "Paradieswärts Düül"): 501129,
    ("Amon Düül II", "Wolf City"): 368873,
    # Add more as needed
}

# Test function to directly investigate a specific problem case
def test_specific_case():
    print("\n=== Testing with new User-Agent header ===\n")
    print("Testing specific problematic case:")
    fetch_cover("Amon Düül I", "Paradieswärts Düül", debug=True)
    print("\nTrying with hybrid approach:")
    fetch_cover_hybrid("Amon Düül I", "Paradieswärts Düül", debug=True)
    print("\nTrying hybrid approach with another album:")
    fetch_cover_hybrid("Amon Düül II", "Wolf City", debug=True)

# Function to fetch by Discogs release ID when known
def fetch_release_by_id(release_id, artist, album, out_dir="covers"):
    print(f"🔍 Fetching release ID: {release_id} for {artist} - {album}")

    r = rate_limited_request(
        f"https://api.discogs.com/releases/{release_id}",
        headers=HEADERS,
    )

    if not r:
        print(f"❌ Failed to fetch release {release_id} after retries")
        return False

    try:
        data = r.json()
        images = []
        if "images" in data and len(data["images"]) > 0:
            for img_info in data["images"]:
                img_url = img_info.get("uri")
                if img_url:
                    time.sleep(1)
                    img_headers = {"User-Agent": USER_AGENT}
                    img_response = requests.get(img_url, headers=img_headers)
                    if img_response.status_code == 200:
                        img_data = img_response.content
                        try:
                            with Image.open(BytesIO(img_data)) as img:
                                images.append((img_data, img.width, img.height))
                        except Exception as e:
                            print(f"⚠️ Error checking image size: {e}")
            best = select_best_image(images)
            if best:
                img_data, width, height = best
                fname = f"{artist} - {album}.jpg".replace("/", "_") if min(width, height) >= 1200 else f"{artist} - {album}_fallback.jpg".replace("/", "_")
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, fname), "wb") as f:
                    f.write(img_data)
                print(f"✓ Downloaded {fname} via release ID ({width}×{height})")
                return True
            else:
                print(f"⚠️ No suitable images for release {release_id}")
                return False
        else:
            print(f"⚠️ No images for release {release_id}")
            return False
    except Exception as e:
        print(f"❌ Failed to fetch release {release_id}: {e}")
        return False

# New function to search Google Images for album covers
def fetch_cover_google_images(artist, album, out_dir="covers", min_primary=1200, min_fallback=800, debug=False):
    """
    Scrape Google Images for album cover images and save the highest resolution found.
    """
    print(f"🔍 [Google Images] Searching for: {artist} - {album}")
    query = f"{artist} {album} album cover"
    search_url = "https://www.google.com/search"
    params = {"tbm": "isch", "q": query}
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        r = requests.get(search_url, params=params, headers=headers)
        if r.status_code != 200:
            print(f"❌ Google Images search failed: {r.status_code}")
            print(f"→ Request URL: {r.url}")
            print(f"→ Response text: {r.text[:500]}")
            return False
        soup = BeautifulSoup(r.text, "html.parser")
        images = []
        for img_tag in soup.select("img"):
            img_url = img_tag.get("src")
            if not img_url or img_url.startswith("data:"):
                continue
            try:
                img_headers = {"User-Agent": USER_AGENT}
                img_response = requests.get(img_url, headers=img_headers, timeout=10)
                if img_response.status_code == 200:
                    img_data = img_response.content
                    with Image.open(BytesIO(img_data)) as pil_img:
                        images.append((img_data, pil_img.width, pil_img.height))
            except Exception as e:
                if debug:
                    print(f"⚠️ [Google Images] Error downloading/checking image: {e}")
        best = select_best_image(images, min_primary=min_primary, min_fallback=min_fallback)
        if best:
            img_data, width, height = best
            fname = f"{artist} - {album}.jpg".replace("/", "_") if min(width, height) >= min_primary else f"{artist} - {album}_fallback.jpg".replace("/", "_")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, fname), "wb") as f_out:
                f_out.write(img_data)
            print(f"✓ [Google Images] Downloaded {fname} ({width}×{height})")
            return True
        print("⚠️ [Google Images] No suitable cover found")
        return False
    except Exception as e:
        print(f"❌ [Google Images] Error: {e}")
        return False

# New function to search archive.org for album covers
def fetch_cover_archiveorg(artist, album, out_dir="covers", min_size=1200, debug=False):
    """
    Search archive.org for album cover images and save if large enough.
    """
    print(f"🔍 [Archive.org] Searching for: {artist} - {album}")
    # Build search query for archive.org
    query = f'{artist} {album} cover'
    endpoint = "https://archive.org/advancedsearch.php"
    params = {
        "q": f'title:("{artist}" AND "{album}") AND mediatype:image',
        "fl[]": "identifier",
        "fl[]": "title",
        "rows": 10,
        "output": "json"
    }
    try:
        r = requests.get(endpoint, params=params)
        if r.status_code != 200:
            print(f"❌ Archive.org search failed: {r.status_code}")
            print(f"→ Request URL: {r.url}")
            print(f"→ Response text: {r.text[:500]}")
            return
        data = r.json()
        docs = data.get("response", {}).get("docs", [])
        images = []
        for doc in docs:
            identifier = doc.get("identifier")
            if not identifier:
                continue
            meta_url = f"https://archive.org/metadata/{identifier}"
            meta_r = requests.get(meta_url)
            if meta_r.status_code != 200:
                if debug:
                    print(f"⚠️ [Archive.org] Metadata fetch failed for {identifier}")
                continue
            meta = meta_r.json()
            files = meta.get("files", [])
            for f in files:
                name = f.get("name", "")
                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_url = f"https://archive.org/download/{identifier}/{name}"
                try:
                    img_headers = {"User-Agent": USER_AGENT}
                    img_response = requests.get(img_url, headers=img_headers, timeout=10)
                    if img_response.status_code == 200:
                        img_data = img_response.content
                        with Image.open(BytesIO(img_data)) as pil_img:
                            images.append((img_data, pil_img.width, pil_img.height))
                    else:
                        if debug:
                            print(f"⚠️ [Archive.org] Image download failed: status {img_response.status_code}")
                except Exception as e:
                    if debug:
                        print(f"⚠️ [Archive.org] Error downloading/checking image: {e}")
        best = select_best_image(images)
        if best:
            img_data, width, height = best
            fname = f"{artist} - {album}.jpg".replace("/", "_") if min(width, height) >= 1200 else f"{artist} - {album}_fallback.jpg".replace("/", "_")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, fname), "wb") as f_out:
                f_out.write(img_data)
            print(f"✓ [Archive.org] Downloaded {fname} ({width}×{height})")
            return
        print("⚠️ [Archive.org] No suitable cover found")
    except Exception as e:
        print(f"❌ [Archive.org] Error: {e}")

def fetch_cover_albumartworkfinder(artist, album, out_dir="covers", min_primary=1200, min_fallback=800, debug=False):
    """
    Scrape albumartworkfinder.com for album cover images and save the highest resolution found.
    """
    print(f"🔍 [AlbumArtworkFinder] Searching for: {artist} - {album}")
    search_url = "https://www.albumartworkfinder.com/search"
    params = {"q": f"{artist} {album}"}
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(search_url, params=params, headers=headers)
        if r.status_code != 200:
            print(f"❌ AlbumArtworkFinder search failed: {r.status_code}")
            print(f"→ Request URL: {r.url}")
            print(f"→ Response text: {r.text[:500]}")
            return
        html = r.text

        # Normalize artist and album for matching
        def norm(s):
            return s.lower().replace(".", "").replace("-", "").replace("ü", "u").replace("&", "and").strip()

        artist_norm = norm(artist)
        album_norm = norm(album)

        # Find all album result blocks
        album_blocks = re.findall(
            r'<div class="album-result">(.*?)</div>\s*</div>', html, re.DOTALL
        )
        found_block = None
        for block in album_blocks:
            # Extract artist and album text from block
            artist_match = re.search(r'<span class="album-artist">(.*?)</span>', block)
            album_match = re.search(r'<span class="album-title">(.*?)</span>', block)
            if artist_match and album_match:
                block_artist = norm(artist_match.group(1))
                block_album = norm(album_match.group(1))
                if block_artist == artist_norm and block_album == album_norm:
                    found_block = block
                    break
        if not found_block:
            print("⚠️ [AlbumArtworkFinder] No matching album found in search results")
            return

        # Find all image URLs and their dimensions in the matching album block
        img_matches = re.findall(r'<img[^>]+src="([^"]+)"[^>]*width="(\d+)"[^>]*height="(\d+)"', found_block)
        images = []
        for img_url, width, height in img_matches:
            try:
                width = int(width)
                height = int(height)
                img_headers = {"User-Agent": USER_AGENT}
                img_response = requests.get(img_url, headers=img_headers, timeout=10)
                if img_response.status_code == 200:
                    img_data = img_response.content
                    images.append((img_data, width, height))
                else:
                    if debug:
                        print(f"⚠️ [AlbumArtworkFinder] Image download failed: status {img_response.status_code}")
            except Exception as e:
                if debug:
                    print(f"⚠️ [AlbumArtworkFinder] Error downloading/checking image: {e}")
        best = select_best_image(images, min_primary=min_primary, min_fallback=min_fallback)
        if best:
            img_data, width, height = best
            fname = f"{artist} - {album}.jpg".replace("/", "_") if min(width, height) >= min_primary else f"{artist} - {album}_fallback.jpg".replace("/", "_")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, fname), "wb") as f_out:
                f_out.write(img_data)
            print(f"✓ [AlbumArtworkFinder] Downloaded {fname} ({width}×{height})")
            return
        print("⚠️ [AlbumArtworkFinder] No suitable cover found")
    except Exception as e:
        print(f"❌ [AlbumArtworkFinder] Error: {e}")

def fetch_cover_hybrid(artist, album, out_dir="covers", debug=False):
    print(f"🔍 Hybrid search for: {artist} - {album}")
    all_images = []
    sources = []

    # Discogs
    r = rate_limited_request(
        "https://api.discogs.com/database/search",
        params={"q": f"{artist} {album}", "type": "release", "per_page": 10},
        headers=HEADERS,
    )
    try:
        if r and r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            match = None
            artist_norm = artist.lower().replace("ü", "u").replace(".", "").replace("-", "").strip()
            album_norm = album.lower().replace("ü", "u").replace(".", "").replace("-", "").strip()
            for result in results:
                result_artist = result.get("artist", "")
                result_title = result.get("title", "")
                if isinstance(result_artist, list):
                    result_artist = " ".join(result_artist)
                result_artist_norm = result_artist.lower().replace("ü", "u").replace(".", "").replace("-", "").strip()
                result_title_norm = result_title.lower().replace("ü", "u").replace(".", "").replace("-", "").strip()
                if result_artist_norm == artist_norm and result_title_norm == album_norm:
                    match = result
                    break
            if not match and results:
                match = results[0]
            if match:
                img_url = match.get("cover_image")
                if img_url:
                    try:
                        img_headers = {"User-Agent": USER_AGENT}
                        img_response = requests.get(img_url, headers=img_headers)
                        if img_response.status_code == 200:
                            img_data = img_response.content
                            with Image.open(BytesIO(img_data)) as img:
                                all_images.append((img_data, img.width, img.height, "Discogs", img_url))
                    except Exception:
                        pass
    except Exception:
        pass

    # MusicBrainz
    mb_url = "https://musicbrainz.org/ws/2/release/"
    params = {
        "query": f'artist:"{artist}" AND release:"{album}"',
        "fmt": "json",
        "limit": 5
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(mb_url, params=params, headers=headers)
        if r.status_code == 200:
            data = r.json()
            releases = data.get("releases", [])
            for rel in releases:
                mbid = rel.get("id")
                if not mbid:
                    continue
                caa_url = f"https://coverartarchive.org/release/{mbid}/front"
                caa_r = requests.get(caa_url, headers=headers)
                if caa_r.status_code == 200:
                    img_data = caa_r.content
                    try:
                        with Image.open(BytesIO(img_data)) as img:
                            all_images.append((img_data, img.width, img.height, "MusicBrainz", caa_url))
                    except Exception:
                        pass
    except Exception:
        pass

    # Google Images
    query = f"{artist} {album} album cover"
    search_url = "https://www.google.com/search"
    params = {"tbm": "isch", "q": query}
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        r = requests.get(search_url, params=params, headers=headers)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            for img_tag in soup.select("img"):
                img_url = img_tag.get("src")
                if not img_url or img_url.startswith("data:"):
                    continue
                try:
                    img_headers = {"User-Agent": USER_AGENT}
                    img_response = requests.get(img_url, headers=img_headers, timeout=10)
                    if img_response.status_code == 200:
                        img_data = img_response.content
                        with Image.open(BytesIO(img_data)) as pil_img:
                            all_images.append((img_data, pil_img.width, pil_img.height, "Google Images", img_url))
                except Exception:
                    pass
    except Exception:
        pass

    # Archive.org
    endpoint = "https://archive.org/advancedsearch.php"
    params = {
        "q": f'title:("{artist}" AND "{album}") AND mediatype:image',
        "fl[]": "identifier",
        "fl[]": "title",
        "rows": 10,
        "output": "json"
    }
    try:
        r = requests.get(endpoint, params=params)
        if r.status_code == 200:
            data = r.json()
            docs = data.get("response", {}).get("docs", [])
            for doc in docs:
                identifier = doc.get("identifier")
                if not identifier:
                    continue
                meta_url = f"https://archive.org/metadata/{identifier}"
                meta_r = requests.get(meta_url)
                if meta_r.status_code != 200:
                    continue
                meta = meta_r.json()
                files = meta.get("files", [])
                for f in files:
                    name = f.get("name", "")
                    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    img_url = f"https://archive.org/download/{identifier}/{name}"
                    try:
                        img_headers = {"User-Agent": USER_AGENT}
                        img_response = requests.get(img_url, headers=img_headers, timeout=10)
                        if img_response.status_code == 200:
                            img_data = img_response.content
                            with Image.open(BytesIO(img_data)) as pil_img:
                                all_images.append((img_data, pil_img.width, pil_img.height, "Archive.org", img_url))
                    except Exception:
                        pass
    except Exception:
        pass

    # AlbumArtworkFinder
    search_url = "https://www.albumartworkfinder.com/search"
    params = {"q": f"{artist} {album}"}
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(search_url, params=params, headers=headers)
        if r.status_code == 200:
            html = r.text
            def norm(s):
                return s.lower().replace(".", "").replace("-", "").replace("ü", "u").replace("&", "and").strip()
            artist_norm = norm(artist)
            album_norm = norm(album)
            album_blocks = re.findall(
                r'<div class="album-result">(.*?)</div>\s*</div>', html, re.DOTALL
            )
            found_block = None
            for block in album_blocks:
                artist_match = re.search(r'<span class="album-artist">(.*?)</span>', block)
                album_match = re.search(r'<span class="album-title">(.*?)</span>', block)
                if artist_match and album_match:
                    block_artist = norm(artist_match.group(1))
                    block_album = norm(album_match.group(1))
                    if block_artist == artist_norm and block_album == album_norm:
                        found_block = block
                        break
            if found_block:
                img_matches = re.findall(r'<img[^>]+src="([^"]+)"[^>]*width="(\d+)"[^>]*height="(\d+)"', found_block)
                for img_url, width, height in img_matches:
                    try:
                        width = int(width)
                        height = int(height)
                        img_headers = {"User-Agent": USER_AGENT}
                        img_response = requests.get(img_url, headers=img_headers, timeout=10)
                        if img_response.status_code == 200:
                            img_data = img_response.content
                            all_images.append((img_data, width, height, "AlbumArtworkFinder", img_url))
                    except Exception:
                        pass
    except Exception:
        pass

    # Choose the largest available image (prefer >=1200, fallback >=800)
    def best_image(images, min_primary=1200, min_fallback=800):
        best_primary = None
        best_fallback = None
        max_fallback_size = 0
        for img_data, width, height, source, url in images:
            min_dim = min(width, height)
            if min_dim >= min_primary:
                return img_data, width, height, source, url
            elif min_dim >= min_fallback and min_dim > max_fallback_size:
                best_fallback = (img_data, width, height, source, url)
                max_fallback_size = min_dim
        return best_fallback

    best = best_image(all_images, min_primary=1200, min_fallback=800)
    if best:
        img_data, width, height, source, url = best
        safe_artist = artist.replace("/", "_").replace(" ", "_")
        safe_album = album.replace("/", "_").replace(" ", "_")
        fname = f"{safe_artist}_-_{safe_album}.jpg" if min(width, height) >= 1200 else f"{safe_artist}_-_{safe_album}_fallback.jpg"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, fname), "wb") as f_out:
            f_out.write(img_data)
        print(f"✓ Downloaded {fname} ({width}×{height}) from {source}: {url}\n")
        return True
    print(f"⚠️ No suitable cover found for {artist} - {album}\n")
    return False


# Expand our database with more release IDs
# You can add more entries here as you find them
KNOWN_RELEASE_IDS.update({
    # Amon Düül II albums
    ("Amon Düül II", "Phallus Dei"): 368869,
    ("Amon Düül II", "Yeti"): 368867,
    ("Amon Düül II", "Carnival in Babylon"): 368872,

    # Ash Ra Tempel albums
    ("Ash Ra Tempel", "Ash Ra Tempel"): 368535,
    ("Ash Ra Tempel", "Schwingungen"): 368536,
    ("Ash Ra Tempel & Timothy Leary", "7Up"): 2278351,
    ("Ash Ra Tempel", "Join Inn"): 368537,

    # Can albums
    ("Can", "Monster Movie"): 113961,
    ("Can", "Soundtracks"): 63566,
    ("Can", "Tago Mago"): 368320,
    ("Can", "Ege Bamyasi"): 7136,
    ("Can", "Delay"): 93422,

    # Cluster albums
    ("Cluster", "Cluster II"): 151317,
    ("Cluster", "Zuckerzeit"): 162825,
    ("Cluster", "Sowiesoso"): 162819,

    # Tony Conrad w/ Faust
    ("Tony Conrad w/ Faust", "Outside the Dream Syndicate"): 487310,

    # Cosmic Jokers albums
    ("Cosmic Jokers", "Cosmic Jokers"): 302493,
    ("Cosmic Jokers", "Galactic Supermarket"): 433470,
    ("Cosmic Jokers", "Planeten Sit-In"): 433478,
    ("Cosmic Jokers & Sternmädchen", "Gilles Zeitschiff"): 548489,

    # Faust albums
    ("Faust", "Faust AKA Clear"): 2006104,
    ("Faust", "So Far"): 555401,
    ("Faust", "The Faust Tapes"): 369438,
    ("Faust", "Faust IV"): 369437,

    # Popol Vuh albums
    ("Popol Vuh", "Affenstunde"): 486390,
    ("Popol Vuh", "In den Garten Pharaos"): 12200,
    ("Popol Vuh", "Einjäger & Siebenjäger"): 472101,
    ("Popol Vuh", "Hosianna Mantra"): 221073,

    # Tangerine Dream albums
    ("Tangerine Dream", "Electronic Meditation"): 87609,
    ("Tangerine Dream", "Alpha Centauri"): 248858,
    ("Tangerine Dream", "Atem"): 87615,
    ("Tangerine Dream", "Zeit"): 87613,

    # Klaus Schulze albums
    ("Klaus Schulze", "Irrlicht"): 1046483,
    ("Klaus Schulze", "Black Dance"): 87316,

    # Walter Wegmüller album
    ("Walter Wegmüller", "Tarot"): 485083,

    # Witthüser & Westrupp album
    ("Witthüser & Westrupp", "Trips & Träume"): 1694823,
})

# Test function to verify Discogs API access
def test_api_access():
    """Test Discogs API access with current credentials"""
    print("\n=== Testing Discogs API Access ===\n")

    # Try making a simple request to test authentication
    r = rate_limited_request(
        "https://api.discogs.com/",
        headers=HEADERS
    )

    if not r:
        print("❌ API request failed completely")
        return False

    try:
        data = r.json()
        print(f"API Response Status: {r.status_code}")

        if r.status_code == 200:
            print("✓ API access successful!")
            if "rate_limit" in data:
                print(f"Rate Limit: {data['rate_limit']}")
            print("\nAPI Response Data:")
            for key, value in data.items():
                print(f"  {key}: {value}")
            return True
        else:
            print(f"❌ API error: {r.status_code}")
            print(f"Message: {data.get('message', 'No message')}")
            return False
    except Exception as e:
        print(f"❌ Failed to parse API response: {e}")
        return False

# Before running the full process, test API access
if not test_api_access():
    print("\n❌ IMPORTANT: Discogs API access failed!")
    print("Please update your API credentials in the script.")
    print("Visit https://www.discogs.com/settings/developers to register your application")
    print("and get proper API credentials (consumer key and secret).")
    print("\nExiting script.")
    exit(1)

# Comment out the test and process the full CSV with hybrid approach
# test_specific_case()

# Process the full CSV using the hybrid approach
print("\n=== Processing full album list ===\n")
missing_artwork = []
with open("krautrock_list.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        # Add a short delay between albums to respect API rate limits
        if i > 0:
            time.sleep(5)  # 5 second delay between albums

        # Process the album
        result = fetch_cover_hybrid(
            row["Artist"],
            row["Album"],
            debug=False  # Set to True for verbose output
        )
        if not result:
            missing_artwork.append(f'{row["Artist"]} - {row["Album"]}')

if missing_artwork:
    print("\n=== Albums missing artwork ===")
    for entry in missing_artwork:
        print(entry)
else:
    print("\n✓ All albums have artwork.")
    for i, row in enumerate(reader):
        # Add a short delay between albums to respect API rate limits
        if i > 0:
            time.sleep(5)  # 5 second delay between albums

        # Process the album
        result = fetch_cover_hybrid(
            row["Artist"],
            row["Album"],
            debug=False  # Set to True for verbose output
        )
        if not result:
            missing_artwork.append(f'{row["Artist"]} - {row["Album"]}')

if missing_artwork:
    print("\n=== Albums missing artwork ===")
    for entry in missing_artwork:
        print(entry)
else:
    print("\n✓ All albums have artwork.")
