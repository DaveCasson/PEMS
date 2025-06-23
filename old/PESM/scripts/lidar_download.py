import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from requests.auth import HTTPBasicAuth

def get_session():
    session = requests.Session()
    session.auth = HTTPBasicAuth('your_username', 'your_password')
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    return session

def download_file(session, url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    local_filename = os.path.join(dest_folder, url.split('/')[-1])
    with session.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def is_matching_file(url, pattern):
    return re.search(pattern, url) is not None

def crawl_and_download(session, base_url, pattern, dest_folder):
    visited_urls = set()
    urls_to_visit = [base_url]
    base_domain = urlparse(base_url).netloc

    while urls_to_visit:
        current_url = urls_to_visit.pop()
        if current_url in visited_urls:
            continue

        print(f"Crawling: {current_url}")
        response = session.get(current_url)
        response.raise_for_status()
        visited_urls.add(current_url)

        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a'):
            href = link.get('href')
            if not href or href in ('#', '/', '?'):
                continue

            full_url = urljoin(current_url, href)
            if urlparse(full_url).netloc != base_domain:
                continue

            if full_url.endswith('/'):
                urls_to_visit.append(full_url)
            elif is_matching_file(full_url, pattern):
                print(f"Downloading: {full_url}")
                download_file(session, full_url, dest_folder)


if __name__ == "__main__":
    base_url = "https://n5eil01u.ecs.nsidc.org/ASO/ASO_50M_SD.001/"  # Replace with the base URL
    #pattern = r'.*USCATB.*\.tif$'
    pattern = r'.*USCATE.*\.tif$'
    
    dest_folder = "/Users/dcasson/Data/snow_data/tuolumne_lidar/"  # Replace with your desired download folder

    session = get_session()
    crawl_and_download(session, base_url, pattern, dest_folder)
