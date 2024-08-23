import math
import os
import requests
from torch.hub import download_url_to_file, get_dir
from tqdm import tqdm
from urllib.parse import urlparse
import argparse

def sizeof_fmt(size, suffix='B'):
    """Convert file size to human-readable format."""
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}{suffix}"
        size /= 1024.0
    return f"{size:3.1f} Y{suffix}"


def download_file_from_google_drive(file_id, save_path):
    """Download a file from Google Drive."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        response = session.get(
            URL, params={'id': file_id, 'confirm': token}, stream=True)

    # Get file size for progress bar
    response_file_size = session.get(
        URL, params={'id': file_id}, stream=True, headers={'Range': 'bytes=0-2'})
    file_size = int(response_file_size.headers.get('Content-Range', '0').split('/')
                    [-1]) if 'Content-Range' in response_file_size.headers else None

    save_response_content(response, save_path, file_size)


def get_confirm_token(response):
    """Get confirmation token for Google Drive download."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination, file_size=None, chunk_size=32768):
    """Save the downloaded content to a file with optional progress bar."""
    with open(destination, 'wb') as f:
        downloaded_size = 0
        pbar = tqdm(total=math.ceil(file_size / chunk_size),
                    unit='chunk') if file_size else None

        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                if pbar:
                    pbar.update(1)
                    pbar.set_description(f"Downloaded {sizeof_fmt(downloaded_size)} / {sizeof_fmt(file_size)}")

        if pbar:
            pbar.close()


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load a file from a URL, downloading it if necessary."""
    if model_dir is None:
        model_dir = os.path.join(get_dir(), 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    filename = file_name or os.path.basename(urlparse(url).path)
    cached_file = os.path.abspath(os.path.join(model_dir, filename))

    if not os.path.exists(cached_file):
        print(f"Downloading: \"{url}\" to {cached_file}\n")
        download_url_to_file(url, cached_file, progress=progress)

    return cached_file


# _LINK = {
#     'vqgan': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth',
#     'vqgan_face256': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/celeba256_vq_f4_dim3_face.pth',
#     'vqgan_face512': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/ffhq512_vq_f8_dim8_face.pth',
#     'v1': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v1.pth',
#     'v2': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v2.pth',
#     'v3': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s4_v3.pth',
#     'bicsr': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_bicsrx4_s4.pth',
#     'inpaint_imagenet': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_imagenet_s4.pth',
#     'inpaint_face': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_face_s4.pth',
#     'faceir': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_faceir_s4.pth',
# }

# if __name__ == "__main__":
#     load_file_from_url(
#         url=_LINK['vqgan_face512'],
#         model_dir='./weights',
#     )


def main():
    parser = argparse.ArgumentParser(description="Download files from URLs or Google Drive.")
    parser.add_argument('--url', type=str, help='URL of the file to download.')
    parser.add_argument('--file_id', type=str, help='Google Drive file ID (if downloading from Google Drive).')
    parser.add_argument('--save_path', type=str, default='./weights', help='Path to save the downloaded file.')
    
    args = parser.parse_args()

    if args.file_id:
        if not args.url:
            download_file_from_google_drive(args.file_id, args.save_path)
        else:
            print("Error: Use either --url or --file_id, not both.")
    elif args.url:
        load_file_from_url(
            url=args.url,
            model_dir=args.save_path,
        )
    else:
        print("Error: You must provide either --url or --file_id.")

if __name__ == "__main__":
    main()