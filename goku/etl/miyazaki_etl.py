import logging
from goku.etl import etl
from goku.etl.etl import log_time

log = logging.getLogger(__name__)
log.setLevel(level=logging.INFO)


def get_all_images(to_dir: str):
    works_links = get_works_links("https://www.ghibli.jp/works/")
    for works_link in works_links:
        _download_images(to_dir, works_link)


def get_works_links(base_url: str):
    soup = etl.get_soup(base_url)

    links = [a.get("href", None) for a in soup.find_all("a")]
    links_proc = [href[len(base_url):] for href in links if href.startswith(base_url)]
    subdomains = [href.split("/")[0] for href in links_proc if href]

    links_works = [base_url + link for link in list(set(subdomains))]

    return list(set(links_works))


@log_time(log)
def _download_images(to_dir: str, base_url: str):
    soup = etl.get_soup(base_url)

    for figure in soup.find_all("figure"):
        for a_tag in figure.find_all("a"):
            img_url = a_tag.get("href", None)

            if img_url is not None:
                etl.download_image(to_dir, img_url)


def download_chihiro(to_dir: str):
    _download_images(to_dir, "https://www.ghibli.jp/works/chihiro/")
