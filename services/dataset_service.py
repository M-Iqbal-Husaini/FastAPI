# services/dataset_service.py
import logging
from typing import Optional, Tuple, List

import requests

log = logging.getLogger("uvicorn.error")

# satu session global biar koneksi keep-alive
_session = requests.Session()


def fetch_dataset_from_laravel(
    base_url: str,
    dataset_id: int,
    page: int,
    per_page: int,
    internal_token: str,
    timeout: Tuple[int, int] = (5, 120),  # (connect, read) seconds
) -> Optional[dict]:
    """
    Ambil satu page dataset dari Laravel:
    Laravel route: GET {base_url}/api/internal/dataset/{dataset_id}?page=&per_page=
    """
    url = f"{base_url.rstrip('/')}/api/internal/dataset/{dataset_id}"
    params = {"page": page, "per_page": per_page}
    headers = {"X-INTERNAL-TOKEN": internal_token}

    try:
        r = _session.get(
            url, params=params, headers=headers, timeout=timeout
        )
        r.raise_for_status()
        data = r.json()
        return data
    except Exception as e:
        log.error("Error fetching dataset from Laravel: %s", e, exc_info=True)
        return None


def fetch_full_dataset_from_laravel(
    base_url: str,
    dataset_id: int,
    per_page: int,
    internal_token: str,
) -> Tuple[List[str], List[int]]:
    """
    Loop semua page sampai habis, return (texts, labels).
    """
    texts: List[str] = []
    labels: List[int] = []

    page = 1
    while True:
        data = fetch_dataset_from_laravel(
            base_url=base_url,
            dataset_id=dataset_id,
            page=page,
            per_page=per_page,
            internal_token=internal_token,
        )
        if not data or "items" not in data:
            break

        items = data.get("items", [])
        if not items:
            break

        for it in items:
            texts.append(it.get("text", ""))
            lab = it.get("label")
            # jika null => default 0
            labels.append(int(lab) if lab is not None and str(lab) != "" else 0)

        total = data.get("total", 0)
        fetched = page * per_page
        log.info(
            "Fetched page %s (%s items), total=%s, fetched=%s",
            page,
            len(items),
            total,
            fetched,
        )
        if fetched >= total:
            break
        page += 1

    log.info(
        "Finished fetching dataset_id=%s: %s texts",
        dataset_id,
        len(texts),
    )
    return texts, labels
