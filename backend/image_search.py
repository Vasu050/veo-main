import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import Request, urlopen


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
MAX_IMAGE_SEARCHES_PER_VIDEO = 2
IMAGE_PROBE_TIMEOUT_SECONDS = 8


def _is_renderable_image_url(image_url: str) -> bool:
    parsed_url = urlparse(image_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname
        not in {"upload.wikimedia.org", "thumb.wikimedia.org"}
        or parsed_url.username
        or parsed_url.password
    ):
        return False

    request = Request(
        image_url,
        headers={"User-Agent": "AI-Video-Studio/1.0"},
    )
    try:
        with urlopen(
            request,
            timeout=IMAGE_PROBE_TIMEOUT_SECONDS,
        ) as response:
            final_url = urlparse(response.geturl())
            content_type = response.headers.get(
                "Content-Type",
                "",
            ).split(";", 1)[0].strip().lower()
            signature = response.read(12)
    except (HTTPError, URLError, TimeoutError):
        return False

    if (
        final_url.scheme != "https"
        or final_url.hostname
        not in {"upload.wikimedia.org", "thumb.wikimedia.org"}
        or not content_type.startswith("image/")
    ):
        return False

    return (
        signature.startswith(b"\x89PNG\r\n\x1a\n")
        or signature.startswith(b"\xff\xd8\xff")
        or signature.startswith((b"GIF87a", b"GIF89a"))
        or (
            signature.startswith(b"RIFF")
            and signature[8:12] == b"WEBP"
        )
    )


def _is_public_domain_or_cc0(image_url: str) -> bool:
    path_parts = unquote(urlparse(image_url).path).strip("/").split("/")
    try:
        commons_index = path_parts.index("commons")
    except ValueError:
        return False

    file_parts = path_parts[commons_index + 1 :]
    if file_parts and file_parts[0] == "thumb":
        file_parts = file_parts[1:]
    if len(file_parts) < 3:
        return False

    filename = file_parts[2]
    query = urlencode({
        "action": "query",
        "format": "json",
        "titles": f"File:{filename}",
        "prop": "imageinfo",
        "iiprop": "extmetadata",
    })
    request = Request(
        f"{COMMONS_API_URL}?{query}",
        headers={"User-Agent": "AI-Video-Studio/1.0"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(
            f"Wikimedia Commons license lookup returned HTTP {exc.code}."
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Wikimedia Commons license lookup failed: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Wikimedia Commons returned invalid license metadata."
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            "Wikimedia Commons returned invalid license metadata."
        )
    query_result = result.get("query")
    if not isinstance(query_result, dict):
        return False
    pages = query_result.get("pages", {})
    if not isinstance(pages, dict):
        return False

    for page in pages.values():
        if not isinstance(page, dict):
            continue
        image_info = page.get("imageinfo", [])
        if (
            not isinstance(image_info, list)
            or not image_info
            or not isinstance(image_info[0], dict)
        ):
            continue
        metadata = image_info[0].get("extmetadata", {})
        if not isinstance(metadata, dict):
            continue
        license_info = metadata.get("LicenseShortName", {})
        license_name = (
            license_info.get("value", "")
            if isinstance(license_info, dict)
            else ""
        )
        if (
            license_name.strip().casefold() == "public domain"
            or license_name.strip().casefold().startswith("cc0")
        ):
            return True
    return False


def _search_commons_image(query: str) -> str | None:
    search_queries = [query]
    normalized_query = query.casefold()
    has_person = any(
        word in normalized_query.split()
        for word in ("man", "men", "woman", "women", "person", "people")
    )
    if has_person:
        terms = [
            term
            for term in query.split()
            if term.casefold() not in {
                "realistic",
                "photorealistic",
                "portrait",
                "photo",
                "photograph",
                "photography",
            }
        ]
        if terms:
            broad_query = f"{' '.join(terms)} photograph filetype:bitmap"
            if broad_query.casefold() != normalized_query:
                search_queries.insert(0, broad_query)

    for search_term in search_queries:
        search_query = urlencode({
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": search_term,
            "gsrnamespace": 6,
            "gsrlimit": 10,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": 1280,
        })
        request = Request(
            f"{COMMONS_API_URL}?{search_query}",
            headers={"User-Agent": "AI-Video-Studio/1.0"},
        )

        try:
            with urlopen(request, timeout=20) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"Wikimedia Commons image search returned HTTP {exc.code}."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Wikimedia Commons image search failed: {exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Wikimedia Commons returned an invalid image-search response."
            ) from exc

        if not isinstance(result, dict):
            raise RuntimeError(
                "Wikimedia Commons returned an invalid image-search response."
            )
        query_result = result.get("query")
        pages = (
            query_result.get("pages")
            if isinstance(query_result, dict)
            else None
        )
        if not isinstance(pages, dict):
            continue

        for page in pages.values():
            if not isinstance(page, dict):
                continue
            image_info = page.get("imageinfo")
            if (
                not isinstance(image_info, list)
                or not image_info
                or not isinstance(image_info[0], dict)
            ):
                continue
            info = image_info[0]
            metadata = info.get("extmetadata")
            if not isinstance(metadata, dict):
                continue
            license_info = metadata.get("LicenseShortName")
            license_name = (
                license_info.get("value", "")
                if isinstance(license_info, dict)
                else ""
            )
            if not isinstance(license_name, str) or not (
                license_name.strip().casefold() == "public domain"
                or license_name.strip().casefold().startswith("cc0")
            ):
                continue
            image_url = info.get("thumburl")
            if (
                isinstance(image_url, str)
                and _is_renderable_image_url(image_url)
            ):
                return image_url
    return None


def _is_valid_web_image(image_url: str) -> bool:
    parsed_url = urlparse(image_url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.hostname:
        return False

    request = Request(
        image_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI-Video-Studio/1.0"},
    )
    try:
        with urlopen(request, timeout=IMAGE_PROBE_TIMEOUT_SECONDS) as response:
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            signature = response.read(12)
    except (HTTPError, URLError, TimeoutError, Exception):
        return False

    if not content_type.startswith("image/"):
        return False

    return (
        signature.startswith(b"\x89PNG\r\n\x1a\n")
        or signature.startswith(b"\xff\xd8\xff")
        or signature.startswith((b"GIF87a", b"GIF89a"))
        or (signature.startswith(b"RIFF") and signature[8:12] == b"WEBP")
    )


def _search_image(query: str, api_key: str | None) -> str | None:
    if not api_key:
        return _search_commons_image(query)

    request = Request(
        TAVILY_SEARCH_URL,
        data=json.dumps({
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_images": True,
            "include_image_descriptions": True,
            "include_answer": False,
            "max_results": 5,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, json.JSONDecodeError):
        return _search_commons_image(query)

    if not isinstance(result, dict):
        return _search_commons_image(query)

    images = result.get("images", [])
    if not isinstance(images, list):
        return _search_commons_image(query)

    for image in images:
        if not isinstance(image, dict):
            continue
        image_url = image.get("url")
        if isinstance(image_url, str) and _is_valid_web_image(image_url):
            return image_url

    return _search_commons_image(query)


def resolve_movie_image_searches(
    movie: dict[str, Any],
) -> list[str]:
    """Replace optional image-search markers with usable URLs or omit them."""

    image_elements: list[tuple[list[Any], dict[str, Any]]] = []
    containers: list[list[Any]] = []
    warnings: list[str] = []
    global_elements = movie.get("elements")
    if isinstance(global_elements, list):
        containers.append(global_elements)

    scenes = movie.get("scenes")
    if isinstance(scenes, list):
        for scene in scenes:
            if isinstance(scene, dict):
                elements = scene.get("elements")
                if isinstance(elements, list):
                    containers.append(elements)

    for elements in containers:
        for element in elements:
            if (
                isinstance(element, dict)
                and "image_search_query" in element
            ):
                if element.get("type") == "image":
                    image_elements.append((elements, element))
                else:
                    element.pop("image_search_query")
                    warnings.append(
                        "An image search marker was attached to a non-image "
                        "element and was ignored."
                    )

    if not image_elements:
        return list(dict.fromkeys(warnings))

    api_key = os.getenv("TAVILY_API_KEY")
    unresolved_queries: list[str] = []
    resolved_queries: dict[str, str | None] = {}
    search_count = 0

    for elements, image_element in image_elements:
        query_value = image_element.pop("image_search_query")
        query = query_value.strip() if isinstance(query_value, str) else ""
        if len(query) > 160:
            query = ""
        image_url = None

        if query:
            if query not in resolved_queries:
                if search_count >= MAX_IMAGE_SEARCHES_PER_VIDEO:
                    warnings.append(
                        "Only the first two distinct image searches are "
                        "used per video; remaining scenes use generated visuals."
                    )
                    resolved_queries[query] = None
                else:
                    search_count += 1
                    try:
                        resolved_queries[query] = _search_image(
                            query,
                            api_key,
                        )
                    except RuntimeError as exc:
                        warnings.append(str(exc))
                        resolved_queries[query] = None
            image_url = resolved_queries[query]

        if image_url:
            image_element["src"] = image_url
        else:
            elements.remove(image_element)
            if not query:
                warnings.append(
                    "An image search marker had no usable search query; "
                    "generated visuals are used instead."
                )
            elif query in resolved_queries and resolved_queries[query] is None:
                unresolved_queries.append(query)

    if unresolved_queries:
        references = " and ".join(
            f"“{query}”" for query in dict.fromkeys(unresolved_queries)
        )
        warnings.append(
            f"No usable public-domain or CC0 images were found for "
            f"{references}; generated visuals are used instead."
        )

    return list(dict.fromkeys(warnings))
