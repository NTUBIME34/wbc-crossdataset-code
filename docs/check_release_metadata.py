#!/usr/bin/env python3
"""Validate draft/publication metadata without inventing externally assigned links."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


HERE = Path(__file__).resolve().parent
DOI = re.compile(r"^10\.\d{4,9}/\S+$")
PLACEHOLDER = re.compile(r"add at|placeholder|to be|example\.|your[-_ ]|<|>", re.I)
PUBLIC_FIELDS = ("release_version", "code.public_repository_url", "code.release_url",
                 "weights.record_doi", "weights.record_url", "weights.download_url")
URL_FIELDS = ("code.public_repository_url", "code.release_url", "weights.record_url",
              "weights.download_url", "paper.url")
DOI_FIELDS = ("code.archive_doi", "weights.record_doi", "paper.doi")


def get(value, key):
    for item in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(item)
    return value


def validate(data, publication=False, online=False):
    errors, warnings, checks = [], [], []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if data.get("release_state") not in {"draft", "published"}:
        errors.append("release_state must be draft or published")
    if not isinstance(get(data, "paper.title"), str) or not get(data, "paper.title").strip():
        errors.append("paper.title is required")
    if get(data, "paper.status") not in {"unpublished_manuscript", "preprint", "published"}:
        errors.append("paper.status must be unpublished_manuscript, preprint or published")
    if get(data, "weights.core_checkpoint_count") != 13:
        errors.append("weights.core_checkpoint_count must describe the unchanged 13-file core")
    publication = publication or data.get("release_state") == "published"
    if publication:
        for field in PUBLIC_FIELDS:
            if not get(data, field):
                errors.append(field + " is required for publication")
        if get(data, "paper.status") == "published" and not (get(data, "paper.doi") or get(data, "paper.url")):
            errors.append("A published paper needs its actual DOI or publisher URL")
    else:
        warnings.append("Draft metadata: no public-availability claim is made")
    for field in PUBLIC_FIELDS + URL_FIELDS + DOI_FIELDS:
        value = get(data, field)
        if value is not None and (not isinstance(value, str) or not value.strip() or PLACEHOLDER.search(value)):
            errors.append(field + " contains an invalid value or placeholder")
    for field in DOI_FIELDS:
        value = get(data, field)
        if value and isinstance(value, str) and not DOI.fullmatch(value):
            errors.append(field + " must be a bare DOI, not a URL")
    for field in URL_FIELDS:
        value = get(data, field)
        if not value or not isinstance(value, str):
            continue
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            errors.append(field + " must be an HTTPS URL without credentials")
        elif online:
            request = urllib.request.Request(value, headers={"User-Agent": "Research-release-metadata-check/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    status = response.status
                    checks.append(dict(field=field, status=status, reachable=200 <= status < 300))
                    if not 200 <= status < 300:
                        errors.append(field + " did not return HTTP success")
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                checks.append(dict(field=field, reachable=False, error=str(exc)))
                errors.append(field + " could not be accessed without credentials")
    if publication and not online:
        warnings.append("URLs were syntax-checked only; use --online before a public-availability claim")
    return dict(mode="publication" if publication else "draft", errors=errors, warnings=warnings,
                url_checks=checks, passed=not errors)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", type=Path, default=HERE / "RELEASE_METADATA.json")
    ap.add_argument("--publication", action="store_true", help="Fail on missing public release information")
    ap.add_argument("--online", action="store_true", help="Check configured HTTPS URLs without credentials")
    ap.add_argument("--artifact", type=Path, help="Also hash an already-built artifact")
    ap.add_argument("--sha256", help="Expected artifact SHA-256; requires --artifact")
    args = ap.parse_args()
    if args.sha256 and not args.artifact:
        ap.error("--sha256 requires --artifact")
    try:
        report = validate(json.loads(args.metadata.read_text()), args.publication, args.online)
        report["metadata_sha256"] = sha(args.metadata)
        if args.artifact:
            actual = sha(args.artifact)
            report["artifact"] = dict(name=args.artifact.name, sha256=actual)
            if args.sha256 and actual.lower() != args.sha256.lower():
                report["errors"].append("Artifact SHA-256 mismatch")
                report["passed"] = False
    except (OSError, ValueError, TypeError) as exc:
        report = dict(passed=False, errors=[str(exc)])
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
