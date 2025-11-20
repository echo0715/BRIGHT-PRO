#!/usr/bin/env python3
import argparse
import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Dict, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google.oauth2.credentials import Credentials as UserOAuthCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from tqdm import tqdm


def compute_md5(file_path: Path) -> str:
	"""Return hex MD5 of a local file."""
	hash_md5 = hashlib.md5()
	with file_path.open("rb") as f:
		for chunk in iter(lambda: f.read(1024 * 1024), b""):
			hash_md5.update(chunk)
	return hash_md5.hexdigest()


def get_drive_service_service_account() -> any:
	"""
	Build a Google Drive service using a service account.
	Requires env var GOOGLE_APPLICATION_CREDENTIALS pointing to the service account JSON key.
	"""
	credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
	if not credentials_path or not Path(credentials_path).is_file():
		raise RuntimeError(
			"Environment variable GOOGLE_APPLICATION_CREDENTIALS must point to a valid service account JSON key."
		)
	creds = ServiceAccountCredentials.from_service_account_file(
		credentials_path,
		scopes=["https://www.googleapis.com/auth/drive"],
	)
	return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_drive_service_oauth(client_secrets_path: str, token_path: str) -> any:
	"""
	Build a Google Drive service using OAuth (end-user Drive storage).
	Requires an OAuth client secrets JSON (Desktop app) and will store a token file.
	"""
	scopes = ["https://www.googleapis.com/auth/drive"]
	creds: Optional[UserOAuthCredentials] = None
	token_file = Path(token_path)
	if token_file.exists():
		creds = UserOAuthCredentials.from_authorized_user_file(str(token_file), scopes=scopes)
	if not creds or not creds.valid:
		if creds and creds.expired and creds.refresh_token:
			creds.refresh(Request())
		else:
			flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, scopes=scopes)
			# Use loopback IP (localhost) OAuth flow. Works with SSH port-forwarding.
			creds = flow.run_local_server(
				host="localhost",
				port=int(os.environ.get("BRIGHT_OAUTH_PORT", "8765")),
				open_browser=False,
				prompt="consent",
				authorization_prompt_message="Open the URL shown above in your browser. If running on a remote machine, first run: ssh -L {port}:localhost:{port} <remote-host>",
			)
		token_file.parent.mkdir(parents=True, exist_ok=True)
		token_file.write_text(creds.to_json())
	return build("drive", "v3", credentials=creds, cache_discovery=False)


def assert_folder_accessible(service, folder_id: str) -> None:
	"""Raise with a helpful message if the target Drive folder is not accessible."""
	try:
		service.files().get(
			fileId=folder_id,
			fields="id, name, mimeType",
			supportsAllDrives=True,
		).execute()
	except HttpError as e:
		if e.resp.status == 404:
			raise RuntimeError(
				f"Drive folder not found or not accessible: {folder_id}. "
				"Verify the ID is correct and share the folder with the Service Account."
			) from e
		raise


def find_item_by_name(service, name: str, parent_id: str, is_folder: bool) -> Optional[dict]:
	"""Find a file or folder by name within a given parent folder."""
	mime_cond = "mimeType = 'application/vnd.google-apps.folder'" if is_folder else "mimeType != 'application/vnd.google-apps.folder'"
	# Precompute escaped name to avoid backslashes in f-string expression
	safe_name = name.replace("'", "\\'")
	q = f"name = '{safe_name}' and '{parent_id}' in parents and {mime_cond} and trashed = false"
	resp = service.files().list(
		q=q,
		fields="files(id, name, md5Checksum, mimeType, modifiedTime)",
		spaces="drive",
		pageSize=1,
		includeItemsFromAllDrives=True,
		supportsAllDrives=True,
		corpora="allDrives",
	).execute()
	files = resp.get("files", [])
	return files[0] if files else None


def ensure_folder(service, name: str, parent_id: str) -> str:
	"""Ensure a folder exists under parent_id; return its file ID."""
	existing = find_item_by_name(service, name, parent_id, is_folder=True)
	if existing:
		return existing["id"]
	file_metadata = {
		"name": name,
		"mimeType": "application/vnd.google-apps.folder",
		"parents": [parent_id],
	}
	folder = service.files().create(
		body=file_metadata,
		fields="id, name",
		supportsAllDrives=True,
	).execute()
	return folder["id"]


def upload_or_update_file(service, local_path: Path, parent_id: str, skip_unchanged: bool = True) -> str:
	"""Upload a single file to Drive, updating if same-named file exists; returns Drive file ID."""
	existing = find_item_by_name(service, local_path.name, parent_id, is_folder=False)
	local_md5 = compute_md5(local_path)

	mime_type, _ = mimetypes.guess_type(str(local_path))
	media = MediaFileUpload(str(local_path), mimetype=mime_type or "application/octet-stream", resumable=True)

	if existing:
		remote_md5 = existing.get("md5Checksum")
		if skip_unchanged and remote_md5 and remote_md5 == local_md5:
			return existing["id"]
		updated = service.files().update(
			fileId=existing["id"],
			media_body=media,
			fields="id, name, md5Checksum",
			supportsAllDrives=True,
		).execute()
		return updated["id"]
	created = service.files().create(
		body={"name": local_path.name, "parents": [parent_id]},
		media_body=media,
		fields="id, name, md5Checksum",
		supportsAllDrives=True,
	).execute()
	return created["id"]


def sync_directory_to_drive(service, local_root: Path, drive_root_id: str, skip_unchanged: bool = True) -> None:
	"""
	Recursively mirror the local directory structure under the given Drive folder ID.
	Files with identical MD5 are skipped if skip_unchanged is True.
	"""
	if not local_root.is_dir():
		raise RuntimeError(f"Local path is not a directory: {local_root}")

	# Ensure root folder is accessible early
	assert_folder_accessible(service, drive_root_id)

	folder_cache: Dict[Path, str] = {local_root.resolve(): drive_root_id}

	all_files: list[Path] = [p for p in local_root.rglob("*") if p.is_file()]
	for file_path in tqdm(all_files, desc="Uploading files", unit="file"):
		parent_local = file_path.parent.resolve()
		# Build/lookup Drive folder chain up to drive_root_id
		stack = []
		while parent_local not in folder_cache and parent_local != local_root.parent:
			stack.append(parent_local)
			parent_local = parent_local.parent
		stack.reverse()

		parent_id = folder_cache[parent_local]
		for folder_local in stack:
			folder_id = ensure_folder(service, folder_local.name, parent_id)
			folder_cache[folder_local] = folder_id
			parent_id = folder_id

		try:
			upload_or_update_file(service, file_path, parent_id, skip_unchanged=skip_unchanged)
		except HttpError as e:
			raise RuntimeError(f"Failed to upload {file_path}: {e}") from e


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Upload local cache directory to Google Drive (mirrors structure).")
	default_cache = (Path(__file__).resolve().parents[1] / "cache").as_posix()
	parser.add_argument("--cache-dir", type=str, default=default_cache, help=f"Local cache directory (default: {default_cache})")
	parser.add_argument("--drive-folder-id", type=str, required=True, help="Target Google Drive folder ID (destination root).")
	parser.add_argument("--force", action="store_true", help="Force re-upload even if checksums match.")
	parser.add_argument("--auth", choices=["service", "oauth"], default="service", help="Authentication method: service (Service Account) or oauth (end-user Drive).")
	parser.add_argument("--oauth-client-secrets", type=str, default="", help="Path to OAuth client secrets JSON (Desktop app). Required if --auth oauth.")
	parser.add_argument("--oauth-token", type=str, default=str(Path.home() / ".config" / "bright_gdrive_token.json"), help="Path to store OAuth user token.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	local_cache = Path(args.cache_dir).resolve()
	if not local_cache.exists():
		raise RuntimeError(f"Cache directory does not exist: {local_cache}")
	if args.auth == "oauth":
		if not args.oauth_client_secrets:
			raise RuntimeError("--oauth-client-secrets is required when --auth oauth.")
		service = get_drive_service_oauth(args.oauth_client_secrets, args.oauth_token)
	else:
		service = get_drive_service_service_account()
	skip_unchanged = not args.force
	sync_directory_to_drive(service, local_cache, args.drive_folder_id, skip_unchanged=skip_unchanged)
	print(f"Upload completed from {local_cache} to Drive folder {args.drive_folder_id}")


if __name__ == "__main__":
	main()


