"""
md_parser.py
------------
Parses your podcast .md files into searchable chunks.

YOUR MD FILE HAS TWO USEFUL STRUCTURES:

1. YAML frontmatter (between --- markers) contains:
   - Episode metadata: title, season, episode number
   - ids.youtube: the YouTube video ID  
   - quotableClips: pre-labeled segments with startOffset (seconds) + url
   - transcript: list of {header, line, sec, time, who} entries

2. We use BOTH to build chunks:
   - Strategy A (Header Chunks): Group transcript lines under each section 
     header (e.g. "Personalization Strategy", "A/B Testing") — these map 
     perfectly to quotableClips for deep links
   - Strategy B (Window Chunks): Sliding 60-second windows over raw lines
     for fine-grained retrieval

WHY BOTH?
   Header chunks are great for topic-level questions ("What did they say about A/B testing?")
   Window chunks are great for specific detail questions ("What exact steps count did they mention?")

HOW TO USE:
   from md_parser import parse_md_file, load_all_episodes
   segments = load_all_episodes("./episodes")
"""

import os
import re
import yaml
import requests
import frontmatter
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# STEP 1: Split the .md file into YAML frontmatter + body
# ─────────────────────────────────────────────────────────────

def split_frontmatter(content: str) -> tuple[dict, str]:
    """
    Parse frontmatter from markdown content using python-frontmatter.
    
    .md files look like this:
    
        ---
        title: "Episode title"
        season: 8
        transcript:
          - header: Introduction
          - line: Hello everyone...
            sec: 1
        ---
        
        Links:
        * [LinkedIn](...)

    This function splits it into:
      - A Python dict of all the YAML data (frontmatter)
      - The markdown body text after the closing ---

    Args:
        content: The raw markdown content (either from file or fetched from GitHub)
    
    Returns (yaml_data, body_text)
    """
    try:
        post = frontmatter.loads(content)
        return post.metadata, post.content
    except Exception as e:
        raise ValueError(f"Could not parse YAML frontmatter: {e}")


# ─────────────────────────────────────────────────────────────
# STEP 2: Build a lookup table from quotableClips
# ─────────────────────────────────────────────────────────────

def build_clip_lookup(quotable_clips: list[dict]) -> dict[str, dict]:
    """
    quotableClips in your YAML looks like:
    
        quotableClips:
          - name: 'Personalization Strategy: Agenda-Driven Recommender Systems'
            startOffset: 2139
            url: https://www.youtube.com/watch?v=IDzhmmKeNG4&t=2139
            endOffset: 2397
    
    We build a dict keyed by clip NAME so we can quickly look up the 
    deep link for any section header we find in the transcript.
    
    Returns: { "Personalization Strategy: ...": {startOffset, url, endOffset}, ... }
    """
    lookup = {}
    for clip in (quotable_clips or []):
        name = clip.get("name", "")
        lookup[name] = {
            "start_seconds": clip.get("startOffset", 0),
            "end_seconds":   clip.get("endOffset", 0),
            "deep_link":     clip.get("url", ""),
            "timestamp":     seconds_to_timestamp(clip.get("startOffset", 0)),
        }
    return lookup


def seconds_to_timestamp(seconds: int) -> str:
    """Convert 150 → '2:30'"""
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


# ─────────────────────────────────────────────────────────────
# STEP 3A: Header-based chunking (PRIMARY strategy)
# ─────────────────────────────────────────────────────────────

def chunk_by_headers(
    transcript: list[dict],
    clip_lookup: dict,
    episode_meta: dict,
) -> list[dict]:
    """
    Your transcript has section headers mixed in with lines:
    
        - header: 'Personalization Strategy: Agenda-Driven Recommender Systems'
        - line: "Yes. Again, this is a work in progress..."
          sec: 2161
          who: Stefan
        - line: "But I think what we are trying to do..."
          sec: 2200
          who: Stefan
        - header: 'A/B Testing as Personalization Foundation'   ← new section starts
    
    This function groups all lines under each header into one chunk.
    It then looks up the matching quotableClip to get the deep link.
    
    Each chunk produced looks like:
    {
        "chunk_type":     "header",
        "section":        "Personalization Strategy: Agenda-Driven Recommender Systems",
        "text":           "Stefan: Yes. Again, this is a work in progress...",
        "speakers":       "Stefan, Alexey",
        "episode_title":  "AI in Healthcare & Digital Therapeutics",
        "video_id":       "IDzhmmKeNG4",
        "season":         8,
        "episode_num":    4,
        "start_seconds":  2139,
        "timestamp":      "35:39",
        "deep_link":      "https://www.youtube.com/watch?v=IDzhmmKeNG4&t=2139",
    }
    """
    chunks = []
    
    current_header = "Introduction"
    current_lines  = []
    current_start  = 0

    def flush_chunk():
        """Save whatever we've accumulated as a completed chunk."""
        if not current_lines:
            return
        
        # Combine lines with speaker attribution: "Stefan: Hello there."
        text_parts = []
        speakers_seen = set()
        for entry in current_lines:
            speaker = entry.get("who", "")
            line    = entry.get("line", "").strip()
            if speaker:
                text_parts.append(f"{speaker}: {line}")
                speakers_seen.add(speaker)
            else:
                text_parts.append(line)
        
        combined_text = " ".join(text_parts)
        
        # Look up the matching quotableClip for the deep link
        clip_info = clip_lookup.get(current_header, {})
        
        # Fallback deep link if no exact clip match
        if clip_info:
            start_sec = clip_info["start_seconds"]
            deep_link = clip_info["deep_link"]
            timestamp = clip_info["timestamp"]
        else:
            start_sec = current_start
            video_id  = episode_meta.get("ids", {}).get("youtube", "")
            deep_link = f"https://www.youtube.com/watch?v={video_id}&t={start_sec}"
            timestamp = seconds_to_timestamp(start_sec)
        
        chunks.append({
            "chunk_type":    "header",
            "section":       current_header,
            "text":          combined_text,
            "speakers":      ", ".join(sorted(speakers_seen)),
            
            # Episode-level metadata
            "episode_title": episode_meta.get("title", ""),
            "short_title":   episode_meta.get("short", ""),
            "video_id":      episode_meta.get("ids", {}).get("youtube", ""),
            "season":        str(episode_meta.get("season", "")),
            "episode_num":   str(episode_meta.get("episode", "")),
            
            # Timing & links
            "start_seconds": start_sec,
            "timestamp":     timestamp,
            "deep_link":     deep_link,
        })

    # Walk through the transcript list item by item
    for item in transcript:
        if "header" in item:
            # We hit a new section — save what we had, start fresh
            flush_chunk()
            current_header = item["header"]
            current_lines  = []
            # Try to find the start time from the clip lookup
            clip_info      = clip_lookup.get(current_header, {})
            current_start  = clip_info.get("start_seconds", 0)
        
        elif "line" in item:
            current_lines.append(item)
    
    # Don't forget the last section
    flush_chunk()
    
    return chunks


# ─────────────────────────────────────────────────────────────
# STEP 3B: Sliding window chunking (SECONDARY strategy)
# ─────────────────────────────────────────────────────────────

def chunk_by_time_window(
    transcript: list[dict],
    episode_meta: dict,
    window_seconds: int = 60,
    step_seconds:   int = 30,   # overlap: new window every 30s
) -> list[dict]:
    """
    Creates overlapping 60-second windows over the raw transcript lines.
    
    WHY OVERLAPPING?
    A question's answer might span the boundary between two windows.
    With 50% overlap (window=60, step=30), every moment appears in 2 windows,
    so we never miss an answer that straddles a boundary.
    
    Example with window=60, step=30:
        Window 1: lines from 0s   → 60s
        Window 2: lines from 30s  → 90s
        Window 3: lines from 60s  → 120s
        ...
    
    Each chunk looks like:
    {
        "chunk_type":    "window",
        "text":          "Alexey: Do you use A/B tests? Stefan: Yes, a lot...",
        "start_seconds": 60,
        "timestamp":     "1:00",
        "deep_link":     "https://youtube.com/watch?v=xxx&t=60",
        ...
    }
    """
    # First, collect only actual transcript lines (skip headers)
    lines = [item for item in transcript if "line" in item]
    
    if not lines:
        return []

    video_id   = episode_meta.get("ids", {}).get("youtube", "")
    base_url   = f"https://www.youtube.com/watch?v={video_id}"
    
    chunks     = []
    total_secs = lines[-1].get("sec", 0)
    
    # Slide the window from 0 to end of episode
    window_start = 0
    while window_start <= total_secs:
        window_end = window_start + window_seconds
        
        # Collect lines that fall inside this window
        window_lines = [
            l for l in lines
            if window_start <= l.get("sec", 0) < window_end
        ]
        
        if window_lines:
            text_parts    = []
            speakers_seen = set()
            
            for entry in window_lines:
                speaker = entry.get("who", "")
                line    = entry.get("line", "").strip()
                if speaker:
                    text_parts.append(f"{speaker}: {line}")
                    speakers_seen.add(speaker)
                else:
                    text_parts.append(line)
            
            chunks.append({
                "chunk_type":    "window",
                "section":       f"~{seconds_to_timestamp(window_start)}",
                "text":          " ".join(text_parts),
                "speakers":      ", ".join(sorted(speakers_seen)),
                
                "episode_title": episode_meta.get("title", ""),
                "short_title":   episode_meta.get("short", ""),
                "video_id":      video_id,
                "season":        str(episode_meta.get("season", "")),
                "episode_num":   str(episode_meta.get("episode", "")),
                
                "start_seconds": window_start,
                "timestamp":     seconds_to_timestamp(window_start),
                "deep_link":     f"{base_url}&t={window_start}",
            })
        
        window_start += step_seconds
    
    return chunks


# ─────────────────────────────────────────────────────────────
# STEP 4: Parse one .md file → all chunks
# ─────────────────────────────────────────────────────────────

def parse_md_file(content: str, filename: str = "unknown.md") -> list[dict]:
    """
    Full pipeline for one .md file:
      1. Split frontmatter
      2. Build clip lookup
      3. Create header chunks  (topic-level)
      4. Create window chunks  (fine-grained)
      5. Return all combined
    
    Args:
        content: Raw markdown content (from file or GitHub API)
        filename: Name of the file for logging purposes
    
    Returns:
        List of all segments ready for minsearch indexing
    """
    print(f"  📄 Parsing: {filename}")
    
    # 1. Parse the YAML frontmatter from content
    data, _ = split_frontmatter(content)
    
    transcript      = data.get("transcript", [])
    quotable_clips  = data.get("quotableClips", [])
    
    if not transcript:
        print(f"     ⚠️  No transcript found in {filename}")
        return []
    
    # 2. Build clip lookup for fast header → deep_link mapping
    clip_lookup = build_clip_lookup(quotable_clips)
    
    # 3. Header-based chunks (best for topic questions)
    header_chunks = chunk_by_headers(transcript, clip_lookup, data)
    
    # 4. Sliding window chunks (best for detail questions)
    window_chunks = chunk_by_time_window(transcript, data, window_seconds=60, step_seconds=30)
    
    all_chunks = header_chunks + window_chunks
    print(f"     ✅ {len(header_chunks)} header chunks + {len(window_chunks)} window chunks = {len(all_chunks)} total")
    
    return all_chunks


# ─────────────────────────────────────────────────────────────
# STEP 5A: Load from GitHub
# ─────────────────────────────────────────────────────────────

def load_all_episodes_from_github(
    repo_owner: str = "DataTalksClub",
    repo_name: str = "datatalksclub.github.io",
    branch: str = "main",
    podcast_path: str = "_podcast"
) -> list[dict]:
    """
    Fetches all .md podcast files from a GitHub repository.
    
    Uses the GitHub API to list files, then downloads each .md file
    and parses it using frontmatter.
    
    Args:
        repo_owner: GitHub repository owner (default: DataTalksClub)
        repo_name: GitHub repository name (default: datatalksclub.github.io)
        branch: Branch to fetch from (default: main)
        podcast_path: Path to podcast files within repo (default: _podcast)
    
    Returns:
        Flat list of ALL segments from ALL episodes,
        ready to be passed to build_index()
    """
    github_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{podcast_path}?ref={branch}"
    raw_github_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{podcast_path}"
    
    print(f"🌐 Fetching episodes from GitHub: {repo_owner}/{repo_name}/{branch}/{podcast_path}")
    
    try:
        # Fetch list of files from GitHub API
        response = requests.get(github_api_url, timeout=10)
        response.raise_for_status()
        files = response.json()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch file list from GitHub: {e}")
    
    # Filter for .md files (skip _template.md and other special files)
    SKIP_FILES = {"_template.md", "README.md"}
    md_files = [f for f in files if f["name"].endswith(".md") and f["name"] not in SKIP_FILES]
    md_files.sort(key=lambda f: f["name"])
    
    if not md_files:
        raise ValueError(f"No .md files found in {github_api_url}")
    
    print(f"📦 Found {len(md_files)} episode file(s)")
    
    all_segments = []
    failed = []
    
    for file_info in md_files:
        filename = file_info["name"]
        file_url = f"{raw_github_url}/{filename}"
        
        try:
            # Download the file content
            response = requests.get(file_url, timeout=10)
            response.raise_for_status()
            content = response.text
            
            # Parse the file
            segments = parse_md_file(content, filename=filename)
            all_segments.extend(segments)
        except Exception as e:
            print(f"     ❌ Failed to fetch/parse {filename}: {e}")
            failed.append(filename)
    
    print(f"\n📊 Total segments loaded: {len(all_segments)}")
    if failed:
        print(f"⚠️  Failed files: {', '.join(failed)}")
    
    return all_segments


# ─────────────────────────────────────────────────────────────
# STEP 5B: Load from local directory (legacy support)
# ─────────────────────────────────────────────────────────────

def load_all_episodes_from_local(episodes_dir: str = "./episodes") -> list[dict]:
    """
    Scans a local directory for all .md files and parses each one.
    
    Directory structure expected:
        episodes/
          ai-in-healthcare.md
          another-episode.md
          ...
    
    Returns:
        Flat list of ALL segments from ALL episodes,
        ready to be passed to build_index()
    """
    episodes_path = Path(episodes_dir)
    
    if not episodes_path.exists():
        raise FileNotFoundError(
            f"Episodes directory not found: {episodes_dir}\n"
            f"Create it and put your .md files inside."
        )
    
    SKIP_FILES = {"_template.md"}  # add more here if needed

    md_files = sorted(
        f for f in episodes_path.glob("**/*.md")
        if f.name not in SKIP_FILES
    )
    
    if not md_files:
        raise ValueError(f"No .md files found in {episodes_dir}")
    
    print(f"🗂️  Found {len(md_files)} episode file(s) in '{episodes_dir}'")
    
    all_segments = []
    failed       = []
    
    for md_file in md_files:
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            segments = parse_md_file(content, filename=md_file.name)
            all_segments.extend(segments)
        except Exception as e:
            print(f"     ❌ Failed to parse {md_file.name}: {e}")
            failed.append(md_file.name)
    
    print(f"\n📊 Total segments loaded: {len(all_segments)}")
    if failed:
        print(f"⚠️  Failed files: {', '.join(failed)}")
    
    return all_segments


# ─────────────────────────────────────────────────────────────
# STEP 5C: Main load function (auto-detects or uses GitHub by default)
# ─────────────────────────────────────────────────────────────

def _local_episodes_md_files(episodes_dir: str, skip_files: set[str]) -> list[Path]:
    """Return a list of markdown files present in the local episodes directory."""
    episodes_path = Path(episodes_dir)
    if not episodes_path.exists():
        return []

    return sorted(
        f for f in episodes_path.glob("**/*.md")
        if f.name not in skip_files
    )


def _download_episodes_to_local(
    episodes_dir: str,
    repo_owner: str = "DataTalksClub",
    repo_name: str = "datatalksclub.github.io",
    branch: str = "main",
    podcast_path: str = "_podcast",
) -> None:
    """Download podcast .md files from GitHub into a local directory.

    This is used for the first run of the app, so that later runs can
    load transcripts from disk without hitting GitHub.
    """
    from pathlib import Path

    episodes_path = Path(episodes_dir)
    episodes_path.mkdir(parents=True, exist_ok=True)

    github_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{podcast_path}?ref={branch}"
    raw_github_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{podcast_path}"

    print(f"🌐 First run: downloading episodes from GitHub to '{episodes_dir}'")

    try:
        response = requests.get(github_api_url, timeout=10)
        response.raise_for_status()
        files = response.json()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch file list from GitHub: {e}")

    SKIP_FILES = {"_template.md", "README.md"}
    md_files = [f for f in files if f["name"].endswith(".md") and f["name"] not in SKIP_FILES]
    md_files.sort(key=lambda f: f["name"])

    if not md_files:
        raise ValueError(f"No .md files found in {github_api_url}")

    failed = []
    for file_info in md_files:
        filename = file_info["name"]
        file_url = f"{raw_github_url}/{filename}"
        local_path = episodes_path / filename

        try:
            response = requests.get(file_url, timeout=10)
            response.raise_for_status()
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(response.text)
        except Exception as e:
            print(f"     ❌ Failed to download {filename}: {e}")
            failed.append(filename)

    if failed:
        raise ValueError(f"Failed to download files: {', '.join(failed)}")


def load_all_episodes(episodes_dir: str = None) -> list[dict]:
    """Load all episodes from either GitHub (default) or local directory.

    By default, loads from GitHub repository.
    If episodes_dir is provided, loads from local directory instead.

    On first run, if the local directory does not exist or contains no markdown
    files, we automatically download the transcripts from GitHub into that
    directory and then load from disk.

    Args:
        episodes_dir: Path to local episodes directory. If None, loads from GitHub.

    Returns:
        List of all segments ready for indexing
    """
    if episodes_dir:
        # Make sure there is at least one .md file locally.
        SKIP_FILES = {"_template.md"}
        md_files = _local_episodes_md_files(episodes_dir, SKIP_FILES)
        if not md_files:
            # First run: fetch from GitHub and save locally
            _download_episodes_to_local(episodes_dir)

        # Load from local directory after ensuring files exist
        return load_all_episodes_from_local(episodes_dir)
    else:
        # Load from GitHub (default)
        return load_all_episodes_from_github()
