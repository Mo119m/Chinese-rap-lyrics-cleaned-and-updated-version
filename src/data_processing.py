# -*- coding: utf-8 -*-
"""
Data Processing Module
Handles parsing and chunking of Chinese rap lyrics
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process raw lyrics into structured chunks"""
    
    def __init__(self, data_dir: Path):
        """
        Initialize data processor
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.raw_txt = self.data_dir / "all_lyrics.txt"
        self.master_csv = self.data_dir / "lyrics_master.csv"
        self.chunks_csv = self.data_dir / "lyrics_chunks_enriched.csv"
        
        # Metadata keywords for filtering
        self.meta_keys = {
            "作词", "作曲", "编曲", "制作", "发行", "录音", "配唱", 
            "混音", "母带", "混音师", "母带工程", "女声", "男声", 
            "和声", "吉他", "贝斯", "鼓", "鼓手", "萨克斯", "管风琴", 
            "钢琴", "弦乐", "Mastered", "Mixed", "Producer", "Vocal"
        }
    
    def is_artist_line(self, s: str) -> bool:
        """Check if line is artist declaration"""
        return s.startswith("歌手姓名：") or s.startswith("歌手姓名:")
    
    def is_zuoci_line(self, s: str) -> bool:
        """Check if line is lyrics credit line"""
        return re.match(r"^\s*作词\s*[：:]", s) is not None
    
    def is_meta_line(self, s: str) -> bool:
        """Check if line contains metadata"""
        if re.search(r"[：:]", s):
            k = re.split(r"[：:]", s, 1)[0].strip()
            if any(k_.lower() in k.lower() for k_ in self.meta_keys):
                return True
        if re.search(r"\b(mixed|mastered)\s+by\b", s, flags=re.IGNORECASE):
            return True
        return False
    
    def parse_lyrics(self) -> pd.DataFrame:
        """
        Parse raw lyrics file into structured format
        
        Returns:
            DataFrame with parsed lyrics
        """
        logger.info(f"Parsing lyrics from {self.raw_txt}")
        
        if not self.raw_txt.exists():
            raise FileNotFoundError(f"Raw lyrics file not found: {self.raw_txt}")
        
        raw = self.raw_txt.read_text(encoding="utf-8", errors="ignore")
        lines = [ln.rstrip("\n\r") for ln in raw.splitlines()]
        
        songs = []
        artist = None
        song = None
        last_nonblank = ""
        song_idx = 0
        
        def flush():
            nonlocal song, song_idx
            if song and song.get("lyrics", "").strip():
                song["lyrics"] = song["lyrics"].strip()
                songs.append(song.copy())
                song = None
                song_idx += 1
        
        for ln in lines:
            t = ln.strip()
            
            # Empty line - preserve in lyrics
            if t == "":
                if song is not None:
                    song["lyrics"] += "\n"
                continue
            
            # New artist
            if self.is_artist_line(t):
                flush()
                song_idx = 0
                artist = t.split("：", 1)[1] if "：" in t else t.split(":", 1)[1]
                artist = artist.strip()
                last_nonblank = ""
                continue
            
            # Song start - title is previous non-blank line
            if artist and self.is_zuoci_line(t):
                flush()
                title = last_nonblank.strip()
                if not title or self.is_artist_line(title) or self.is_meta_line(title):
                    title = f"未命名-{artist}-{song_idx:03d}"
                song = {
                    "artist": artist,
                    "song_title": title,
                    "meta": {},
                    "lyrics": ""
                }
                # Add this zuoci line to meta
                k, v = re.split(r"[：:]", t, 1)
                song["meta"][k.strip()] = v.strip()
                continue
            
            # Metadata line
            if song is not None and self.is_meta_line(t):
                if re.search(r"[：:]", t):
                    k, v = re.split(r"[：:]", t, 1)
                    k, v = k.strip(), v.strip()
                    song["meta"][k] = (
                        song["meta"].get(k, "") + 
                        (" / " if k in song["meta"] else "") + 
                        v
                    )
                else:
                    song["meta"][t] = True
                last_nonblank = t
                continue
            
            # Regular lyrics line
            if song is not None:
                song["lyrics"] += t + "\n"
            
            # Update last non-blank
            if not (self.is_artist_line(t) or self.is_meta_line(t) or 
                    self.is_zuoci_line(t)):
                last_nonblank = t
        
        flush()
        
        # Convert to DataFrame
        rows = []
        for i, s in enumerate(songs):
            rows.append({
                "artist": s["artist"],
                "song_id": f"{s['artist']}_{i:04d}",
                "song_title": s["song_title"],
                "meta_json": json.dumps(s.get("meta", {}), ensure_ascii=False),
                "lyrics": s["lyrics"]
            })
        
        df_master = pd.DataFrame(rows)
        df_master.to_csv(self.master_csv, index=False)
        
        logger.info(f"Saved {len(df_master)} songs to {self.master_csv}")
        return df_master
    
    def split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        return [
            p.strip() 
            for p in re.split(r"(?:\n\s*\n)+", str(text).strip()) 
            if p.strip()
        ]
    
    def create_chunks(
        self, 
        min_chars: int = 15,
        max_chars: int = 1500,
        overlap: int = 100
    ) -> pd.DataFrame:
        """
        Create text chunks from parsed lyrics
        
        Args:
            min_chars: Minimum chunk length
            max_chars: Maximum chunk length
            overlap: Overlap between consecutive chunks
        
        Returns:
            DataFrame with text chunks
        """
        logger.info("Creating text chunks")
        
        if not self.master_csv.exists():
            raise FileNotFoundError(
                f"Master CSV not found: {self.master_csv}. Run parse_lyrics() first."
            )
        
        df_master = pd.read_csv(self.master_csv)
        
        def chunk_long(t: str):
            """Split long text with overlap"""
            if len(t) <= max_chars:
                return [t]
            out, i = [], 0
            while i < len(t):
                j = min(len(t), i + max_chars)
                out.append(t[i:j].strip())
                if j == len(t):
                    break
                i = max(0, j - overlap)
            return [x for x in out if x]
        
        chunks = []
        for _, r in df_master.iterrows():
            cid = 0
            for para in self.split_paragraphs(r["lyrics"]):
                if len(para) < min_chars:
                    continue
                for sub in chunk_long(para):
                    if len(sub) >= min_chars:
                        chunks.append({
                            "artist": r["artist"],
                            "song_id": r["song_id"],
                            "song_title": r["song_title"],
                            "chunk_id": cid,
                            "text": sub
                        })
                        cid += 1
        
        df_chunks = pd.DataFrame(chunks)
        df_chunks.to_csv(self.chunks_csv, index=False)
        
        logger.info(f"Created {len(df_chunks)} chunks -> {self.chunks_csv}")
        return df_chunks
