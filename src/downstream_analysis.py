# -*- coding: utf-8 -*-
"""
Downstream Analysis Module
Performs cluster analysis, keyword extraction, and similarity search
"""

import logging
import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class DownstreamAnalyzer:
    """Downstream analysis tasks"""
    
    def __init__(
        self,
        data_dir: Path,
        top_artists: int = 10,
        representative_samples: int = 20,
        top_keywords: int = 30
    ):
        """
        Initialize downstream analyzer
        
        Args:
            data_dir: Directory containing data files
            top_artists: Number of top artists per cluster
            representative_samples: Number of representative samples per cluster
            top_keywords: Number of top keywords per cluster
        """
        self.data_dir = Path(data_dir)
        self.top_artists = top_artists
        self.representative_samples = representative_samples
        self.top_keywords = top_keywords
        
        self.merged_csv = self.data_dir / "chunks_with_umap_clusters_enriched_v2.csv"
        self.contrast_npy = self.data_dir / "Z_contrastive_v2.npy"
        self.emb_npy = self.data_dir / "qwen15b_embeddings_v2.npy"
        
        # Load stopwords
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """Load Chinese stopwords"""
        stop_file = self.data_dir / "stopwords-zh.txt"
        stopwords = set()
        
        if stop_file.exists():
            stopwords = {
                w.strip() 
                for w in open(stop_file, encoding="utf-8") 
                if w.strip()
            }
        
        # Add common filler words
        extra_stop = {
            "的", "了", "是", "在", "就", "都", "和", "也", "与", "而", "及",
            "还有", "一个", "没有", "我们", "你们", "他们", "什么", "怎么",
            "因为", "然后", "啦", "啊", "呀", "嘛", "吧", "呢", "哦", "喔",
            "啊", "嗯", "呜", "啦啦"
        }
        stopwords |= extra_stop
        
        return stopwords
    
    def analyze_top_artists(self) -> pd.DataFrame:
        """
        Find top artists in each cluster
        
        Returns:
            DataFrame with top artists per cluster
        """
        logger.info("Analyzing top artists per cluster")
        
        df = pd.read_csv(self.merged_csv)
        df["cluster"] = df["cluster"].astype(int)
        
        rows = []
        for c, sub in df.groupby("cluster"):
            pct = (sub["artist"].value_counts(normalize=True) * 100).round(2)
            for art, p in pct.head(self.top_artists).items():
                rows.append({
                    "cluster": int(c),
                    "artist": art,
                    "percentage": float(p)
                })
        
        df_top = pd.DataFrame(rows)
        output = self.data_dir / "cluster_top_artists_v2.csv"
        df_top.to_csv(output, index=False)
        
        logger.info(f"Saved top artists: {output}")
        return df_top
    
    def find_representatives(self) -> pd.DataFrame:
        """
        Find representative samples for each cluster
        
        Returns:
            DataFrame with representative samples
        """
        logger.info("Finding representative samples per cluster")
        
        df = pd.read_csv(self.merged_csv)
        
        # Load embeddings
        if self.contrast_npy.exists():
            Z = np.load(self.contrast_npy).astype("float32")
        else:
            Z = np.load(self.emb_npy).astype("float32")
        
        Z = normalize(Z)
        
        out = []
        for c, idx in df.reset_index().groupby("cluster")["index"]:
            idx = idx.values
            Zc = Z[idx]
            
            # Compute centroid
            centroid = normalize(Zc.mean(axis=0, keepdims=True))
            
            # Find closest samples
            sims = (Zc @ centroid.T).ravel()
            order = np.argsort(-sims)[:self.representative_samples]
            
            for rank, i in enumerate(idx[order], 1):
                r = df.iloc[i]
                out.append({
                    "cluster": int(c),
                    "rank": rank,
                    "cos_sim": float(sims[order][rank - 1]),
                    "artist": r["artist"],
                    "song_title": r["song_title"],
                    "text": r["text"]
                })
        
        df_rep = pd.DataFrame(out)
        output = self.data_dir / "cluster_representatives_v2.csv"
        df_rep.to_csv(output, index=False)
        
        logger.info(f"Saved representatives: {output}")
        return df_rep
    
    def zh_tokenize(self, s: str) -> List[str]:
        """Tokenize Chinese text"""
        s = re.sub(r"\s+", "", str(s))
        toks = [w for w in jieba.lcut(s, HMM=True) if w]
        
        # Whitelist for single characters
        single_char_whitelist = {
            "爱", "梦", "心", "夜", "雨", "花", "海", "光", "路", "风", "歌", "酒"
        }
        
        out = []
        for w in toks:
            # Filter stopwords, numbers, punctuation
            if w in self.stopwords:
                continue
            if re.fullmatch(r"[0-9A-Za-z]+", w):
                continue
            if re.fullmatch(r"\W+", w):
                continue
            # Filter most single characters
            if len(w) == 1 and w not in single_char_whitelist:
                continue
            out.append(w)
        
        return out
    
    def extract_keywords(self) -> pd.DataFrame:
        """
        Extract keywords using c-TF-IDF
        
        Returns:
            DataFrame with keywords per cluster
        """
        logger.info("Extracting keywords per cluster")
        
        df = pd.read_csv(self.merged_csv)
        
        # Merge all texts per cluster
        docs_by_cluster = (
            df.groupby("cluster")["text"]
            .apply(lambda x: "\n".join(x.astype(str)))
            .reset_index()
        )
        
        clusters = docs_by_cluster["cluster"].astype(int).tolist()
        docs = docs_by_cluster["text"].tolist()
        C = len(docs)
        
        # Count vectorizer for c-TF-IDF
        vect = CountVectorizer(
            tokenizer=self.zh_tokenize,
            ngram_range=(1, 2),
            min_df=2
        )
        
        X = vect.fit_transform(docs)
        terms = np.array(vect.get_feature_names_out())
        T = X.toarray().astype("float32")
        
        # c-TF-IDF
        tf = T / (T.sum(axis=1, keepdims=True) + 1e-9)
        df_c = (T > 0).sum(axis=0)
        idf = np.log((C + 1) / (df_c + 1)) + 1.0
        ctfidf = tf * idf
        
        # Filter terms appearing in too many clusters
        df_ratio = df_c / C
        mask = df_ratio <= 0.6
        ctfidf = ctfidf[:, mask]
        terms = terms[mask]
        
        # Extract top keywords per cluster
        rows = []
        for ci in range(C):
            row = ctfidf[ci]
            if row.sum() == 0:
                continue
            idx = row.argsort()[::-1][:self.top_keywords]
            for rank, j in enumerate(idx, 1):
                rows.append({
                    "cluster": int(clusters[ci]),
                    "rank": rank,
                    "term": terms[j],
                    "score": float(row[j])
                })
        
        df_kw = pd.DataFrame(rows).sort_values(["cluster", "rank"])
        output = self.data_dir / "cluster_keywords_v2.csv"
        df_kw.to_csv(output, index=False)
        
        logger.info(f"Saved keywords: {output}")
        return df_kw
    
    def analyze_emotions(self) -> pd.DataFrame:
        """
        Analyze emotions in each cluster
        
        Returns:
            DataFrame with emotion analysis
        """
        logger.info("Analyzing emotions per cluster")
        
        df = pd.read_csv(self.merged_csv)
        
        # Emotion lexicon
        emo_lex = {
            "积极": ["快乐", "开心", "喜悦", "幸福", "甜蜜", "希望", "自由", 
                   "灿烂", "笑", "阳光", "温暖", "热爱", "热烈", "勇敢", 
                   "梦想", "胜利", "美好", "美妙"],
            "消极": ["悲伤", "难过", "寂寞", "孤独", "眼泪", "崩溃", "痛苦", 
                   "绝望", "遗憾", "愧疚", "惆怅", "失望", "崩塌", "心碎", "忧郁"],
            "愤怒": ["愤怒", "生气", "怒", "操", "厌烦", "不爽", "火大", 
                   "擦", "去死", "混蛋", "操蛋"],
            "思念": ["想你", "想念", "怀念", "牵挂", "等你", "等候", "相思", 
                   "梦见", "回忆", "记得"],
            "励志": ["坚持", "相信", "勇气", "追逐", "远方", "出发", "不放弃", 
                   "站起来", "明天", "未来", "热血", "奋斗", "梦想"],
            "派对": ["派对", "酒", "狂欢", "舞", "DJ", "夜店", "开车", 
                   "飙车", "嗨", "耶", "yeah", "woo", "woohoo"],
            "城市/意象": ["城市", "街头", "地铁", "北京", "上海", "重庆", 
                       "成都", "霓虹", "夜色", "屋顶", "旅馆", "巷口", 
                       "烟", "雨", "风", "月亮", "星辰", "银河"],
            "爱情": ["爱情", "恋爱", "拥抱", "亲吻", "爱人", "你和我", 
                   "暧昧", "心动", "浪漫", "告白", "分手", "复合"],
        }
        
        stats = []
        for c, sub in df.groupby("cluster"):
            N = len(sub)
            bucket = defaultdict(int)
            
            for t in sub["text"].astype(str):
                txt = re.sub(r"\s+", "", str(t))
                for emo, words in emo_lex.items():
                    for w in words:
                        if w in txt:
                            bucket[emo] += 1
            
            total_hits = sum(bucket.values())
            row = {"cluster": int(c), "chunks": N, "hits": total_hits}
            
            for emo in emo_lex:
                row[f"hit_{emo}"] = bucket[emo]
                row[f"ratio_{emo}"] = bucket[emo] / max(total_hits, 1)
            
            # Suggested label
            top_emo = max(emo_lex.keys(), key=lambda k: row[f"ratio_{k}"])
            row["suggested_label"] = top_emo if total_hits > 0 else "未判定"
            
            stats.append(row)
        
        df_emo = pd.DataFrame(stats).sort_values("cluster")
        output = self.data_dir / "cluster_emotions_v2.csv"
        df_emo.to_csv(output, index=False)
        
        logger.info(f"Saved emotion analysis: {output}")
        return df_emo
    
    def create_song_vectors(self) -> tuple:
        """
        Create song-level vectors by averaging chunks
        
        Returns:
            Tuple of (metadata DataFrame, vectors array)
        """
        logger.info("Creating song-level vectors")
        
        df_chunks = pd.read_csv(self.merged_csv)
        
        # Load embeddings
        if self.contrast_npy.exists():
            Z_chunks = np.load(self.contrast_npy).astype("float32")
        else:
            Z_chunks = np.load(self.emb_npy).astype("float32")
        
        Z_chunks = normalize(Z_chunks)
        
        # Song metadata
        songs_meta = (
            df_chunks.groupby(["artist", "song_id", "song_title"])
            .size()
            .reset_index(name="num_chunks")
        )
        
        # Aggregate vectors
        song_vecs = []
        for _, row in songs_meta.iterrows():
            mask = df_chunks["song_id"] == row["song_id"]
            V = Z_chunks[mask.values]
            v = normalize(V.mean(axis=0, keepdims=True))[0]
            song_vecs.append(v)
        
        Z_songs = np.vstack(song_vecs).astype("float32")
        
        # Save
        songs_meta.to_csv(self.data_dir / "songs_meta_v2.csv", index=False)
        np.save(self.data_dir / "song_vectors_v2.npy", Z_songs)
        
        logger.info(f"Created {len(Z_songs)} song vectors")
        
        return songs_meta, Z_songs
    
    def create_artist_vectors(self) -> tuple:
        """
        Create artist-level vectors by averaging chunks
        
        Returns:
            Tuple of (metadata DataFrame, vectors array)
        """
        logger.info("Creating artist-level vectors")
        
        df_chunks = pd.read_csv(self.merged_csv)
        
        # Load embeddings
        if self.contrast_npy.exists():
            Z_chunks = np.load(self.contrast_npy).astype("float32")
        else:
            Z_chunks = np.load(self.emb_npy).astype("float32")
        
        Z_chunks = normalize(Z_chunks)
        
        # Artist metadata
        artists_meta = (
            df_chunks.groupby(["artist"])
            .size()
            .reset_index(name="num_chunks")
        )
        
        # Aggregate vectors
        artist_vecs = []
        for _, row in artists_meta.iterrows():
            mask = df_chunks["artist"] == row["artist"]
            V = Z_chunks[mask.values]
            v = normalize(V.mean(axis=0, keepdims=True))[0]
            artist_vecs.append(v)
        
        Z_artists = np.vstack(artist_vecs).astype("float32")
        
        # Save
        artists_meta.to_csv(self.data_dir / "artists_meta_v2.csv", index=False)
        np.save(self.data_dir / "artist_vectors_v2.npy", Z_artists)
        
        logger.info(f"Created {len(Z_artists)} artist vectors")
        
        return artists_meta, Z_artists
    
    def generate_report(self) -> str:
        """
        Generate comprehensive cluster report
        
        Returns:
            Report text
        """
        logger.info("Generating cluster report")
        
        df = pd.read_csv(self.merged_csv)
        df["cluster"] = df["cluster"].astype(int)
        
        # Load analysis results
        top_file = self.data_dir / "cluster_top_artists_v2.csv"
        rep_file = self.data_dir / "cluster_representatives_v2.csv"
        kw_file = self.data_dir / "cluster_keywords_v2.csv"
        
        df_top = pd.read_csv(top_file) if top_file.exists() else pd.DataFrame()
        df_rep = pd.read_csv(rep_file) if rep_file.exists() else pd.DataFrame()
        df_kw = pd.read_csv(kw_file) if kw_file.exists() else pd.DataFrame()
        
        lines = ["# Cluster Analysis Report\n"]
        
        for c in sorted(df["cluster"].unique()):
            lines.append(f"## Cluster {c}\n")
            
            # Keywords
            if not df_kw.empty and c in df_kw["cluster"].unique():
                kws = df_kw[df_kw["cluster"] == c].sort_values("rank").head(20)["term"].tolist()
                lines.append("**Keywords:** " + "、".join(kws) + "\n")
            
            # Top artists
            if not df_top.empty and c in df_top["cluster"].unique():
                top = df_top[df_top["cluster"] == c].sort_values("percentage", ascending=False).head(8)
                top_txt = "，".join([f"{r.artist}（{r.percentage:.1f}%）" for r in top.itertuples()])
                lines.append("**Top Artists:** " + top_txt + "\n")
            
            # Representatives
            if not df_rep.empty and c in df_rep["cluster"].unique():
                lines.append("**Representative Samples:**")
                rep = df_rep[df_rep["cluster"] == c].sort_values("rank").head(5)
                for r in rep.itertuples():
                    preview = str(r.text).replace("\n", " ")[:120]
                    lines.append(f"- *{r.artist} · {r.song_title}* — {preview}… (sim={r.cos_sim:.3f})")
                lines.append("")
        
        report = "\n".join(lines)
        output = self.data_dir / "cluster_report_v2.md"
        output.write_text(report, encoding="utf-8")
        
        logger.info(f"Saved report: {output}")
        
        return report
