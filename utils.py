#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script for common analysis tasks
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize


def similarity_search(data_dir, query_type='song', query_id=0, top_n=10):
    """
    Perform similarity search
    
    Args:
        data_dir: Data directory
        query_type: 'song' or 'artist'
        query_id: Index or name of query
        top_n: Number of results
    """
    data_dir = Path(data_dir)
    
    if query_type == 'song':
        meta = pd.read_csv(data_dir / "songs_meta_v2.csv")
        vecs = normalize(np.load(data_dir / "song_vectors_v2.npy"))
    else:
        meta = pd.read_csv(data_dir / "artists_meta_v2.csv")
        vecs = normalize(np.load(data_dir / "artist_vectors_v2.npy"))
    
    # Handle string query
    if isinstance(query_id, str):
        if query_type == 'song':
            mask = meta['song_title'].str.contains(query_id, case=False)
        else:
            mask = meta['artist'].str.contains(query_id, case=False)
        
        if mask.sum() == 0:
            print(f"No {query_type} found matching '{query_id}'")
            return
        
        query_idx = mask.idxmax()
        print(f"Using {query_type}: {meta.iloc[query_idx].to_dict()}")
    else:
        query_idx = query_id
    
    # Compute similarities
    sims = vecs @ vecs[query_idx]
    order = sims.argsort()[::-1]
    
    # Skip self and get top N
    results = []
    for idx in order:
        if idx == query_idx:
            continue
        results.append(idx)
        if len(results) >= top_n:
            break
    
    # Display results
    result_df = meta.iloc[results].copy()
    result_df['similarity'] = [float(sims[i]) for i in results]
    
    print(f"\nTop {top_n} similar {query_type}s:")
    print(result_df.to_string(index=False))
    
    return result_df


def cluster_summary(data_dir, cluster_id):
    """
    Print summary for a specific cluster
    
    Args:
        data_dir: Data directory
        cluster_id: Cluster ID
    """
    data_dir = Path(data_dir)
    
    # Load data
    df = pd.read_csv(data_dir / "chunks_with_umap_clusters_enriched_v2.csv")
    kw = pd.read_csv(data_dir / "cluster_keywords_v2.csv")
    top = pd.read_csv(data_dir / "cluster_top_artists_v2.csv")
    rep = pd.read_csv(data_dir / "cluster_representatives_v2.csv")
    
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id} SUMMARY")
    print(f"{'='*80}\n")
    
    # Basic stats
    cluster_df = df[df['cluster'] == cluster_id]
    print(f"Size: {len(cluster_df)} chunks")
    print(f"Artists: {cluster_df['artist'].nunique()}")
    print(f"Songs: {cluster_df['song_id'].nunique()}\n")
    
    # Keywords
    cluster_kw = kw[kw['cluster'] == cluster_id].head(15)
    print("Top Keywords:")
    for _, row in cluster_kw.iterrows():
        print(f"  {row['rank']}. {row['term']} (score: {row['score']:.3f})")
    print()
    
    # Top artists
    cluster_top = top[top['cluster'] == cluster_id].head(8)
    print("Top Artists:")
    for _, row in cluster_top.iterrows():
        print(f"  {row['artist']}: {row['percentage']:.1f}%")
    print()
    
    # Representative sample
    cluster_rep = rep[rep['cluster'] == cluster_id].head(3)
    print("Representative Samples:")
    for _, row in cluster_rep.iterrows():
        preview = str(row['text']).replace('\n', ' ')[:150]
        print(f"  {row['artist']} - {row['song_title']}")
        print(f"  \"{preview}...\"")
        print(f"  (similarity: {row['cos_sim']:.3f})\n")


def export_cluster(data_dir, cluster_id, output_file):
    """
    Export all chunks from a cluster to CSV
    
    Args:
        data_dir: Data directory
        cluster_id: Cluster ID
        output_file: Output CSV path
    """
    data_dir = Path(data_dir)
    
    df = pd.read_csv(data_dir / "chunks_with_umap_clusters_enriched_v2.csv")
    cluster_df = df[df['cluster'] == cluster_id]
    
    cluster_df.to_csv(output_file, index=False)
    print(f"Exported {len(cluster_df)} chunks from cluster {cluster_id} to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Utility tools for lyrics analysis')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Similarity search
    sim_parser = subparsers.add_parser('search', help='Similarity search')
    sim_parser.add_argument('--data-dir', required=True, help='Data directory')
    sim_parser.add_argument('--type', choices=['song', 'artist'], default='song')
    sim_parser.add_argument('--query', required=True, help='Query ID or name')
    sim_parser.add_argument('--top-n', type=int, default=10, help='Number of results')
    
    # Cluster summary
    sum_parser = subparsers.add_parser('summary', help='Cluster summary')
    sum_parser.add_argument('--data-dir', required=True, help='Data directory')
    sum_parser.add_argument('--cluster', type=int, required=True, help='Cluster ID')
    
    # Export cluster
    exp_parser = subparsers.add_parser('export', help='Export cluster')
    exp_parser.add_argument('--data-dir', required=True, help='Data directory')
    exp_parser.add_argument('--cluster', type=int, required=True, help='Cluster ID')
    exp_parser.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    if args.command == 'search':
        # Try to parse as integer, otherwise use as string
        try:
            query = int(args.query)
        except ValueError:
            query = args.query
        
        similarity_search(
            args.data_dir,
            query_type=args.type,
            query_id=query,
            top_n=args.top_n
        )
    
    elif args.command == 'summary':
        cluster_summary(args.data_dir, args.cluster)
    
    elif args.command == 'export':
        export_cluster(args.data_dir, args.cluster, args.output)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
