#!/usr/bin/env python3
"""
OpenSearch Ingestion Job - Ingest changes table into OpenSearch KNN index
"""
import time
import json
import requests
import boto3
from pyspark.sql.functions import col, expr
from sentinel_processing.config import CONFIG, get_ssm_parameter
from sentinel_processing.lib.sedona_utils import create_sedona_session

def create_opensearch_index(opensearch_host, index_name):
    """Create OpenSearch index with KNN and geo mappings"""
    
    index_mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            }
        },
        "mappings": {
            "properties": {
                "chip_id": {"type": "keyword"},
                "scene_id": {"type": "keyword"},
                "@timestamp": {"type": "date"},
                "geohash": {"type": "keyword"},
                "change_score": {"type": "float"},
                "model_version": {"type": "keyword"},
                "created_at": {"type": "date"},
                "location": {"type": "geo_shape"},
                "embedding_vector": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib"
                    }
                }
            }
        }
    }
    
    # Delete existing index
    try:
        requests.delete(f"{opensearch_host}/{index_name}")
        print(f"Deleted existing index: {index_name}")
    except:
        pass
    
    # Create new index
    response = requests.put(
        f"{opensearch_host}/{index_name}",
        headers={"Content-Type": "application/json"},
        data=json.dumps(index_mapping)
    )
    
    if response.status_code in [200, 201]:
        print(f"Created OpenSearch index: {index_name}")
    else:
        raise Exception(f"Failed to create index: {response.text}")



def main():
    start_time = time.time()
    
    spark = create_sedona_session("OpenSearchIngestion")
    
    # Configuration
    input_table = CONFIG.jobs.change_detection.output_table
    opensearch_host = get_ssm_parameter("/sentinel/opensearch/endpoint")
    index_name = CONFIG.jobs.opensearch_ingestion.index_name
    region = boto3.Session().region_name
    
    try:
        # Create OpenSearch index with mappings
        create_opensearch_index(opensearch_host, index_name)
        
        # Read changes from Iceberg table
        changes_df = spark.read.format("iceberg").load(f"sentinel.{input_table}")
        
        total_changes = changes_df.count()
        print(f"Ingesting {total_changes:,} change records to OpenSearch")
        
        # Transform for OpenSearch ingestion
        opensearch_df = changes_df.select(
            col("id").alias("chip_id"),
            col("scene_id"),
            col("datetime").cast("timestamp").alias("@timestamp"),
            col("geohash"),
            col("cosine_similarity_to_previous").alias("change_score"),
            col("model_version"),
            col("created_at"),
            # Convert WKB geometry to GeoJSON string (OpenSearch-Hadoop handles parsing)
            expr("ST_AsGeoJSON(geometry)").alias("location"),
            # Embedding vector for KNN search
            col("global_embedding").alias("embedding_vector")
        )  # Include all records (baseline has change_score=0.0)
        
        # OpenSearch-Hadoop configuration from spark config
        opensearch_df.write \
            .format("org.opensearch.spark.sql") \
            .option("opensearch.nodes", opensearch_host) \
            .option("opensearch.aws.sigv4.region", region) \
            .option("opensearch.resource", f"{index_name}/_doc") \
            .option("opensearch.mapping.id", "chip_id") \
            .mode("append") \
            .save()
        
        print(f"OpenSearch ingestion completed in {(time.time() - start_time) / 60:.1f}m")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()