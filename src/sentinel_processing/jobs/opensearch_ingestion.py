#!/usr/bin/env python3
"""
OpenSearch Ingestion Job - Ingest changes table into OpenSearch KNN index
"""
import time
import json
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from pyspark.sql.functions import col, expr
from sentinel_processing.config import CONFIG, get_ssm_parameter
from sentinel_processing.lib.sedona_utils import create_sedona_session

def create_opensearch_index(opensearch_host, index_name, region):
    """Create OpenSearch index with KNN and geo mappings"""
    
    # Set up OpenSearch client with SigV4 auth
    host = opensearch_host.replace('https://', '')
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, 'es')
    
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )
    
    index_mapping = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 3,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "scene_id": {"type": "keyword"},
                "@timestamp": {"type": "date", "format": "epoch_millis"},
                "geohash": {"type": "keyword"},
                "scl_mean": {"type": "float"},
                "change_score": {"type": "float"},
                "model_version": {"type": "keyword"},
                "created_at": {"type": "date", "format": "epoch_millis"},
                "geometry": {"type": "geo_shape"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene"
                    }
                }
            }
        }
    }
    
    # Delete existing index
    try:
        client.indices.delete(index=index_name)
        print(f"Deleted existing index: {index_name}")
    except:
        pass
    
    # Create new index
    response = client.indices.create(
        index=index_name,
        body=index_mapping
    )
    
    print(f"Created OpenSearch index: {index_name}")
    print(response)



def main():
    start_time = time.time()
    
    spark = create_sedona_session("OpenSearchIngestion")
    
    # Configuration
    input_table = CONFIG.jobs.change_detection.output_table
    database_name = CONFIG.sentinel.database_name
    opensearch_host = get_ssm_parameter("/sentinel/opensearch/endpoint")
    index_name = CONFIG.jobs.opensearch_ingestion.index_name
    region = boto3.Session().region_name
    
    try:
        # Create OpenSearch index with mappings
        create_opensearch_index(opensearch_host, index_name, region)
        
        # Join chips with embeddings and changes for OpenSearch ingestion
        chips_df = spark.read.format("iceberg").load(f"{database_name}.{CONFIG.jobs.chip_extraction.output_table}").drop("clay_tensor", "geotiff", "created_at")
        embeddings_df = spark.read.format("iceberg").load(f"{database_name}.{CONFIG.jobs.embedding_generation.output_table}").drop("created_at")
        changes_df = spark.read.format("iceberg").load(f"{database_name}.{input_table}")
        
        full_df = chips_df \
            .join(embeddings_df, ["id", "geohash", "datetime"]) \
            .join(changes_df, ["id", "geohash", "datetime"])
        
        total_changes = changes_df.count()
        print(f"Ingesting {total_changes:,} change records to OpenSearch")
        
        # Transform for OpenSearch ingestion
        opensearch_df = full_df \
            .withColumnRenamed("datetime", "@timestamp") \
            .withColumn("geometry", expr("from_json(ST_AsGeoJSON(ST_GeomFromWKB(geometry)), 'struct<type:string,coordinates:array<array<array<double>>>>')"))
        
        # OpenSearch-Hadoop configuration
        opensearch_df.write \
            .format("org.opensearch.spark.sql") \
            .option("opensearch.nodes", opensearch_host) \
            .option("opensearch.aws.sigv4.region", region) \
            .option("opensearch.resource", f"{index_name}") \
            .mode("append") \
            .save()
        
        print(f"OpenSearch ingestion completed in {(time.time() - start_time) / 60:.1f}m")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()