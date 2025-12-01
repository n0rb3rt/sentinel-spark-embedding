#!/usr/bin/env python3

from aws_cdk import (
    aws_emr as emr,
    aws_ec2 as ec2,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_ssm as ssm,
    aws_iam as iam,
    CfnOutput
)
from constructs import Construct

class EMRConstruct(Construct):
    def __init__(self, scope: Construct, construct_id: str, 
                 vpc_construct, storage_construct):
        super().__init__(scope, construct_id)
        
        # EMR Service Role
        self.service_role = iam.Role(
            self, "ServiceRole",
            assumed_by=iam.ServicePrincipal("elasticmapreduce.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonElasticMapReduceRole")
            ]
        )
        
        # EMR Instance Role
        self.instance_role = iam.Role(
            self, "InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonElasticMapReduceforEC2Role")
            ]
        )
        
        # Instance Profile
        self.instance_profile = iam.CfnInstanceProfile(
            self, "InstanceProfile",
            roles=[self.instance_role.role_name]
        )
        
        # Bootstrap script
        bootstrap_script = \
"""#!/bin/bash
set -ex

# Install Python 3.12 and git (Java 17 already available on EMR 7.9.0)
sudo yum update -y
sudo yum install -y python3.12 python3.12-pip python3.12-devel git

# Mount CGroup v1 for GPU support
sudo mkdir -p /spark-rapids-cgroup/devices
sudo mount -t cgroup -o devices cgroupv1-devices /spark-rapids-cgroup/devices
sudo chmod a+rwx -R /spark-rapids-cgroup
"""
        
        bootstrap_deployment = s3deploy.BucketDeployment(
            self, "BootstrapScript",
            sources=[s3deploy.Source.data("gpu-cgroup-bootstrap.sh", bootstrap_script)],
            destination_bucket=storage_construct.bucket,
            destination_key_prefix="dist/bootstrap/"
        )
        
        # EMR Cluster
        self.cluster = emr.CfnCluster(
            self, "Cluster",
            name="sentinel-embedding-gpu",
            release_label="emr-7.9.0",
            service_role=self.service_role.role_arn,
            job_flow_role=self.instance_profile.ref,
            log_uri=f"s3://{storage_construct.bucket.bucket_name}/logs/",
            ebs_root_volume_size=50,
            instances=emr.CfnCluster.JobFlowInstancesConfigProperty(
                master_instance_group=emr.CfnCluster.InstanceGroupConfigProperty(
                    instance_count=1,
                    instance_type="g4dn.xlarge"
                ),
                core_instance_group=emr.CfnCluster.InstanceGroupConfigProperty(
                    instance_count=2,
                    instance_type="g4dn.xlarge"
                ),
                ec2_subnet_id=vpc_construct.vpc.private_subnets[0].subnet_id,
                additional_master_security_groups=[vpc_construct.emr_security_group.security_group_id],
                additional_slave_security_groups=[vpc_construct.emr_security_group.security_group_id]
            ),
            applications=[
                emr.CfnCluster.ApplicationProperty(name="Spark"),
                emr.CfnCluster.ApplicationProperty(name="Hadoop")
            ],
            configurations=[
                # Java 17 configuration for Hadoop
                emr.CfnCluster.ConfigurationProperty(
                    classification="hadoop-env",
                    configurations=[
                        emr.CfnCluster.ConfigurationProperty(
                            classification="export",
                            configuration_properties={
                                "JAVA_HOME": "/usr/lib/jvm/java-17-amazon-corretto.x86_64"
                            }
                        )
                    ]
                ),
                # Java 17 configuration for Spark
                emr.CfnCluster.ConfigurationProperty(
                    classification="spark-env",
                    configurations=[
                        emr.CfnCluster.ConfigurationProperty(
                            classification="export",
                            configuration_properties={
                                "JAVA_HOME": "/usr/lib/jvm/java-17-amazon-corretto.x86_64"
                            }
                        )
                    ]
                ),
                # Enable Spark Rapids (required)
                emr.CfnCluster.ConfigurationProperty(
                    classification="spark",
                    configuration_properties={"enableSparkRapids": "true"}
                ),
                # Iceberg configuration
                emr.CfnCluster.ConfigurationProperty(
                    classification="iceberg-defaults",
                    configuration_properties={"iceberg.enabled": "true"}
                ),
                # YARN configuration 
                emr.CfnCluster.ConfigurationProperty(
                    classification="yarn-site",
                    configuration_properties={
                        # Log aggregation 
                        "yarn.log-aggregation-enable": "true",
                        "yarn.log-aggregation.retain-seconds": "-1",
                        "yarn.nodemanager.remote-app-log-dir": f"s3://{storage_construct.bucket.bucket_name}/logs",
                        # GPU configuration 
                        "yarn.nodemanager.resource-plugins": "yarn.io/gpu",
                        "yarn.resource-types": "yarn.io/gpu",
                        "yarn.nodemanager.resource-plugins.gpu.allowed-gpu-devices": "auto",
                        "yarn.nodemanager.resource-plugins.gpu.path-to-discovery-executables": "/usr/bin",
                        "yarn.nodemanager.linux-container-executor.cgroups.mount": "true",
                        "yarn.nodemanager.linux-container-executor.cgroups.mount-path": "/spark-rapids-cgroup",
                        "yarn.nodemanager.linux-container-executor.cgroups.hierarchy": "yarn",
                        "yarn.nodemanager.container-executor.class": "org.apache.hadoop.yarn.server.nodemanager.LinuxContainerExecutor",
                    }
                ),
                # Container executor for GPU 
                emr.CfnCluster.ConfigurationProperty(
                    classification="container-executor",
                    configurations=[
                        emr.CfnCluster.ConfigurationProperty(
                            classification="gpu",
                            configuration_properties={"module.enabled": "true"}
                        ),
                        emr.CfnCluster.ConfigurationProperty(
                            classification="cgroups",
                            configuration_properties={
                                "root": "/spark-rapids-cgroup",
                                "yarn-hierarchy": "yarn"
                            }
                        )
                    ]
                ),
                # Spark GPU configuration
                emr.CfnCluster.ConfigurationProperty(
                    classification="spark-defaults",
                    configuration_properties={
                        "spark.yarn.appMasterEnv.AWS_DEFAULT_REGION": scope.region,
                        "spark.yarn.executorEnv.AWS_DEFAULT_REGION": scope.region,
                        # Java 17 for executors and ApplicationMaster
                        "spark.executorEnv.JAVA_HOME": "/usr/lib/jvm/java-17-amazon-corretto.x86_64",
                        "spark.yarn.appMasterEnv.JAVA_HOME": "/usr/lib/jvm/java-17-amazon-corretto.x86_64",
                        # RAPIDS plugin available but not enabled by default
                        "spark.plugins": "com.nvidia.spark.SQLPlugin",
                        "spark.rapids.sql.enabled": "false",  # Disable RAPIDS by default
                        "spark.executor.resource.gpu.discoveryScript": "/usr/lib/spark/scripts/gpu/getGpusResources.sh",
                        "spark.executor.extraLibraryPath": "/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/compat/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/lib/hadoop/lib/native:/usr/lib/hadoop-lzo/lib/native:/docker/usr/lib/hadoop/lib/native:/docker/usr/lib/hadoop-lzo/lib/native",
                        # GPU resource configuration (available when needed)
                        "spark.rapids.sql.concurrentGpuTasks": "1",
                        "spark.executor.resource.gpu.amount": "1",
                        "spark.executor.cores": "2",
                        "spark.task.cpus": "1",
                        "spark.task.resource.gpu.amount": "0.5",
                        "spark.rapids.memory.pinnedPool.size": "0",
                        "spark.executor.memoryOverhead": "2G",
                        "spark.locality.wait": "0s",
                        "spark.sql.shuffle.partitions": "200",
                        "spark.sql.files.maxPartitionBytes": "512m"
                    }
                ),
                # Capacity scheduler for GPU
                emr.CfnCluster.ConfigurationProperty(
                    classification="capacity-scheduler",
                    configuration_properties={
                        "yarn.scheduler.capacity.resource-calculator": "org.apache.hadoop.yarn.util.resource.DominantResourceCalculator"
                    }
                )
            ],
            bootstrap_actions=[
                emr.CfnCluster.BootstrapActionConfigProperty(
                    name="Mount CGroup v1 for GPU",
                    script_bootstrap_action=emr.CfnCluster.ScriptBootstrapActionConfigProperty(
                        path=f"s3://{storage_construct.bucket.bucket_name}/dist/bootstrap/gpu-cgroup-bootstrap.sh"
                    )
                )
            ]
        )
        
        self.cluster.node.add_dependency(bootstrap_deployment)
        

        
