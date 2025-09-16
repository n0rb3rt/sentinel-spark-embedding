from aws_cdk import (
    aws_emr as emr,
    aws_iam as iam,
    aws_ssm as ssm,
    aws_ec2 as ec2,
    aws_s3 as s3,
    Stack,
    StackProps,
    CfnOutput,
    Environment as cdk,
    aws_s3_deployment as s3deploy,
)
from aws_cdk.aws_lakeformation import CfnPermissions as LF
from constructs import Construct

class EMRGPUStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # Get iceberg bucket from existing stack (only dependency)
        
        iceberg_bucket_name = ssm.StringParameter.value_for_string_parameter(self, "/airflow/variables/bucket_name")
        bucket = s3.Bucket.from_bucket_name(self, "IcebergBucket", iceberg_bucket_name)
        
        # Use existing VPC
        vpc = ec2.Vpc.from_lookup(self, "ExistingVPC", vpc_id="vpc-0f0b7d620bee1f8a1", region=self.region)
        
        # EMR Security Group
        emr_security_group = ec2.SecurityGroup(self, "EMRSecurityGroup",
            vpc=vpc,
            description="Security group for EMR GPU cluster",
            allow_all_outbound=True
        )
        
        # Allow EMR internal communication
        emr_security_group.add_ingress_rule(
            peer=emr_security_group,
            connection=ec2.Port.all_traffic(),
            description="EMR internal communication"
        )
        
        # EMR Service Role
        emr_service_role = iam.Role(
            self, "EMRServiceRole",
            assumed_by=iam.ServicePrincipal("elasticmapreduce.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonElasticMapReduceRole")
            ]
        )

        # EMR Instance Role
        emr_instance_role = iam.Role(
            self, "EMRInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonElasticMapReduceforEC2Role")
            ]
        )

        # Add SSM permissions for parameter access
        emr_instance_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=["ssm:GetParameter", "ssm:GetParameters"],
            resources=[f"arn:aws:ssm:{self.region}:{self.account}:parameter/airflow/variables/*"]
        ))
        
        # Add S3 permissions for Iceberg and temp directories
        emr_instance_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            resources=[
                f"arn:aws:s3:::{iceberg_bucket_name}",
                f"arn:aws:s3:::{iceberg_bucket_name}/*"
            ]
        ))
        
        # Add Glue permissions for Iceberg catalog
        emr_instance_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "glue:GetDatabase",
                "glue:GetTable",
                "glue:CreateTable",
                "glue:UpdateTable",
                "glue:DeleteTable",
                "glue:GetPartitions"
            ],
            resources=[
                f"arn:aws:glue:{self.region}:{self.account}:catalog",
                f"arn:aws:glue:{self.region}:{self.account}:database/sentinel",
                f"arn:aws:glue:{self.region}:{self.account}:table/sentinel/*"
            ]
        ))
        
        # Add permissions to access Sentinel-2 COGs (public dataset)
        emr_instance_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "s3:GetObject",
                "s3:ListBucket"
            ],
            resources=[
                "arn:aws:s3:::sentinel-cogs",
                "arn:aws:s3:::sentinel-cogs/*"
            ]
        ))
        
        # Lake Formation permissions for EMR instance role
        database_resource = LF.ResourceProperty(database_resource=LF.DatabaseResourceProperty(name="sentinel"))
        table_resource = LF.ResourceProperty(table_resource=LF.TableResourceProperty(database_name="sentinel", table_wildcard={}))
        
        emr_principal = LF.DataLakePrincipalProperty(data_lake_principal_identifier=emr_instance_role.role_arn)
        
        emr_db_perms = LF(self, "EMRDatabasePermissions",
            data_lake_principal=emr_principal, resource=database_resource, permissions=["ALL"])
        emr_table_perms = LF(self, "EMRTablePermissions", 
            data_lake_principal=emr_principal, resource=table_resource, permissions=["ALL"])

        emr_instance_profile = iam.CfnInstanceProfile(
            self, "EMRInstanceProfile",
            roles=[emr_instance_role.role_name]
        )

        # EMR GPU Cluster
        self.emr_cluster = emr.CfnCluster(
            self, "EMRGPUCluster",
            name="sentinel-embedding-gpu",
            release_label="emr-7.9.0",
            service_role=emr_service_role.role_arn,
            job_flow_role=emr_instance_profile.ref,
            log_uri=f"s3://{iceberg_bucket_name}/logs/",
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
                ec2_subnet_id=vpc.private_subnets[0].subnet_id,
                additional_master_security_groups=[emr_security_group.security_group_id],
                additional_slave_security_groups=[emr_security_group.security_group_id]
            ),
            applications=[
                emr.CfnCluster.ApplicationProperty(name="Spark"),
                emr.CfnCluster.ApplicationProperty(name="Hadoop")
            ],
            configurations=[
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
                        "yarn.nodemanager.remote-app-log-dir": f"s3://{iceberg_bucket_name}/logs",
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
                # Spark GPU configuration - https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-rapids.html
                emr.CfnCluster.ConfigurationProperty(
                    classification="spark-defaults",
                    configuration_properties={
                        "spark.yarn.appMasterEnv.AWS_DEFAULT_REGION": self.region,
                        "spark.yarn.executorEnv.AWS_DEFAULT_REGION": self.region,
                        # RAPIDS plugin configuration
                        "spark.plugins": "com.nvidia.spark.SQLPlugin",
                        "spark.executor.resource.gpu.discoveryScript": "/usr/lib/spark/scripts/gpu/getGpusResources.sh",
                        "spark.executor.extraLibraryPath": "/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/compat/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/lib/hadoop/lib/native:/usr/lib/hadoop-lzo/lib/native:/docker/usr/lib/hadoop/lib/native:/docker/usr/lib/hadoop-lzo/lib/native",
                        # GPU resource configuration
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
            # Bootstrap action for EMR 7.x CGroup v1 mounting
            bootstrap_actions=[
                emr.CfnCluster.BootstrapActionConfigProperty(
                    name="Mount CGroup v1 for GPU",
                    script_bootstrap_action=emr.CfnCluster.ScriptBootstrapActionConfigProperty(
                        path=f"s3://{iceberg_bucket_name}/dist/bootstrap/gpu-cgroup-bootstrap.sh"
                    )
                )
            ]
        )
        
        # Get S3 bucket from SSM for artifact downloads
        bucket_name = ssm.StringParameter.value_for_string_parameter(self, "/sentinel-spark/s3-bucket")
        
        # Create bootstrap script for Python 3.12, venv, and model staging
        bootstrap_script = f"""
#!/bin/bash
set -ex

# Install Python 3.12
sudo yum update -y
sudo yum install -y python3.12 python3.12-pip python3.12-devel

# Mount CGroup v1 for GPU support
sudo mkdir -p /spark-rapids-cgroup/devices
sudo mount -t cgroup -o devices cgroupv1-devices /spark-rapids-cgroup/devices
sudo chmod a+rwx -R /spark-rapids-cgroup

# Check if this is a core node by looking for GPU
if lspci | grep -i nvidia &>/dev/null; then
    echo "Core node detected (has GPU) - installing ML dependencies"
    
    # Use instance store /mnt1 for ML artifacts
    sudo mkdir -p /mnt1/opt/venv /mnt1/opt/clay-model
    
    # Download and extract venv
    sudo aws s3 cp s3://{bucket_name}/dist/venv/venv.tar.gz /tmp/
    sudo tar -xzf /tmp/venv.tar.gz -C /mnt1/opt/venv/
    sudo chmod -R 755 /mnt1/opt/venv
    
    # Download Clay model files
    sudo aws s3 sync s3://{bucket_name}/dist/model/ /mnt1/opt/clay-model/
    sudo chmod -R 755 /mnt1/opt/clay-model
else
    echo "Master node detected (no GPU) - skipping ML dependencies"
fi
"""
             
        bootstrap_deployment = s3deploy.BucketDeployment(
            self, "BootstrapScriptDeployment",
            sources=[s3deploy.Source.data("gpu-cgroup-bootstrap.sh", bootstrap_script)],
            destination_bucket=bucket,
            destination_key_prefix="dist/bootstrap/"
        )
        
        # Ensure bootstrap script is deployed before cluster creation
        self.emr_cluster.node.add_dependency(bootstrap_deployment)
        
        # Export cluster ID to SSM for other stacks to reference
        ssm.StringParameter(
            self, "EMRClusterIdParameter",
            parameter_name="/sentinel-spark/emr-cluster-id",
            string_value=self.emr_cluster.ref
        )

        CfnOutput(self, "EMRClusterId", value=self.emr_cluster.ref)