resource "aws_elasticache_cluster" "redis_worker_cache" {
  cluster_id           = "vectorlogic-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro" 
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
}
