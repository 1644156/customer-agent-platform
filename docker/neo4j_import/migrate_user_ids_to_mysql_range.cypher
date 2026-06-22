// Align demo graph users with MySQL order users.
// Before: Neo4j User.user_id = 1..50
// After:  Neo4j User.user_id = 1001..1050
//
// PowerShell usage:
// Get-Content .\docker\neo4j_import\migrate_user_ids_to_mysql_range.cypher | docker exec -i ecs-neo4j cypher-shell -u neo4j -p 12345678

MATCH (u:User)
WHERE u.user_id < 1000
SET u.user_id = u.user_id + 1000
RETURN count(u) AS updated_users;
