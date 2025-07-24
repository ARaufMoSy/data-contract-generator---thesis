import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache
from datetime import datetime
from azure.cosmos import CosmosClient
from databricks import sql
from config import Config


class MetadataRepository:
    """Handles Cosmos DB metadata operations."""
    
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize Cosmos DB client
        self.client = CosmosClient(config.cosmos_endpoint, config.cosmos_key)
        self.container = (self.client
                         .get_database_client(config.cosmos_database)
                         .get_container_client(config.cosmos_container))
        
        self.logger.info("✅ Cosmos DB repository initialized")
    
    def fetch_metadata(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata by identifier (table name or document ID)."""
        try:
            # Check if identifier is a table name (contains dots)
            if '.' in identifier and identifier.count('.') == 2:
                return self._fetch_by_table_name(identifier)
            else:
                return self._fetch_by_document_id(identifier)
        except Exception as e:
            self.logger.error(f"❌ Error fetching metadata: {e}")
            return None
    
    def _fetch_by_table_name(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata by table name."""
        parts = table_name.split('.')
        if len(parts) != 3:
            self.logger.warning(f"Invalid table name format: {table_name}")
            return None
        
        catalog, schema, table = parts
        query = """
        SELECT * FROM c 
        WHERE c.storage_info.name = @catalog 
        AND c.storage_info.schema_name = @schema 
        AND c.storage_info.table_name = @table
        """
        
        items = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@catalog", "value": catalog},
                {"name": "@schema", "value": schema},
                {"name": "@table", "value": table}
            ],
            enable_cross_partition_query=True
        ))
        
        if items:
            self.logger.info(f"✅ Found metadata for table: {table_name}")
            return items[0]
        else:
            self.logger.warning(f"⚠️ No metadata found for table: {table_name}")
            return None
    
    def _fetch_by_document_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata by document ID."""
        try:
            item = self.container.read_item(item=document_id, partition_key=document_id)
            self.logger.info(f"✅ Found metadata for ID: {document_id}")
            return item
        except Exception as e:
            self.logger.error(f"❌ Error fetching by ID {document_id}: {e}")
            return None


class SchemaRepository:
    """Handles Databricks Unity Catalog schema operations."""
    
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("✅ Databricks schema repository initialized")
    
    @lru_cache(maxsize=128)
    def extract_unity_catalog_schema(self, table_full_name: str) -> Dict[str, Any]:
        """Extract real schema from Unity Catalog with actual Databricks types."""
        try:
            self.logger.info(f"🔍 Extracting Unity Catalog schema for: {table_full_name}")
            
            # Validate Databricks configuration
            if not all([self.config.databricks_host, self.config.databricks_token, self.config.databricks_http_path]):
                self.logger.warning("⚠️ Databricks configuration incomplete, skipping schema extraction")
                return {}
            
            # Connect to Databricks
            connection = sql.connect(
                server_hostname=self.config.databricks_host,
                http_path=self.config.databricks_http_path,
                access_token=self.config.databricks_token
            )
            
            cursor = connection.cursor()
            
            # Extract column information
            columns = self._extract_columns(cursor, table_full_name)
            
            # Fetch sample data for examples
            examples = self._fetch_sample_data(cursor, table_full_name, columns)
            
            cursor.close()
            connection.close()
            
            # Build the models structure
            result = {"models": {}, "examples": examples}
            
            if columns:
                table_name = table_full_name.split('.')[-1]
                
                fields = {}
                for col in columns:
                    fields[col['name']] = {
                        'type': col['type'],
                        'required': False,
                        'description': col['comment']
                    }
                
                models = {
                    table_name: {
                        'description': f"Table {table_name} from Unity Catalog",
                        'fields': fields
                    }
                }
                
                result["models"] = models
                self.logger.info(f"✅ Extracted {len(columns)} columns with actual Databricks types")
            
            return result
                
        except Exception as e:
            self.logger.error(f"❌ Failed to extract Unity Catalog schema: {e}")
            return {"models": {}, "examples": []}
    
    def _extract_columns(self, cursor, table_full_name: str) -> List[Dict[str, str]]:
        """Extract column information from table."""
        describe_query = f"DESCRIBE TABLE {table_full_name}"
        cursor.execute(describe_query)
        
        columns = []
        for row in cursor.fetchall():
            col_name = row[0]
            col_type = row[1]
            col_comment = row[2] if len(row) > 2 and row[2] else "null"
            
            # Stop when we hit the metadata section
            if not col_name or col_name.strip() == '' or col_name.startswith('#'):
                break
                
            # Only include actual column data
            if col_name and col_name.strip() and not col_name.startswith('#'):
                columns.append({
                    'name': col_name.strip(),
                    'type': col_type.strip(),
                    'comment': col_comment.strip() if col_comment else "null"
                })
        
        return columns
    
    def _fetch_sample_data(self, cursor, table_full_name: str, columns: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Fetch sample data for examples."""
        examples = []
        if not columns:
            return examples
            
        try:
            sample_query = f"SELECT * FROM {table_full_name} LIMIT 1"
            cursor.execute(sample_query)
            sample_rows = cursor.fetchall()
            
            if sample_rows:
                table_name = table_full_name.split('.')[-1]
                
                for i, row in enumerate(sample_rows):
                    example_record = {}
                    for j, col in enumerate(columns):
                        col_name = col['name']
                        value = row[j] if j < len(row) else None
                        
                        # Format values properly for YAML output
                        if value is None:
                            example_record[col_name] = None
                        elif isinstance(value, str):
                            example_record[col_name] = value
                        elif isinstance(value, (int, float)):
                            example_record[col_name] = value
                        elif isinstance(value, datetime):
                            example_record[col_name] = value.isoformat()
                        else:
                            example_record[col_name] = str(value)
                    
                    examples.append({
                        "name": "Data Example",
                        "description": "Actual Data Example",
                        "model": table_name,
                        "record": example_record
                    })
                
                self.logger.info(f"✅ Generated {len(examples)} example rows")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not fetch sample data: {e}")
            examples = []
        
        return examples
