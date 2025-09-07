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
                    field_def = {
                        'type': col['type'],  # Keep exact Unity Catalog type
                        'required': False,
                        'description': col['comment']
                    }
                    
                    # Add items property for complex types (arrays, structs, maps)
                    items_def = self._extract_items_definition(col['type'])
                    if items_def:
                        field_def['items'] = items_def
                    
                    fields[col['name']] = field_def
                    
                
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
    
    def _extract_items_definition(self, databricks_type: str) -> Optional[Dict[str, Any]]:
        """Extract items definition for complex types while preserving exact Unity Catalog types."""
        if not databricks_type:
            return None
        
        type_str = databricks_type.strip()
        type_lower = type_str.lower()
        
        try:
            # Handle array types: array<element_type>
            if type_lower.startswith('array<'):
                # Extract inner type from array<inner_type>
                inner_start = type_str.find('<') + 1
                inner_end = self._find_matching_bracket(type_str, inner_start - 1)
                if inner_end > inner_start:
                    inner_type = type_str[inner_start:inner_end].strip()
                    
                    items_def = {
                        'type': inner_type  # Keep exact Unity Catalog type
                    }
                    
                    # Recursively handle nested complex types
                    nested_items = self._extract_items_definition(inner_type)
                    if nested_items:
                        items_def['items'] = nested_items
                    
                    return items_def
            
            # Handle struct types: struct<field1:type1,field2:type2,...>
            elif type_lower.startswith('struct<'):
                # Extract struct definition
                inner_start = type_str.find('<') + 1
                inner_end = self._find_matching_bracket(type_str, inner_start - 1)
                if inner_end > inner_start:
                    struct_def = type_str[inner_start:inner_end].strip()
                    return self._parse_struct_definition(struct_def)
            
            # Handle map types: map<key_type,value_type>
            elif type_lower.startswith('map<'):
                # Extract map definition
                inner_start = type_str.find('<') + 1
                inner_end = self._find_matching_bracket(type_str, inner_start - 1)
                if inner_end > inner_start:
                    map_def = type_str[inner_start:inner_end].strip()
                    return self._parse_map_definition(map_def)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to parse complex type {databricks_type}: {e}")
        
        return None
    
    def _find_matching_bracket(self, text: str, start_pos: int) -> int:
        """Find the matching closing bracket for a given opening bracket position."""
        if start_pos >= len(text) or text[start_pos] != '<':
            return -1
        
        bracket_count = 1
        pos = start_pos + 1
        
        while pos < len(text) and bracket_count > 0:
            if text[pos] == '<':
                bracket_count += 1
            elif text[pos] == '>':
                bracket_count -= 1
            pos += 1
        
        return pos - 1 if bracket_count == 0 else -1
    
    def _parse_struct_definition(self, struct_def: str) -> Dict[str, Any]:
        """Parse struct definition into items format with exact Unity Catalog types."""
        try:
            # For structs, we define the properties of the object
            properties = {}
            
            # Split fields by comma, but be careful with nested structures
            fields = self._split_struct_fields(struct_def)
            
            for field in fields:
                if ':' in field:
                    field_name, field_type = field.split(':', 1)
                    field_name = field_name.strip()
                    field_type = field_type.strip()
                    
                    prop_def = {
                        'type': field_type  # Keep exact Unity Catalog type
                    }
                    
                    # Handle nested complex types
                    nested_items = self._extract_items_definition(field_type)
                    if nested_items:
                        prop_def['items'] = nested_items
                    
                    properties[field_name] = prop_def
            
            return {
                'type': 'struct',  # Use 'struct' to indicate it's a struct type
                'properties': properties
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to parse struct definition {struct_def}: {e}")
            return {'type': 'struct'}
    
    def _parse_map_definition(self, map_def: str) -> Dict[str, Any]:
        """Parse map definition into items format with exact Unity Catalog types."""
        try:
            # Find the comma that separates key and value types, accounting for nested structures
            comma_pos = self._find_map_separator(map_def)
            if comma_pos > 0:
                key_type = map_def[:comma_pos].strip()
                value_type = map_def[comma_pos + 1:].strip()
                
                # For maps, we represent the structure
                value_def = {
                    'type': value_type  # Keep exact Unity Catalog type
                }
                
                # Handle nested complex types in values
                nested_items = self._extract_items_definition(value_type)
                if nested_items:
                    value_def['items'] = nested_items
                
                return {
                    'type': 'map',  # Use 'map' to indicate it's a map type
                    'keyType': key_type,  # Preserve exact key type
                    'valueType': value_def  # Value type with potential nesting
                }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to parse map definition {map_def}: {e}")
        
        return {'type': 'map'}
    
    def _find_map_separator(self, map_def: str) -> int:
        """Find the comma that separates key and value types in a map definition."""
        bracket_count = 0
        for i, char in enumerate(map_def):
            if char == '<':
                bracket_count += 1
            elif char == '>':
                bracket_count -= 1
            elif char == ',' and bracket_count == 0:
                return i
        return -1
    
    def _split_struct_fields(self, struct_def: str) -> List[str]:
        """Split struct fields by comma, handling nested structures properly."""
        fields = []
        current_field = ""
        bracket_count = 0
        
        for char in struct_def:
            if char == '<':
                bracket_count += 1
            elif char == '>':
                bracket_count -= 1
            elif char == ',' and bracket_count == 0:
                if current_field.strip():
                    fields.append(current_field.strip())
                current_field = ""
                continue
            
            current_field += char
        
        # Add the last field
        if current_field.strip():
            fields.append(current_field.strip())
        
        return fields
    
    def _analyze_field_completeness(self, cursor, table_full_name: str, columns: List[Dict[str, str]]) -> Dict[str, float]:
        """Analyze completeness for each field (optional - for better accuracy)."""
        field_completeness = {}
        
        try:
            # Sample 100 rows for quick analysis
            sample_query = f"SELECT * FROM {table_full_name} LIMIT 100"
            cursor.execute(sample_query)
            sample_rows = cursor.fetchall()
            
            if sample_rows:
                for idx, col in enumerate(columns):
                    col_name = col['name']
                    empty_count = 0
                    
                    for row in sample_rows:
                        value = row[idx] if idx < len(row) else None
                        # Check for null, empty string, or '0'
                        if value is None or str(value).strip() == '' or str(value).strip() == '0':
                            empty_count += 1
                    
                    completeness = ((len(sample_rows) - empty_count) / len(sample_rows)) * 100
                    field_completeness[col_name] = completeness
            
            return field_completeness
        
        except Exception as e:
            self.logger.warning(f"Could not analyze field completeness: {e}")
            return {}
        