import os
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from datacontract.data_contract import DataContract
from databricks import sql
from functools import lru_cache
import json
from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from packaging import version

@dataclass
class Config:
    """Unified configuration class."""
    cosmos_endpoint: str
    cosmos_key: str
    cosmos_database: str = "metadata"
    cosmos_container: str = "table_metadata"

    databricks_host: str = ""
    databricks_token: str = ""
    databricks_http_path: str = ""
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment."""
        load_dotenv("credentials.env")
        
        endpoint = os.getenv("COSMOS_DB_ENDPOINT")
        key = os.getenv("COSMOS_DB_KEY")
        
        if not endpoint or not key:
            raise ValueError("COSMOS_DB_ENDPOINT and COSMOS_DB_KEY must be set")
        
        return cls(
            cosmos_endpoint=endpoint,
            cosmos_key=key,
            cosmos_database=os.getenv("COSMOS_DB_DATABASE", "metadata"),
            cosmos_container=os.getenv("COSMOS_DB_CONTAINER", "table_metadata"),
            databricks_host=os.getenv("DATABRICKS_HOST", ""),
            databricks_token=os.getenv("DATABRICKS_TOKEN", ""),
            databricks_http_path=os.getenv("DATABRICKS_HTTP_PATH", "")
        )


class DataContractGenerator:
    """ Main contract generator Class with all functionality."""
    
    def __init__(self):
        """Initialize generator with logging and configuration."""
        self.logger = self._setup_logging()
        self.config = Config.from_env()
        
        # Initialize Cosmos DB client
        self.client = CosmosClient(self.config.cosmos_endpoint, self.config.cosmos_key)
        self.container = (self.client
                         .get_database_client(self.config.cosmos_database)
                         .get_container_client(self.config.cosmos_container))
        
        self.logger.info("✅ Generator initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('data_contract_generator.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def fetch_metadata(self, identifier: str) -> Optional[Dict[str, Any]]:
        """ Fetch metadata by identifier(table name or document ID).
            Args:identifier: Table full name (catalog.schema.table) or document ID """
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
        
    @lru_cache(maxsize=128)
    def extract_unity_catalog_schema(self, table_full_name: str) -> Dict[str, Any]:
        """ Extract real schema from Unity Catalog with actual Databricks types. """
        try:
            self.logger.info(f"🔍 Extracting Unity Catalog schema for: {table_full_name}")
            
            # Validate Databricks configuration
            if not all([self.config.databricks_host, self.config.databricks_token, self.config.databricks_http_path]):
                self.logger.warning("⚠️ Databricks configuration incomplete, skipping schema extraction")
                return {}
            
            # Connect directly to Databricks to get actual schema
            connection = sql.connect(
                server_hostname=self.config.databricks_host,
                http_path=self.config.databricks_http_path,
                access_token=self.config.databricks_token
            )
            
            cursor = connection.cursor()
            
            # Use DESCRIBE TABLE (not EXTENDED) to get only column information
            describe_query = f"DESCRIBE TABLE {table_full_name}"
            cursor.execute(describe_query)
            
            columns = []
            for row in cursor.fetchall():
                col_name = row[0]
                col_type = row[1]
                col_comment = row[2] if len(row) > 2 and row[2] else "null"
                
                # Stop when we hit the metadata section (marked by empty line or # )
                if not col_name or col_name.strip() == '' or col_name.startswith('#'):
                    break
                    
                # Only include actual column data
                if col_name and col_name.strip() and not col_name.startswith('#'):
                    columns.append({
                        'name': col_name.strip(),
                        'type': col_type.strip(),
                        'comment': col_comment.strip() if col_comment else "null"
                    })
            
            # Fetch sample data for examples
            examples = []
            if columns:
                try:
                    sample_query = f"SELECT * FROM {table_full_name} LIMIT 1"
                    cursor.execute(sample_query)
                    sample_rows = cursor.fetchall()
                    
                    if sample_rows:
                        table_name = table_full_name.split('.')[-1]  # Get just the table name
                        
                        # Create example records as dictionaries with proper structure
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
                                    # Convert other types to string representation
                                    example_record[col_name] = str(value)
                            
                            # Structure the example according to your desired format
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
            
            cursor.close()
            connection.close()
            
            # Build the models structure with actual types
            result = {"models": {}, "examples": examples}
            
            if columns:
                table_name = table_full_name.split('.')[-1]  # Get just the table name
                
                fields = {}
                for col in columns:
                    fields[col['name']] = {
                        'type': col['type'],  # Actual Databricks type (int, varchar(10), etc.)
                        'required': False,    # Default to False
                        'description': col['comment']  # Use column comment as description
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

    def _enhance_models_with_metadata(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance extracted models with only type, required, and description fields.
        """
        enhanced_models = {}
        
        for model_name, model_data in models.items():
            if isinstance(model_data, dict):
                # Ensure model has description
                if 'description' not in model_data or not model_data['description']:
                    model_data['description'] = "null"
                
                # Process fields - keep only type, required, and description
                fields = model_data.get('fields', {})
                cleaned_fields = {}
                
                for field_name, field_data in fields.items():
                    if isinstance(field_data, dict):
                        # Extract only the three required fields
                        cleaned_field = {
                            'type': field_data.get('type', 'string'),  # Keep actual type from table
                            'required': field_data.get('required', False),  # Default to False
                            'description': field_data.get('description', 'null')  # Default to "nulln"
                        }
                        cleaned_fields[field_name] = cleaned_field
                    else:
                        # Handle case where field_data is not a dict
                        cleaned_fields[field_name] = {
                            'type': 'string',
                            'required': False,
                            'description': 'null'
                        }
                
                # Update model with cleaned fields
                model_data['fields'] = cleaned_fields
                enhanced_models[model_name] = model_data
            else:
                enhanced_models[model_name] = model_data
        
        return enhanced_models

    def map_to_contract(self, cosmos_metadata: Dict[str, Any], table_full_name: str = "") -> Dict[str, Any]:
        """
        Map Cosmos DB metadata to Data Contract format with Unity Catalog schema.
        """
        self.logger.info("🔄 Mapping metadata to contract structure...")
        
        metadata = cosmos_metadata.get('metadata', {})
        storage_info = cosmos_metadata.get('storage_info', {})
        
        # Build contract sections in the correct order
        contract = {
            "dataContractSpecification": "1.1.0",
            "id": cosmos_metadata.get("id", ""),
            "info": self._build_info(metadata),
            "servers": self._build_servers(storage_info),
            "terms": self._extract_fields(metadata, ["usage", "limitations"]),  # Terms come after servers
        }
        
        # Extract Unity Catalog schema and examples
        unity_data = {"models": {}, "examples": []}
        if table_full_name:
            unity_data = self.extract_unity_catalog_schema(table_full_name)
        
        # Add models section
        contract["models"] = unity_data.get("models", {})
        
        # Add remaining sections in the correct order
        contract.update({
            "examples": unity_data.get("examples", []),  # Examples come right after models
            "quality": self._extract_fields(metadata, ["data_category", "data_format", "unique_keys"]),
            "servicelevels": self._build_service_levels(metadata),
            "links": {"dataPark": "https://data-products.prd.dis.siemensgamesa.com/"},
            "tags": metadata.get("tags", []),
            "definitions": {}
        })
        
        # Remove empty sections (except models and examples which should always be present)
        contract = {k: v for k, v in contract.items() if v or k in ["models", "examples"]}
        
        self.logger.info("✅ Successfully mapped metadata to contract with Unity Catalog schema and examples")
        return contract
    
    def _build_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Build info section."""
        info = self._extract_fields(metadata, ["title", "description", "source_system"])
        info.update({
            "version": metadata.get("version", "1.0.0"),
            "status": metadata.get("status", "active")
        })
        
        # Add contact if available
        contact = self._extract_fields(metadata, ["owner", "owner_email"], 
                                     {"owner": "name", "owner_email": "email"})
        if contact:
            info["contact"] = contact
        
        return info
    
    def _build_servers(self, storage_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build servers section."""
        if not storage_info:
            return {}
        
        server = {"type": "databricks"}
        
        # Add Databricks host if available
        if os.getenv("DATABRICKS_HOST"):
            server["host"] = os.getenv("DATABRICKS_HOST")
        
        # Map storage info fields
        field_mapping = {
            "name": "catalog",
            "schema_name": "schema", 
            "table_name": "table_name",
            "location": "location"
        }
        
        for src_key, dst_key in field_mapping.items():
            if src_key in storage_info:
                server[dst_key] = storage_info[src_key]
        
        return {"production": server} if len(server) > 1 else {}
    
    def _build_service_levels(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Build service levels section."""
        servicelevels = {}
        
        # Add frequency
        if "update_frequency" in metadata:
            servicelevels["frequency"] = {"description": metadata["update_frequency"]}
        
        # Add freshness timestamps
        freshness = {}
        for key in ["last_updated_at", "created_at"]:
            if key in metadata:
                freshness[key] = self._format_timestamp(metadata[key])
        
        if freshness:
            servicelevels["freshness"] = freshness
        
        return servicelevels
    
    def _extract_fields(self, source: Dict[str, Any], fields: list, 
                       mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Extract and optionally rename fields from source dictionary."""
        result = {}
        mapping = mapping or {}
        
        for field in fields:
            if field in source and source[field]:
                key = mapping.get(field, field)
                result[key] = source[field]
        
        return result
    
    def _format_timestamp(self, timestamp: Any) -> str:
        """Format timestamp to ISO format."""
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp).isoformat()
        return str(timestamp)
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for file system."""
        return re.sub(r'[<>:"/\\|?*\s]+', '_', filename).strip('_') or "data_contract"
    
    def _setup_yaml_formatting(self) -> None:
        """Configure YAML formatting."""
        def str_representer(dumper, data):
            if '\n' in data:
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)
        
        yaml.add_representer(str, str_representer)
    
    def _normalize_text_fields(self, data: Any) -> Any:
        """Recursively normalize text fields to handle encoding issues."""
        import unicodedata
        
        if isinstance(data, str):
            # Normalize Unicode characters and remove problematic ones
            normalized = unicodedata.normalize('NFKD', data)
            # Replace smart quotes and other problematic characters
            replacements = {
                '\u2018': "'",  # Left single quotation mark
                '\u2019': "'",  # Right single quotation mark
                '\u201c': '"',  # Left double quotation mark
                '\u201d': '"',  # Right double quotation mark
                '\u2013': '-',  # En dash
                '\u2014': '--', # Em dash
                '\u2026': '...',# Horizontal ellipsis
                '\u00a0': ' ',  # Non-breaking space
            }
            for old, new in replacements.items():
                normalized = normalized.replace(old, new)
            
            # Keep only ASCII and common Unicode characters
            return ''.join(c for c in normalized if ord(c) < 127 or c.isalnum())
        
        elif isinstance(data, dict):
            return {key: self._normalize_text_fields(value) for key, value in data.items()}
        
        elif isinstance(data, list):
            return [self._normalize_text_fields(item) for item in data]
        
        else:
            return data
    
    def save_contract(self, contract_dict: Dict[str, Any], output_dir: str = "DataContracts") -> Path:
        """Save contract to YAML file with proper version management."""
        
        # Create output directory
        directory_path = Path(output_dir)
        directory_path.mkdir(exist_ok=True)
        
        # Generate base filename
        info = contract_dict.get('info', {})
        title = info.get('title', 'data_contract')
        base_filename = self._sanitize_filename(title)
        
        # Find existing files for version management
        existing_files = list(directory_path.glob(f"{base_filename}*.yaml"))
        previous_contract_path = None
        
        if existing_files:
            # Find the latest version file
            previous_contract_path = self._find_latest_version_file(contract_dict, directory_path)
        
        # Handle version management
        contract_dict = self._handle_version_management(contract_dict, previous_contract_path, directory_path)
        
        # Generate final filename with the NEW version
        version = contract_dict.get('info', {}).get('version', '1.0.0')
        filename = f"{base_filename}_v{version}"
        filepath = directory_path / f"{filename}.yaml"
        
        # Check if file already exists with this version
        if filepath.exists():
            self.logger.info(f"ℹ️ Contract version {version} already exists. No changes needed.")
            return filepath
        
        # Save file with new version
        try:
            self._setup_yaml_formatting()
            normalized_contract = self._normalize_text_fields(contract_dict)
            
            with open(filepath, 'w', encoding='utf-8', newline='\n') as file:
                yaml.dump(normalized_contract, file, sort_keys=False, allow_unicode=True, 
                        default_flow_style=False, width=1000, indent=2)
            
            self.logger.info(f"✅ Contract saved to: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save contract: {e}")
            raise
    
    def generate_draft_contract(self, identifier: str, output_dir: str = "DataContracts") -> Path:
        """
        Generate complete contract from identifier with Unity Catalog schema and version management.
        """
        self.logger.info(f"🚀 Starting contract generation for: {identifier}")
        
        # Fetch metadata
        metadata = self.fetch_metadata(identifier)
        if not metadata:
            raise ValueError(f"No metadata found for identifier: {identifier}")
        
        # Determine table name for schema extraction
        table_name = ""
        if '.' in identifier and identifier.count('.') == 2:
            table_name = identifier
        else:
            # Extract table name from storage_info if available
            storage_info = metadata.get('storage_info', {})
            if all(key in storage_info for key in ['name', 'schema_name', 'table_name']):
                table_name = f"{storage_info['name']}.{storage_info['schema_name']}.{storage_info['table_name']}"
        
        # Map to contract format with schema
        contract = self.map_to_contract(metadata, table_name)
        
        # Save contract (with version management)
        filepath = self.save_contract(contract, output_dir)
        
        # Check if this is actually a new file or if no changes were detected
        if filepath.exists():
            # Reload the contract to get the final version
            with open(filepath, 'r', encoding='utf-8') as file:
                final_contract = yaml.safe_load(file)
            
            # Display summary
            self._display_summary(final_contract, filepath)
        
        return filepath
    
    def _display_summary(self, contract: Dict[str, Any], filepath: Path, validation_results: Optional[Dict[str, Any]] = None) -> None:
        """Simplified summary display - minimal terminal output."""
        info = contract.get('info', {})
        
        print(f"✅ Contract generated: {info.get('title', 'Unknown')} v{info.get('version', 'Unknown')}")
        
        if validation_results:
            print(f"📊 Health Score: {validation_results['health_score']}% | Status: {validation_results['status']}")
            print(f"📄 Results saved to: {filepath.parent / f'{filepath.stem}_health_report.json'}")

    def _handle_version_management(self, contract_dict: Dict[str, Any], previous_contract_path: Optional[Path], output_dir: Path) -> Dict[str, Any]:
        """Handle version management by comparing with the latest version contract."""

        if not previous_contract_path or not previous_contract_path.exists():
            self.logger.info("🆕 No previous contract found. Using initial version.")
            return contract_dict
        
        # Find the actual latest version file, not just by modification time
        latest_contract_path = self._find_latest_version_file(contract_dict, output_dir)
        
        if not latest_contract_path:
            self.logger.info("🆕 No versioned contracts found. Using initial version.")
            return contract_dict
        
        try:
            # Load the latest contract for comparison
            with open(latest_contract_path, 'r', encoding='utf-8') as file:
                latest_contract_dict = yaml.safe_load(file)
            
            # Get the latest version from the actual latest contract
            latest_version = latest_contract_dict.get("info", {}).get("version", "1.0.0")
            self.logger.info(f"🔍 Comparing against latest version: {latest_version} from {latest_contract_path.name}")
            
            # Create temporary file for current contract
            temp_current_path = output_dir / f"temp_current_contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            
            # Save current contract temporarily with latest version for comparison
            temp_contract = contract_dict.copy()
            temp_contract["info"]["version"] = latest_version  # Set same version for comparison
            
            self._setup_yaml_formatting()
            normalized_contract = self._normalize_text_fields(temp_contract)
            
            with open(temp_current_path, 'w', encoding='utf-8', newline='\n') as file:
                yaml.dump(normalized_contract, file, sort_keys=False, allow_unicode=True, 
                        default_flow_style=False, width=1000, indent=2)
            
            # Determine version increment based on changes
            new_version = self._determine_version_increment(latest_contract_dict, contract_dict, latest_version)
            
            # Update contract with new version
            contract_dict["info"]["version"] = new_version
            self.logger.info(f"🔄 Version updated from {latest_version} to {new_version}")
            
            # Clean up temporary file
            if temp_current_path.exists():
                temp_current_path.unlink()
            
            return contract_dict
            
        except Exception as e:
            self.logger.error(f"❌ Version management failed: {e}")
            return contract_dict
    
    def _find_latest_version_file(self, contract_dict: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Find the latest version file based on semantic versioning."""
        try:
            # Generate base filename
            info = contract_dict.get('info', {})
            title = info.get('title', 'data_contract')
            base_filename = self._sanitize_filename(title)
            
            # Find all files with same base name
            existing_files = list(output_dir.glob(f"{base_filename}*.yaml"))
            
            if not existing_files:
                return None
            
            # Extract versions and find the latest
            versioned_files = []
            for file_path in existing_files:
                # Extract version from filename (e.g., "datacontract_v1.2.0.yaml" -> "1.2.0")
                filename = file_path.stem
                if '_v' in filename:
                    version_part = filename.split('_v')[-1]
                    try:
                        # Validate version format
                        version.parse(version_part)
                        versioned_files.append((file_path, version_part))
                    except:
                        continue
            
            if not versioned_files:
                return None
            
            # Sort by version and return the latest
            versioned_files.sort(key=lambda x: version.parse(x[1]), reverse=True)
            latest_file = versioned_files[0][0]
            
            self.logger.info(f"📋 Found latest version file: {latest_file.name}")
            return latest_file
            
        except Exception as e:
            self.logger.error(f"❌ Error finding latest version: {e}")
            return None
        
    def _determine_version_increment(self, old_contract: Dict[str, Any], new_contract: Dict[str, Any], current_version: str) -> str:
        """Determine the appropriate version increment based on changes."""
        try:
            old_models = old_contract.get('models', {})
            new_models = new_contract.get('models', {})
            
            major, minor, patch = map(int, current_version.split('.'))
            
            # Check for breaking changes
            if self._has_breaking_changes(old_models, new_models):
                return f"{major + 1}.0.0"
            
            # Check for non-breaking changes (new fields, models, etc.)
            if self._has_minor_changes(old_models, new_models):
                return f"{major}.{minor + 1}.0"
            
            # No significant changes detected
            self.logger.info("✅ No significant schema changes detected. Version remains the same.")
            return current_version
            
        except Exception as e:
            self.logger.error(f"❌ Version increment determination failed: {e}")
            return current_version

    # Add these helper methods for change detection
    def _has_breaking_changes(self, old_models: Dict[str, Any], new_models: Dict[str, Any]) -> bool:
        """Check for breaking changes (removed fields, type changes, removed models)."""
        # Check for removed models
        if set(old_models.keys()) - set(new_models.keys()):
            removed_models = set(old_models.keys()) - set(new_models.keys())
            self.logger.info(f"🚨 Removed models: {removed_models}")
            return True
        
        # Check for field-level breaking changes
        for model_name, model_data in old_models.items():
            if model_name in new_models:
                old_fields = model_data.get('fields', {})
                new_fields = new_models[model_name].get('fields', {})
                
                # Check for removed fields
                removed_fields = set(old_fields.keys()) - set(new_fields.keys())
                if removed_fields:
                    self.logger.info(f"🚨 Removed fields in {model_name}: {removed_fields}")
                    return True
                
                # Check for type changes
                for field_name, field_data in old_fields.items():
                    if field_name in new_fields:
                        old_type = field_data.get('type', '')
                        new_type = new_fields[field_name].get('type', '')
                        if old_type != new_type:
                            self.logger.info(f"🚨 Type change in {model_name}.{field_name}: {old_type} -> {new_type}")
                            return True
        
        return False

    def _has_minor_changes(self, old_models: Dict[str, Any], new_models: Dict[str, Any]) -> bool:
        """Check for minor changes (added fields, added models, description changes)."""
        # Check for added models
        if set(new_models.keys()) - set(old_models.keys()):
            added_models = set(new_models.keys()) - set(old_models.keys())
            self.logger.info(f"📝 Added models: {added_models}")
            return True
        
        # Check for added fields
        for model_name, model_data in new_models.items():
            if model_name in old_models:
                old_fields = old_models[model_name].get('fields', {})
                new_fields = model_data.get('fields', {})
                
                # Check for added fields
                added_fields = set(new_fields.keys()) - set(old_fields.keys())
                if added_fields:
                    self.logger.info(f"📝 Added fields in {model_name}: {added_fields}")
                    return True
                
                # Check for description changes
                for field_name, field_data in new_fields.items():
                    if field_name in old_fields:
                        old_desc = old_fields[field_name].get('description', '')
                        new_desc = field_data.get('description', '')
                        if old_desc != new_desc:
                            self.logger.info(f"📝 Description change in {model_name}.{field_name}")
                            return True
        
        return False
        
    def validate_contract(self, contract_dict: Dict[str, Any], filepath: Path) -> Dict[str, Any]:
        """Validate contract with simplified KPI-focused results."""
        try:
            self.logger.info("🔍 Validating contract...")
            
            # Normalize the contract data before validation
            normalized_contract = self._normalize_text_fields(contract_dict)
            
            data_contract = DataContract(
                data_contract_file=str(filepath),
                schema_location="validationtest.json"
            )
            
            # Run validation tests
            validation_result = data_contract.test()
            lint_result = None
            
            try:
                lint_result = data_contract.lint()
            except Exception as e:
                self.logger.warning(f"⚠️ Lint validation failed: {e}")
            
            # Build simplified KPI results
            results = self._build_simple_kpi_results(normalized_contract, validation_result, lint_result)
            
            self.logger.info(f"✅ Validation completed - Health Score: {results['health_score']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return self._build_error_kpi_results(contract_dict, str(e))
    
    def _build_simple_kpi_results(self, contract_dict: Dict[str, Any], validation_result, lint_result) -> Dict[str, Any]:
        """Build simplified KPI-focused validation results."""
        
        # Extract basic contract info
        info = contract_dict.get('info', {})
        contact = info.get('contact', {})
        models = contract_dict.get('models', {})
        
        # Process validation results - simplified
        validation_passed = self._is_validation_passed(validation_result)
        lint_passed = self._is_lint_passed(lint_result)
        
        # Count total fields across all models
        total_fields = sum(len(model.get('fields', {})) for model in models.values())
        
        # Check for missing descriptions
        has_missing_descriptions = self._has_missing_descriptions(contract_dict)
        
        # Calculate comprehensive health score (0-100)
        health_score = 100
        if not validation_passed:
            health_score -= 40
        if not lint_passed:
            health_score -= 25
        if has_missing_descriptions:
            health_score -= 35  # Deduct 35 points if any descriptions are missing
        
        return {
            "contract_id": contract_dict.get('id', 'unknown'),
            "contract_name": info.get('title', 'unknown'),
            "validation_timestamp": datetime.now().isoformat(),
            "health_score": max(0, round(health_score, 1)),
            "health_score_calculation": {
                "base_score": 100,
                "validation_deduction": 0 if validation_passed else 40,
                "lint_deduction": 0 if lint_passed else 25,
                "description_deduction": 35 if has_missing_descriptions else 0
            },
            "validation_passed": validation_passed,
            "lint_passed": lint_passed,
            "owner": contact.get('email', contact.get('name', 'unknown')),
            "total_models": len(models),
            "total_fields": total_fields,
            "has_examples": len(contract_dict.get('examples', [])) > 0
        }

    def _clean_description(self, description: Optional[str]) -> str:
        """Clean a description, treating None or 'null' as an empty string."""
        if description is None or description.strip().lower() == 'null':
            return ''
        return description

    def _build_error_kpi_results(self, contract_dict: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Build simplified error results."""
        info = contract_dict.get('info', {})
        contact = info.get('contact', {})
        models = contract_dict.get('models', {})
        
        return {
            "contract_id": contract_dict.get('id', 'unknown'),
            "contract_name": info.get('title', 'unknown'),
            "validation_timestamp": datetime.now().isoformat(),
            "health_score": 0,
            "health_score_calculation": {
                "base_score": 100,
                "validation_deduction": 40,
                "lint_deduction": 25,
                "description_deduction": 35
            },
            "validation_passed": False,
            "lint_passed": False,
            "owner": contact.get('email', contact.get('name', 'unknown')),
            "total_models": len(models),
            "total_fields": sum(len(model.get('fields', {})) for model in models.values()),
            "has_examples": len(contract_dict.get('examples', [])) > 0,
            "error": error_message
        }

    def _is_validation_passed(self, validation_result) -> bool:
        """Simple check if validation passed."""
        if hasattr(validation_result, 'results') and validation_result.results:
            return all(self._check_test_status(test) for test in validation_result.results)
        else:
            return self._check_test_status(validation_result)

    def _is_lint_passed(self, lint_result) -> bool:
        """Simple check if lint passed."""
        if not lint_result:
            return True
        
        if hasattr(lint_result, 'results') and lint_result.results:
            return all(self._check_test_status(test) for test in lint_result.results)
        else:
            return self._check_test_status(lint_result)
    
    def _build_error_results(self, contract_dict: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Build error results when validation fails completely."""
        info = contract_dict.get('info', {})
        contact = info.get('contact', {})
        servers = contract_dict.get('servers', {}).get('production', {})
        
        return {
            "contract_id": contract_dict.get('id', 'unknown'),
            "contract_name": self._extract_contract_name(info, servers),
            "validation_timestamp": datetime.now().isoformat(),
            "health_score": 0,
            "health_score_calculation": {
                "base_score": 100,
                "validation_deduction": 40,
                "lint_deduction": 25,
                "description_deduction": 35
            },
            "validation_passed": False,
            "lint_passed": False,
            "issues": 1,
            "risk_level": "CRITICAL",
            "owner": contact.get('email', contact.get('name', 'unknown')),
            "domain": self._extract_domain(servers, info),
            "environment": self._extract_environment(servers),
            "last_data_update": "unknown",
            "data_freshness_days": -1,
            "alerts": [f"Validation error: {error_message}"]
        }

    
    def _has_missing_descriptions(self, contract_dict: Dict[str, Any]) -> bool:
        """Check if any descriptions are missing (None or 'null')."""
        # Check main contract description
        info = contract_dict.get('info', {})
        if not self._clean_description(info.get('description')):
            return True
        
        # Check models and their fields
        models = contract_dict.get('models', {})
        for model_name, model_data in models.items():
            # Check model description
            if not self._clean_description(model_data.get('description')):
                return True
            
            # Check field descriptions
            fields = model_data.get('fields', {})
            for field_name, field_data in fields.items():
                if not self._clean_description(field_data.get('description')):
                    return True
        
        return False

    def _check_test_status(self, test_result) -> bool:
        """Check if a test/lint result passed."""
        if hasattr(test_result, 'result'):
            return test_result.result == "passed" or test_result.result is True
        elif hasattr(test_result, 'status'):
            return test_result.status == "passed" or test_result.status == "success"
        elif hasattr(test_result, 'passed'):
            return test_result.passed
        elif hasattr(test_result, 'success'):
            return test_result.success
        return False

    def save_validation_results(self, results: Dict[str, Any], contract_filepath: Path) -> Path:
        """Save business-relevant validation results to JSON file."""
        try:
            # Generate results filename
            contract_stem = contract_filepath.stem
            results_filename = f"{contract_stem}_report.json"
            results_filepath = contract_filepath.parent / results_filename
            
            # Save results with proper formatting
            with open(results_filepath, 'w', encoding='utf-8') as file:
                json.dump(results, file, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Health report saved to: {results_filepath}")
            return results_filepath
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save health report: {e}")
            raise


    def generate_and_validate_contract(self, identifier: str, output_dir: str = "DataContracts") -> tuple[Path, Path]:
        """Generate contract and validate it with SDK."""
        self.logger.info(f"🚀 Starting contract generation and validation for: {identifier}")
        
        # Generate contract
        contract_filepath = self.generate_draft_contract(identifier, output_dir)
        
        # Load and validate contract
        with open(contract_filepath, 'r', encoding='utf-8') as file:
            contract_dict = yaml.safe_load(file)
        
        validation_results = self.validate_contract(contract_dict, contract_filepath)
        results_filepath = self.save_validation_results(validation_results, contract_filepath)
        
        # Display combined summary
        self._display_summary(contract_dict, contract_filepath, validation_results)
        
        return contract_filepath, results_filepath

def main():
    """Main CLI function with validation."""
    try:
        print("🚀 Data Contract Generator with Validation")
        print("=" * 50)
        
        # Get user input
        identifier = input("Enter table full name (catalog.schema.table) or document ID: ").strip()
        if not identifier:
            print("❌ Identifier cannot be empty")
            return
        
        output_dir = input("Enter output directory (default: DataContracts): ").strip()
        if not output_dir:
            output_dir = "DataContracts"
        
        # Generate and validate contract
        generator = DataContractGenerator()
        contract_path, results_path = generator.generate_and_validate_contract(identifier, output_dir)
        
        print(f"\n🎉 Contract generated and validated successfully!")
        print(f"📄 Contract: {contract_path}")
        print(f"📊 Results: {results_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.getLogger(__name__).error(f"Application error: {e}", exc_info=True)


if __name__ == "__main__":
    main()