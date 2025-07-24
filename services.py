import os
import re
import yaml
import json
import logging
import unicodedata
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from packaging import version
from datacontract.data_contract import DataContract


class ContractService:
    """Core contract generation and mapping logic."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def map_to_contract(self, cosmos_metadata: Dict[str, Any], unity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map Cosmos DB metadata and Unity Catalog schema to Data Contract format."""
        self.logger.info("🔄 Mapping metadata to contract structure...")
        
        metadata = cosmos_metadata.get('metadata', {})
        storage_info = cosmos_metadata.get('storage_info', {})
        
        # Build contract sections in the correct order
        contract = {
            "dataContractSpecification": "1.1.0",
            "id": cosmos_metadata.get("id", ""),
            "info": self._build_info(metadata),
            "servers": self._build_servers(storage_info),
            "terms": self._extract_fields(metadata, ["usage", "limitations"]),
        }
        
        # Add models and examples from Unity Catalog
        contract["models"] = unity_data.get("models", {})
        contract["examples"] = unity_data.get("examples", [])
        
        # Add remaining sections
        contract.update({
            "quality": self._extract_fields(metadata, ["data_category", "data_format", "unique_keys"]),
            "servicelevels": self._build_service_levels(metadata),
            "links": {"dataPark": "https://data-products.prd.dis.siemensgamesa.com/"},
            "tags": metadata.get("tags", []),
            "definitions": {}
        })
        
        # Remove empty sections (except models and examples)
        contract = {k: v for k, v in contract.items() if v or k in ["models", "examples"]}
        
        self.logger.info("✅ Successfully mapped metadata to contract")
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
    
    def _extract_fields(self, source: Dict[str, Any], fields: List[str], 
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


class FileService:
    """File operations and version management."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._setup_yaml_formatting()
    
    def save_contract(self, contract_dict: Dict[str, Any], output_dir: str = "DataContracts") -> Path:
        """Save contract to YAML file with proper version management."""
        
        # Create output directory
        directory_path = Path(output_dir)
        directory_path.mkdir(exist_ok=True)
        
        # Generate base filename
        info = contract_dict.get('info', {})
        title = info.get('title', 'data_contract')
        base_filename = self._sanitize_filename(title)
        
        # Handle version management
        contract_dict = self._handle_version_management(contract_dict, directory_path)
        
        # Generate final filename with version
        version = contract_dict.get('info', {}).get('version', '1.0.0')
        filename = f"{base_filename}_v{version}"
        filepath = directory_path / f"{filename}.yaml"
        
        # Check if file already exists with this version
        if filepath.exists():
            self.logger.info(f"ℹ️ Contract version {version} already exists. No changes needed.")
            return filepath
        
        # Save file with new version
        try:
            normalized_contract = self._normalize_text_fields(contract_dict)
            
            with open(filepath, 'w', encoding='utf-8', newline='\n') as file:
                yaml.dump(normalized_contract, file, sort_keys=False, allow_unicode=True, 
                        default_flow_style=False, width=1000, indent=2)
            
            self.logger.info(f"✅ Contract saved to: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save contract: {e}")
            raise
    
    def _handle_version_management(self, contract_dict: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Handle version management by comparing with the latest version contract."""
        
        latest_contract_path = self._find_latest_version_file(contract_dict, output_dir)
        
        if not latest_contract_path:
            self.logger.info("🆕 No previous contract found. Using initial version.")
            return contract_dict
        
        try:
            # Load the latest contract for comparison
            with open(latest_contract_path, 'r', encoding='utf-8') as file:
                latest_contract_dict = yaml.safe_load(file)
            
            # Get the latest version
            latest_version = latest_contract_dict.get("info", {}).get("version", "1.0.0")
            self.logger.info(f"🔍 Comparing against latest version: {latest_version}")
            
            # Determine version increment based on changes
            new_version = self._determine_version_increment(latest_contract_dict, contract_dict, latest_version)
            
            # Update contract with new version
            contract_dict["info"]["version"] = new_version
            self.logger.info(f"🔄 Version updated from {latest_version} to {new_version}")
            
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
                filename = file_path.stem
                if '_v' in filename:
                    version_part = filename.split('_v')[-1]
                    try:
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
            
            # Check for non-breaking changes
            if self._has_minor_changes(old_models, new_models):
                return f"{major}.{minor + 1}.0"
            
            # No significant changes detected
            self.logger.info("✅ No significant schema changes detected. Version remains the same.")
            return current_version
            
        except Exception as e:
            self.logger.error(f"❌ Version increment determination failed: {e}")
            return current_version
    
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
        
        return False
    
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


class ValidationService:
    """Contract validation operations."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
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
            
            self.logger.info(f"✅ Validation completed - Status: {results['status']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return self._build_error_kpi_results(contract_dict, str(e))
    
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
        
        # Calculate simple health score (0-100)
        health_score = 100
        if not validation_passed:
            health_score -= 50
        if not lint_passed:
            health_score -= 25
        
        # Determine simple status
        if health_score >= 75:
            status = "HEALTHY"
        elif health_score >= 50:
            status = "WARNING" 
        else:
            status = "CRITICAL"
        
        return {
            "contract_id": contract_dict.get('id', 'unknown'),
            "contract_name": info.get('title', 'unknown'),
            "validation_timestamp": datetime.now().isoformat(),
            "health_score": max(0, health_score),
            "status": status,
            "validation_passed": validation_passed,
            "lint_passed": lint_passed,
            "owner": contact.get('email', contact.get('name', 'unknown')),
            "total_models": len(models),
            "total_fields": total_fields,
            "has_examples": len(contract_dict.get('examples', [])) > 0
        }

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
            "status": "ERROR",
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
    
    def _normalize_text_fields(self, data: Any) -> Any:
        """Recursively normalize text fields to handle encoding issues."""
        if isinstance(data, str):
            normalized = unicodedata.normalize('NFKD', data)
            replacements = {
                '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
                '\u2013': '-', '\u2014': '--', '\u2026': '...', '\u00a0': ' ',
            }
            for old, new in replacements.items():
                normalized = normalized.replace(old, new)
            return ''.join(c for c in normalized if ord(c) < 127 or c.isalnum())
        elif isinstance(data, dict):
            return {key: self._normalize_text_fields(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._normalize_text_fields(item) for item in data]
        else:
            return data
