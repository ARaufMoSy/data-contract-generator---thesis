import yaml
import logging
from pathlib import Path
from typing import Tuple

from config import Config, setup_logging
from repositories import MetadataRepository, SchemaRepository
from services import ContractService, FileService, ValidationService


class DataContractGenerator:
    """Main orchestrator that composes all services using dependency injection."""
    
    def __init__(self):
        """Initialize generator with all dependencies."""
        # Setup logging and configuration
        self.logger = setup_logging()
        self.config = Config.from_env()
        
        # Initialize repositories (data access layer)
        self.metadata_repo = MetadataRepository(self.config, self.logger)
        self.schema_repo = SchemaRepository(self.config, self.logger)
        
        # Initialize services (business logic layer)
        self.contract_service = ContractService(self.logger)
        self.file_service = FileService(self.logger)
        self.validation_service = ValidationService(self.logger)
        
        self.logger.info("✅ Data Contract Generator initialized successfully")
    
    def generate_contract(self, identifier: str, output_dir: str = "DataContracts") -> Path:
        """Generate complete contract from identifier with Unity Catalog schema."""
        self.logger.info(f"🚀 Starting contract generation for: {identifier}")
        
        # Step 1: Fetch metadata from Cosmos DB
        metadata = self.metadata_repo.fetch_metadata(identifier)
        if not metadata:
            raise ValueError(f"No metadata found for identifier: {identifier}")
        
        # Step 2: Determine table name for schema extraction
        table_name = self._determine_table_name(identifier, metadata)
        
        # Step 3: Extract Unity Catalog schema and examples
        unity_data = {"models": {}, "examples": []}
        if table_name:
            unity_data = self.schema_repo.extract_unity_catalog_schema(table_name)
        
        # Step 4: Map to contract format
        contract = self.contract_service.map_to_contract(metadata, unity_data)
        
        # Step 5: Save contract with version management
        filepath = self.file_service.save_contract(contract, output_dir)
        
        # Step 6: Display summary
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as file:
                final_contract = yaml.safe_load(file)
            self._display_summary(final_contract, filepath)
        
        return filepath
    
    def generate_and_validate_contract(self, identifier: str, output_dir: str = "DataContracts") -> Tuple[Path, Path]:
        """Generate contract and validate it with comprehensive health report."""
        self.logger.info(f"🚀 Starting contract generation and validation for: {identifier}")
        
        # Generate contract
        contract_filepath = self.generate_contract(identifier, output_dir)
        
        # Load and validate contract
        with open(contract_filepath, 'r', encoding='utf-8') as file:
            contract_dict = yaml.safe_load(file)
        
        # Validate contract and save results
        validation_results = self.validation_service.validate_contract(contract_dict, contract_filepath)
        results_filepath = self.validation_service.save_validation_results(validation_results, contract_filepath)
        
        # Display combined summary
        self._display_summary(contract_dict, contract_filepath, validation_results)
        
        return contract_filepath, results_filepath
    
    def _determine_table_name(self, identifier: str, metadata: dict) -> str:
        """Determine the full table name for Unity Catalog schema extraction."""
        # If identifier is already a table name
        if '.' in identifier and identifier.count('.') == 2:
            return identifier
        
        # Extract from storage_info if available
        storage_info = metadata.get('storage_info', {})
        if all(key in storage_info for key in ['name', 'schema_name', 'table_name']):
            return f"{storage_info['name']}.{storage_info['schema_name']}.{storage_info['table_name']}"
        
        return ""
    
    def _display_summary(self, contract: dict, filepath: Path, validation_results: dict = None) -> None:
        """Display contract generation summary."""
        info = contract.get('info', {})
        
        print(f"✅ Contract generated: {info.get('title', 'Unknown')} v{info.get('version', 'Unknown')}")
        
        if validation_results:
            print(f"📊 Health Score: {validation_results['health_score']}% | Status: {validation_results['status']}")
            print(f"📄 Results saved to: {filepath.parent / f'{filepath.stem}_report.json'}")


def main():
    """Main CLI function with user interaction."""
    try:
        print("🚀 Data Contract Generator with Clean Architecture")
        print("=" * 55)
        
        # Get user input
        identifier = input("Enter table full name (catalog.schema.table) or document ID: ").strip()
        if not identifier:
            print("❌ Identifier cannot be empty")
            return
        
        output_dir = input("Enter output directory (default: DataContracts): ").strip()
        if not output_dir:
            output_dir = "DataContracts"
        
        # Initialize generator and execute (always with validation)
        generator = DataContractGenerator()
        
        # Generate contract with validation
        contract_path, results_path = generator.generate_and_validate_contract(identifier, output_dir)
        print(f"\n🎉 Contract generated and validated successfully!")
        print(f"📄 Contract: {contract_path}")
        print(f"📊 Health Report: {results_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.getLogger(__name__).error(f"Application error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
