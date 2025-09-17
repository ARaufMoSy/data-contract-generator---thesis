# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Data Contract Generator - A Python application that generates and validates data contracts from Cosmos DB metadata and Databricks Unity Catalog schema. The application follows Clean Architecture principles with clear separation of concerns across 4 main layers.

## Development Commands

### Setup and Installation
```powershell
# Install dependencies
pip install -r requirements.txt

# Setup environment (create credentials.env file)
# Required environment variables:
# COSMOS_DB_ENDPOINT, COSMOS_DB_KEY, COSMOS_DB_DATABASE, COSMOS_DB_CONTAINER
# DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH
```

### Running the Application
```powershell
# Main application (clean architecture version)
python main.py

# Simple script version (single file)
python simplescript.py
```

### Report Generation
```powershell
# Generate data quality reports
python reports/generate_report.py
```

### Testing and Validation
```powershell
# The application includes built-in validation using datacontract-cli
# Validation runs automatically with health scoring (0-100%)
# Results are saved as JSON reports alongside YAML contracts
```

## Architecture Overview

### Clean Architecture Design
The codebase implements Clean Architecture with 4 distinct layers:

**1. Configuration Layer (`config.py`)**
- `Config` dataclass for environment management
- `setup_logging()` function for centralized logging configuration
- Handles credentials.env file loading

**2. Data Access Layer (`repositories.py`)**
- `MetadataRepository`: Cosmos DB operations and queries
- `SchemaRepository`: Databricks Unity Catalog schema extraction
- Abstract data sources behind repository patterns

**3. Business Logic Layer (`services.py`)**
- `ContractService`: Core contract generation and mapping logic
- `FileService`: File operations and semantic version management
- `ValidationService`: Contract validation and health scoring

**4. Presentation Layer (`main.py`)**
- `DataContractGenerator`: Main facade class coordinating all services
- `main()`: CLI interface for user interaction

### Key Architectural Patterns

**Dependency Injection**: All services receive logger and config through constructors
**Repository Pattern**: Data access is abstracted behind repository interfaces
**Service Layer**: Business logic is encapsulated in focused service classes
**Facade Pattern**: DataContractGenerator provides unified interface to complex subsystem

### Version Management Strategy
- Semantic versioning (major.minor.patch) based on schema changes
- Breaking changes (removed fields/models, type changes) → Major version bump
- Non-breaking changes (added fields/models) → Minor version bump
- No significant changes → Version remains same
- Files named as `{contract_name}_v{version}.yaml`

### Health Scoring Algorithm
Contract health is scored 0-100% based on:
- Validation tests (40% weight)
- Data completeness (30% weight) 
- Lint checks (15% weight)
- Documentation completeness (15% weight)

## Data Flow Architecture

1. **Metadata Fetching**: Cosmos DB queries by table name or document ID
2. **Schema Extraction**: Databricks Unity Catalog DESCRIBE TABLE operations
3. **Contract Mapping**: Transform raw metadata into Data Contract 1.1.0 format
4. **Version Management**: Compare with existing contracts for semantic versioning
5. **File Generation**: Save YAML contracts with normalized text encoding
6. **Validation**: Use datacontract-cli for schema and format validation
7. **Health Reporting**: Generate JSON reports with KPIs and health scores

## External Dependencies

### Data Sources
- **Azure Cosmos DB**: Metadata storage and queries
- **Databricks Unity Catalog**: Schema extraction and sample data

### Key Libraries
- `azure-cosmos`: Cosmos DB client
- `databricks-sql-connector`: Unity Catalog connectivity
- `datacontract-cli`: Contract validation framework
- `PyYAML`: YAML file generation
- `python-dotenv`: Environment configuration

## File Structure Context

```
├── main.py              # Clean architecture entry point (preferred)
├── simplescript.py      # Single-file version (1000+ lines)
├── config.py           # Configuration and logging setup
├── repositories.py     # Data access layer (Cosmos DB, Databricks)
├── services.py         # Business logic layer (contracts, files, validation)
├── requirements.txt    # Python dependencies
├── reports/           # Generated health reports and analysis
├── DataContracts/     # Generated YAML contracts (default output)
└── *.puml            # Architecture diagrams (PlantUML format)
```

## Development Guidelines

### Code Organization
- Use the clean architecture version (`main.py`) for new features
- `simplescript.py` is legacy single-file version - avoid modifications
- Each service class has single responsibility and clear boundaries
- All data access goes through repository classes

### Configuration Management
- All credentials in `credentials.env` file (not tracked in git)
- Use `Config.from_env()` for centralized configuration loading
- Environment variables follow COSMOS_DB_* and DATABRICKS_* patterns

### Error Handling
- Comprehensive logging at INFO level to files and console
- Graceful degradation when Databricks connection fails
- Validation errors result in health score penalties, not crashes

### Text Processing
- Unicode normalization for special characters in metadata
- Smart quote and encoding issue handling in `_normalize_text_fields()`
- YAML formatting with proper multi-line string handling

## Output Artifacts

### Generated Files
- **Contract YAML**: Complete data contract specification following DataContract 1.1.0
- **Health Report JSON**: Validation results, health scores, and quality metrics
- **Log Files**: Application execution logs for debugging

### Health Report Structure
Key metrics include: contract_id, health_score, validation_passed, total_models, total_fields, owner, validation_timestamp, and detailed health_score_calculation breakdown.

## Integration Points

- **Cosmos DB**: Query metadata by catalog.schema.table or document ID patterns
- **Databricks**: Extract live schema using DESCRIBE TABLE and sample data with LIMIT 1
- **DataContract CLI**: Validate generated contracts against schema definitions
