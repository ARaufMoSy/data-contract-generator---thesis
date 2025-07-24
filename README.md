# Data Contract Generator

A Python application that generates and validates data contracts from Cosmos DB metadata and Databricks Unity Catalog schema.

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Environment**
   Create `credentials.env` file:
   ```env
   COSMOS_DB_ENDPOINT=https://your-cosmos-account.documents.azure.com:443/
   COSMOS_DB_KEY=your-cosmos-primary-key
   COSMOS_DB_DATABASE=metadata
   COSMOS_DB_CONTAINER=table_metadata
   
   DATABRICKS_HOST=your-databricks-workspace.databricks.com
   DATABRICKS_TOKEN=your-databricks-token
   DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your-warehouse-id
   ```

3. **Run the Application**
   ```bash
   python main.py
   ```

## Project Structure

```
├── config.py           # Configuration management
├── repositories.py     # Data access (Cosmos DB & Databricks)
├── services.py         # Business logic
├── main.py            # CLI interface
└── DataContracts/     # Generated contracts
```

## Usage

The application will prompt you for:
1. **Identifier**: Table name (`catalog.schema.table`) or document ID
2. **Output Directory**: Where to save contracts (default: `DataContracts`)

Example:
```
Enter table full name or document ID: dfp_minerva_prd.towerdatahub.catalogue_towers_3x
Enter output directory (default: DataContracts): 

 Contract generated: Catalogue_Towers_3X v1.0.0
 Health Score: 100% | Status: HEALTHY

 Contract generated and validated successfully!
 Contract: DataContracts\Catalogue_Towers_3X_v1.0.0.yaml
 Health Report: DataContracts\Catalogue_Towers_3X_v1.0.0_report.json
```

## What It Does

1. **Fetches metadata** from Cosmos DB
2. **Extracts schema** from Databricks Unity Catalog
3. **Generates YAML contract** with real field definitions
4. **Validates contract** and creates health report
5. **Manages versions** automatically using semantic versioning

## Features

-  Clean Architecture with 4 focused files
-  Automatic version management
-  Health scoring (0-100%)
-  Real data examples
-  Error handling and logging
-  Professional YAML output

## Requirements

- Python 3.8+
- Access to Azure Cosmos DB
- Access to Databricks Unity Catalog
- Dependencies in `requirements.txt`

## Output Files

- **Contract YAML**: Complete data contract specification
- **Health Report JSON**: Validation results and quality metrics
