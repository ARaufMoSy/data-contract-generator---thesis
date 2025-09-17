#!/usr/bin/env python3
"""
Test script for Databricks integration with data contract validation.
This script demonstrates how to:
1. Test Databricks connectivity
2. Create the table if needed
3. Validate contracts and write results to Databricks
4. Read data back from Databricks for verification
"""

import logging
import sys
import os
from pathlib import Path
from services import ValidationService, ContractService
from databricks_service import DatabricksService
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_databricks_connection():
    """Test basic Databricks connectivity."""
    logger.info("🔍 Testing Databricks connection...")
    
    try:
        databricks_service = DatabricksService(logger=logger)
        
        # Test connection
        if databricks_service.test_connection():
            logger.info("✅ Databricks connection successful!")
            return databricks_service
        else:
            logger.error("❌ Databricks connection failed!")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error testing Databricks connection: {e}")
        return None

def create_databricks_table():
    """Create the Databricks table if it doesn't exist."""
    logger.info("🏗️  Creating/verifying Databricks table...")
    
    try:
        databricks_service = DatabricksService(logger=logger)
        
        if databricks_service.create_table_if_not_exists():
            logger.info("✅ Databricks table ready!")
            return True
        else:
            logger.error("❌ Failed to create/verify Databricks table!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error creating Databricks table: {e}")
        return False

def validate_and_save_contracts():
    """Validate contracts and save results to Databricks."""
    logger.info("🔍 Validating contracts and saving to Databricks...")
    
    try:
        # Initialize services
        contract_service = ContractService(logger=logger)
        validation_service = ValidationService(logger=logger, use_databricks=True)
        
        # Find contract files in DataContracts folder
        contracts_dir = Path("DataContracts")
        if not contracts_dir.exists():
            logger.error("❌ DataContracts folder not found!")
            return False
            
        contract_files = list(contracts_dir.glob("*.yaml")) + list(contracts_dir.glob("*.yml"))
        
        if not contract_files:
            logger.warning("⚠️ No contract files found in DataContracts directory")
            return False
        
        # Use only the first contract for testing
        contract_files = contract_files[:1]
        logger.info(f"📋 Testing with 1 contract file: {contract_files[0].name}")
        
        success_count = 0
        total_count = len(contract_files)
        
        for contract_file in contract_files:
            try:
                logger.info(f"📝 Processing: {contract_file.name}")
                
                # Load contract YAML file
                try:
                    with open(contract_file, 'r', encoding='utf-8') as file:
                        import yaml
                        contract_dict = yaml.safe_load(file)
                except Exception as load_error:
                    logger.error(f"❌ Failed to load contract {contract_file.name}: {load_error}")
                    continue
                
                if not contract_dict:
                    logger.error(f"❌ Empty contract file: {contract_file.name}")
                    continue
                
                # Validate contract
                validation_result = validation_service.validate_contract(contract_dict, contract_file)
                
                # Save to Databricks
                if validation_service.save_validation_results(validation_result, contract_file):
                    success_count += 1
                    logger.info(f"✅ Successfully processed: {contract_file.name}")
                else:
                    logger.error(f"❌ Failed to save results for: {contract_file.name}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {contract_file.name}: {e}")
        
        logger.info(f"📊 Processing complete: {success_count}/{total_count} contracts successfully saved to Databricks")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"❌ Error validating contracts: {e}")
        return False

def verify_databricks_data():
    """Verify data was written to Databricks by reading it back."""
    logger.info("🔍 Verifying data in Databricks...")
    
    try:
        databricks_service = DatabricksService(logger=logger)
        
        # Read recent data from Databricks
        df = databricks_service.read_validation_results_from_databricks(days_back=1)
        
        if df.empty:
            logger.warning("⚠️ No data found in Databricks table")
            return False
        
        logger.info(f"✅ Found {len(df)} records in Databricks table")
        
        # Display summary statistics
        if not df.empty:
            logger.info("📊 Data Summary:")
            logger.info(f"   • Contracts: {df['contract_name'].nunique()}")
            logger.info(f"   • Average Health Score: {df['health_score'].mean():.1f}")
            logger.info(f"   • Domains: {df['contract_schema'].nunique()}")
            logger.info(f"   • Latest Validation: {df['validation_timestamp'].max()}")
            
            # Show top 5 contracts by health score
            logger.info("🏆 Top 5 Contracts by Health Score:")
            top_contracts = df.nlargest(5, 'health_score')[['contract_name', 'health_score', 'status']]
            for _, row in top_contracts.iterrows():
                logger.info(f"   • {row['contract_name']}: {row['health_score']:.1f} ({row['status']})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error verifying Databricks data: {e}")
        return False

def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("🚀 Starting Databricks Integration Test")
    logger.info("=" * 60)
    
    # Load configuration from config file
    try:
        config = Config.from_env()
        
        # Set environment variables for Databricks service
        if config.databricks_host:
            os.environ['DATABRICKS_SERVER_HOSTNAME'] = config.databricks_host
        if config.databricks_token:
            os.environ['DATABRICKS_ACCESS_TOKEN'] = config.databricks_token
        if config.databricks_http_path:
            os.environ['DATABRICKS_HTTP_PATH'] = config.databricks_http_path
        
        # Check if Databricks config is available
        if not all([config.databricks_host, config.databricks_token, config.databricks_http_path]):
            logger.error("❌ Databricks configuration incomplete in credentials.env")
            logger.info("💡 Please ensure these are set in credentials.env:")
            logger.info("   DATABRICKS_HOST=<your_databricks_host>")
            logger.info("   DATABRICKS_TOKEN=<your_access_token>")
            logger.info("   DATABRICKS_HTTP_PATH=<your_http_path>")
            sys.exit(1)
            
        logger.info(f"✅ Loaded Databricks config from credentials.env: {config.databricks_host}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        logger.info("💡 Please ensure credentials.env file exists with Databricks configuration")
        sys.exit(1)
    
    try:
        # Step 1: Test connectivity
        logger.info("\n📡 Step 1: Testing Databricks connectivity")
        if not test_databricks_connection():
            logger.error("❌ Cannot proceed without Databricks connection")
            sys.exit(1)
        
        # Step 2: Create table
        logger.info("\n🏗️  Step 2: Creating/verifying Databricks table")
        if not create_databricks_table():
            logger.error("❌ Cannot proceed without Databricks table")
            sys.exit(1)
        
        # Step 3: Validate and save contracts
        logger.info("\n📝 Step 3: Validating contracts and saving to Databricks")
        if not validate_and_save_contracts():
            logger.error("❌ Failed to validate and save contracts")
            sys.exit(1)
        
        # Step 4: Verify data
        logger.info("\n🔍 Step 4: Verifying data in Databricks")
        if not verify_databricks_data():
            logger.warning("⚠️ Data verification had issues")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Databricks Integration Test Complete!")
        logger.info("=" * 60)
        logger.info("\n💡 Next Steps:")
        logger.info("   1. Check your Databricks table: dfp_minerva_prd.datacontracts.datacontracts_overview")
        logger.info("   2. Query the data using Databricks SQL or notebooks")
        logger.info("   3. Update your report generation to read from this table")
        logger.info("   4. Set up automated pipelines to run validation regularly")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
