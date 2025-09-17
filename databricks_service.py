import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import os


class DatabricksService:
    """Service for writing data contract validation results to Databricks table."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.table_name = "dfp_minerva_prd.datacontracts.datacontracts_overview"
        self._spark = None
        self._sql_connector = None
        
    def _get_spark_session(self):
        """Get or create Spark session for Databricks."""
        if self._spark is None:
            try:
                from pyspark.sql import SparkSession
                self._spark = SparkSession.builder.getOrCreate()
                self.logger.info("✅ Connected to Databricks via Spark session")
            except ImportError as e:
                self.logger.error(f"❌ PySpark not available: {e}")
                raise ImportError("PySpark is required for Databricks integration. Install with: pip install pyspark")
        return self._spark
    
    def _get_sql_connector(self):
        """Get SQL connector for Databricks (alternative to Spark)."""
        if self._sql_connector is None:
            try:
                from databricks import sql
                self._sql_connector = sql.connect(
                    server_hostname=os.getenv('DATABRICKS_SERVER_HOSTNAME'),
                    http_path=os.getenv('DATABRICKS_HTTP_PATH'),
                    access_token=os.getenv('DATABRICKS_ACCESS_TOKEN')
                )
                self.logger.info("✅ Connected to Databricks via SQL connector")
            except ImportError as e:
                self.logger.error(f"❌ Databricks SQL connector not available: {e}")
                raise ImportError("Databricks SQL connector is required. Install with: pip install databricks-sql-connector")
            except Exception as e:
                self.logger.error(f"❌ Failed to connect to Databricks: {e}")
                raise
        return self._sql_connector
    
    def write_validation_result_to_databricks(self, validation_result: Dict[str, Any], 
                                            contract_filepath: str) -> bool:
        """Write a single validation result to Databricks table."""
        try:
            # Transform validation result to match table schema
            table_row = self._transform_validation_result(validation_result, contract_filepath)
            
            # Try Spark first, fall back to SQL connector
            try:
                return self._write_via_spark([table_row])
            except Exception as spark_error:
                self.logger.warning(f"⚠️ Spark write failed, trying SQL connector: {spark_error}")
                return self._write_via_sql_connector([table_row])
                
        except Exception as e:
            self.logger.error(f"❌ Failed to write validation result to Databricks: {e}")
            return False
    
    def write_multiple_results_to_databricks(self, validation_results: List[Dict[str, Any]], 
                                           contract_filepaths: List[str]) -> bool:
        """Write multiple validation results to Databricks table in batch."""
        try:
            # Transform all results
            table_rows = []
            for result, filepath in zip(validation_results, contract_filepaths):
                table_row = self._transform_validation_result(result, filepath)
                table_rows.append(table_row)
            
            # Write in batch
            try:
                return self._write_via_spark(table_rows)
            except Exception as spark_error:
                self.logger.warning(f"⚠️ Spark batch write failed, trying SQL connector: {spark_error}")
                return self._write_via_sql_connector(table_rows)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to write batch validation results to Databricks: {e}")
            return False
    
    def _transform_validation_result(self, result: Dict[str, Any], filepath: str) -> Dict[str, Any]:
        """Transform validation result to match Databricks table schema."""
        
        # Extract contract name from filepath (version not used in current schema)
        from pathlib import Path
        contract_name = Path(filepath).stem
        
        # Map validation result fields to table columns (all calculated fields from JSON report)
        table_row = {
            # Primary identifiers  
            'contract_id': result.get('contract_id', 'unknown'),
            'contract_schema': result.get('contract_schema', 'unknown'),
            'owner': result.get('owner', 'unknown'),
            
            # Timestamps
            'validation_timestamp': result.get('validation_timestamp', datetime.now().isoformat()),
            
            # Health metrics (converted to integers for BIGINT columns)
            'health_score': int(round(result.get('health_score', 0))),
            
            # Validation scores breakdown (from health_score_calculation, converted to integers)
            'validation_score': int(round(result.get('health_score_calculation', {}).get('validation_score', 0))),
            'completeness_score': int(round(result.get('health_score_calculation', {}).get('completeness_score', 0))),
            'lint_score': int(round(result.get('health_score_calculation', {}).get('lint_score', 0))),
            'documentation_score': int(round(result.get('health_score_calculation', {}).get('documentation_score', 0))),
            
            # Technical metrics
            'validation_passed': bool(result.get('validation_passed', False)),
            'total_models': int(result.get('total_models', 0)),
            'total_fields': int(result.get('total_fields', 0)),
            'empty_fields_count': int(result.get('empty_fields_count', 0)),
            'has_examples': bool(result.get('has_examples', False))
        }
        
        return table_row
    
    def _write_via_spark(self, table_rows: List[Dict[str, Any]]) -> bool:
        """Write data to Databricks using Spark DataFrame."""
        try:
            spark = self._get_spark_session()
            
            # Create DataFrame
            df = spark.createDataFrame(table_rows)
            
            # Write to table (append mode)
            df.write \
              .mode("append") \
              .option("mergeSchema", "true") \
              .saveAsTable(self.table_name)
            
            self.logger.info(f"✅ Successfully wrote {len(table_rows)} rows to {self.table_name} via Spark")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Spark write failed: {e}")
            raise
    
    def _write_via_sql_connector(self, table_rows: List[Dict[str, Any]]) -> bool:
        """Write data to Databricks using SQL connector."""
        try:
            connection = self._get_sql_connector()
            cursor = connection.cursor()
            
            # Prepare INSERT statement
            if not table_rows:
                return True
                
            # Get column names from first row
            columns = list(table_rows[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            column_names = ', '.join(columns)
            
            insert_sql = f"""
            INSERT INTO {self.table_name} ({column_names}) 
            VALUES ({placeholders})
            """
            
            # Prepare data for insertion
            rows_data = []
            for row in table_rows:
                row_values = [row.get(col) for col in columns]
                rows_data.append(row_values)
            
            # Execute batch insert
            cursor.executemany(insert_sql, rows_data)
            cursor.close()
            
            self.logger.info(f"✅ Successfully wrote {len(table_rows)} rows to {self.table_name} via SQL connector")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ SQL connector write failed: {e}")
            raise
    
    def read_validation_results_from_databricks(self, days_back: int = 30) -> pd.DataFrame:
        """Read validation results from Databricks table for report generation."""
        try:
            # Try Spark first
            try:
                spark = self._get_spark_session()
                query = f"""
                SELECT * FROM {self.table_name} 
                WHERE validation_timestamp >= date_sub(current_date(), {days_back})
                ORDER BY validation_timestamp DESC
                """
                df = spark.sql(query).toPandas()
                self.logger.info(f"✅ Read {len(df)} records from {self.table_name} via Spark")
                return df
                
            except Exception as spark_error:
                self.logger.warning(f"⚠️ Spark read failed, trying SQL connector: {spark_error}")
                return self._read_via_sql_connector(days_back)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to read from Databricks: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def _read_via_sql_connector(self, days_back: int) -> pd.DataFrame:
        """Read data from Databricks using SQL connector."""
        try:
            connection = self._get_sql_connector()
            
            query = f"""
            SELECT * FROM {self.table_name} 
            WHERE validation_timestamp >= date_sub(current_date(), {days_back})
            ORDER BY validation_timestamp DESC
            """
            
            df = pd.read_sql(query, connection)
            self.logger.info(f"✅ Read {len(df)} records from {self.table_name} via SQL connector")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ SQL connector read failed: {e}")
            raise
    
    def create_table_if_not_exists(self) -> bool:
        """Create the Databricks table if it doesn't exist."""
        try:
            # Table schema matching your actual table structure
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                contract_id BIGINT,
                contract_schema STRING,
                owner STRING,
                validation_timestamp TIMESTAMP,
                health_score BIGINT,
                validation_score BIGINT,
                completeness_score BIGINT,
                lint_score BIGINT,
                documentation_score BIGINT,
                validation_passed BOOLEAN,
                total_models BIGINT,
                total_fields BIGINT,
                empty_fields_count BIGINT,
                has_examples BOOLEAN
            ) USING DELTA
            TBLPROPERTIES (
                'description' = 'Data Contract Validation Overview - replaces JSON reports',
                'created_by' = 'data_contract_validation_pipeline'
            )
            """
            
            # Try both connection methods
            try:
                spark = self._get_spark_session()
                spark.sql(create_table_sql)
                self.logger.info(f"✅ Table {self.table_name} created/verified via Spark")
                return True
            except Exception as spark_error:
                self.logger.warning(f"⚠️ Spark table creation failed, trying SQL connector: {spark_error}")
                connection = self._get_sql_connector()
                cursor = connection.cursor()
                cursor.execute(create_table_sql)
                cursor.close()
                self.logger.info(f"✅ Table {self.table_name} created/verified via SQL connector")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create table {self.table_name}: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test connection to Databricks."""
        try:
            # Try reading a simple query
            try:
                spark = self._get_spark_session()
                spark.sql("SELECT 1 as test").collect()
                self.logger.info("✅ Databricks Spark connection successful")
                return True
            except Exception:
                connection = self._get_sql_connector()
                cursor = connection.cursor()
                cursor.execute("SELECT 1 as test")
                cursor.fetchall()
                cursor.close()
                self.logger.info("✅ Databricks SQL connection successful")
                return True
        except Exception as e:
            self.logger.error(f"❌ Databricks connection test failed: {e}")
            return False
