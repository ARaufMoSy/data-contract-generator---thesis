import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def load_all_reports(reports_dir):
    """Load all JSON reports from the reports directory with filename tracking"""
    reports_data = []
    reports_dir = Path(reports_dir)
    
    for json_file in reports_dir.glob("*_report.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Add filename to data for version extraction
                data['_filename'] = json_file.name
                reports_data.append(data)
                print(f"Loaded: {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
    
    return reports_data

def create_dataframe(reports_data):
    """Convert reports data to pandas DataFrame with proper domain extraction and version tracking"""
    df_data = []
    
    for i, report in enumerate(reports_data):
        # Extract version from filename if available (passed from load function)
        filename = report.get('_filename', '')
        version = extract_version_from_filename(filename)
        contract_name = extract_contract_name_from_filename(filename)
        
        row = {
            'contract_id': report.get('contract_id', ''),
            'contract_name': report.get('contract_name', contract_name),
            'filename': filename,
            'version': version,
            'owner': report.get('owner', ''),
            'health_score': report.get('health_score', 0),
            'validation_score': report.get('health_score_calculation', {}).get('validation_score', 0),
            'completeness_score': report.get('health_score_calculation', {}).get('completeness_score', 0),
            'lint_score': report.get('health_score_calculation', {}).get('lint_score', 0),
            'documentation_score': report.get('health_score_calculation', {}).get('documentation_score', 0),
            'validation_passed': report.get('validation_passed', False),
            'total_models': report.get('total_models', 0),
            'total_fields': report.get('total_fields', 0),
            'empty_fields_count': report.get('empty_fields_count', 0),
            'has_examples': report.get('has_examples', False),
        }
        
        # Handle timestamp parsing with proper error handling
        timestamp_str = report.get('validation_timestamp', '')
        try:
            if timestamp_str:
                # Parse ISO format timestamp
                row['validation_timestamp'] = pd.to_datetime(timestamp_str)
            else:
                row['validation_timestamp'] = pd.NaT  # Not a Time (pandas null timestamp)
        except (ValueError, TypeError):
            print(f"Warning: Could not parse timestamp '{timestamp_str}' for contract {row['contract_name']}")
            row['validation_timestamp'] = pd.NaT
        
        # Calculate field completeness percentage
        row['completeness_percentage'] = (
            (row['total_fields'] - row['empty_fields_count']) / max(row['total_fields'], 1) * 100
        )
        
        # Extract domain properly from contract_schema or contract_name
        domain = 'unknown'
        
        # First try contract_schema field (most reliable)
        if 'contract_schema' in report and report['contract_schema']:
            domain = report['contract_schema'].lower().strip()
        else:
            # Fall back to extracting from contract name
            contract_name = row['contract_name'].lower() if row['contract_name'] else ''
            
            # Map common patterns to domains based on actual contract names
            domain_patterns = {
                'salesforce': ['salesforce', 'opportunity', 'proposal', 'quote'],
                'towerdatahub': ['catalogue_towers', 'tower'],
                'sst': ['sst_', 'site_specific'],
                'projectconfig': ['projectconfig', 'config'],
                'zcv': ['zcv_', 'dfp_cash', 'dfp_free'],
                'datatypes': ['datatypes'],
                'stp': ['stp_time'],
                'sap_bw': ['sap_', 'bw_'],
                'towerselect': ['towerselect']
            }
            
            for domain_key, patterns in domain_patterns.items():
                if any(pattern in contract_name for pattern in patterns):
                    domain = domain_key
                    break
        
        row['domain'] = domain
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Sort by contract name and timestamp for proper version tracking
    df = df.sort_values(['contract_name', 'validation_timestamp'], na_position='last')
    
    # Add derived columns for trend analysis
    if not df['validation_timestamp'].isna().all():
        # Add time-based features
        df['report_date'] = df['validation_timestamp'].dt.date
        df['report_month'] = df['validation_timestamp'].dt.to_period('M')
        df['report_week'] = df['validation_timestamp'].dt.to_period('W')
        df['report_hour'] = df['validation_timestamp'].dt.hour
        df['report_day_of_week'] = df['validation_timestamp'].dt.day_name()
        
        # Calculate days since first report for trend analysis
        first_date = df['validation_timestamp'].min()
        df['days_since_start'] = (df['validation_timestamp'] - first_date).dt.days
    
    # Add health score categorization
    df['health_category'] = pd.cut(df['health_score'], 
                                   bins=[0, 40, 60, 80, 100],
                                   labels=['Critical', 'Poor', 'Good', 'Excellent'])
    
    # Add field completeness categorization
    df['completeness_category'] = pd.cut(df['completeness_percentage'],
                                         bins=[0, 25, 50, 75, 100],
                                         labels=['Very Low', 'Low', 'Medium', 'High'])
    
    # Calculate composite quality score
    df['composite_quality_score'] = (
        df['validation_score'] * 0.3 +
        df['completeness_score'] * 0.3 +
        df['lint_score'] * 0.2 +
        df['documentation_score'] * 0.2
    )
    
    # Add flags for issues
    df['has_validation_issues'] = ~df['validation_passed']
    df['has_documentation_issues'] = df['documentation_score'] < 50
    df['has_completeness_issues'] = df['completeness_percentage'] < 70
    df['needs_urgent_attention'] = (df['health_score'] < 50) | df['has_validation_issues']
    
    # Group by contract to identify different versions
    contract_versions = df.groupby('contract_name').agg({
        'version': 'nunique',
        'health_score': ['min', 'max', 'mean'],
        'validation_timestamp': ['min', 'max']
    }).round(2)
    
    print(f"Created DataFrame with {len(df)} records")
    print(f"Date range: {df['validation_timestamp'].min()} to {df['validation_timestamp'].max()}")
    print(f"Domains found: {df['domain'].unique().tolist()}")
    print(f"Contracts with multiple versions: {len(contract_versions[contract_versions[('version', 'nunique')] > 1])}")
    
    # Debug: Show domain counts
    domain_counts = df['domain'].value_counts()
    print(f"Domain distribution: {domain_counts.to_dict()}")
    
    return df

def extract_version_from_filename(filename):
    """Extract version from filename like 'Contract_v1.0.0_report.json'"""
    import re
    if not filename:
        return '1.0.0'
    
    # Look for version pattern like v1.0.0, v1.1.0, etc.
    version_match = re.search(r'_v(\d+\.\d+\.\d+)_', filename)
    if version_match:
        return version_match.group(1)
    return '1.0.0'

def extract_contract_name_from_filename(filename):
    """Extract contract name from filename"""
    if not filename:
        return 'unknown'
    
    # Remove .json and _report.json
    name = filename.replace('_report.json', '').replace('.json', '')
    
    # Remove version part
    import re
    name = re.sub(r'_v\d+\.\d+\.\d+', '', name)
    
    return name

def generate_comprehensive_csv(df, output_dir='.'):
    """Generate only the main detailed CSV file for analysis (single file output)"""
    from pathlib import Path
    from datetime import datetime
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Only export the main detailed data CSV
    main_csv_path = output_dir / f'data_quality_detailed_analysis_{timestamp}.csv'
    # Select columns to match your sample CSV
    columns = [
        'contract_name', 'owner', 'health_score', 'validation_score', 'completeness_score',
        'lint_score', 'documentation_score', 'validation_passed', 'total_models',
        'total_fields', 'empty_fields_count', 'has_examples', 'completeness_percentage',
        # If you want empty_fields_percentage and domain, add them if available
    ]
    # Add columns if they exist in the DataFrame
    for col in ['empty_fields_percentage', 'domain']:
        if col in df.columns and col not in columns:
            columns.append(col)

    df[columns].to_csv(main_csv_path, index=False)
    print(f"✓ Generated single detailed CSV: {main_csv_path}")

    return {'main': main_csv_path}

def create_comprehensive_report(df):
    """Create a single comprehensive report with all requested visualizations including time series"""
    
    # Set up the corporate color scheme
    colors = {
        'primary_green': '#009B77',    # RGB(0, 155, 119)
        'secondary_gray': '#A7A8AA',   # RGB(167, 168, 170)
        'accent_blue': '#0072B8',      # RGB(0, 114, 184)
        'light_gray': '#F0F0F0',       # RGB(240, 240, 240)
        'dark_gray': '#4D4D4D',        # RGB(77, 77, 77)
        'warning_red': '#e74c3c'       # RGB(231, 76, 60)
    }
    
    # Create figure with updated layout to accommodate time series
    fig = plt.figure(figsize=(16, 24))  # Increased height
    fig.patch.set_facecolor('white')
    
    # Add main title
    fig.suptitle('Data Contract Quality Report - Siemens Energy', 
                 fontsize=24, fontweight='bold', color=colors['dark_gray'], y=0.98)
    fig.text(0.5, 0.955, f'Analysis of {len(df)} Data Contracts | September 2025', 
             ha='center', fontsize=14, color=colors['secondary_gray'])
    
    # Calculate key metrics
    overall_health = df['health_score'].mean()
    validation_pass_rate = (df['validation_passed'].sum() / len(df)) * 100
    field_completeness = df['completeness_percentage'].mean()
    description_coverage = (df['documentation_score'] > 0).sum() / len(df) * 100
    
    # 1. TOP SECTION - Key Metrics Cards (4 cards in a row)
    card_height = 0.06
    card_width = 0.18
    card_y = 0.87
    card_spacing = 0.2
    
    metrics = [
        ('Overall Health Score', f'{overall_health:.1f}', colors['primary_green']),
        ('Validation Pass Rate', f'{validation_pass_rate:.1f}%', colors['accent_blue']),
        ('Field Completeness', f'{field_completeness:.1f}%', colors['secondary_gray']),
        ('Description Coverage', f'{description_coverage:.1f}%', colors['dark_gray'])
    ]
    
    for i, (label, value, color) in enumerate(metrics):
        ax = fig.add_axes([0.1 + i * card_spacing, card_y, card_width, card_height])
        ax.text(0.5, 0.7, value, ha='center', va='center', fontsize=28, fontweight='bold', color=color)
        ax.text(0.5, 0.3, label, ha='center', va='center', fontsize=12, color=colors['dark_gray'])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        # Add border
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(color)
            ax.spines[spine].set_linewidth(3)
    
    # 2. Domain Performance Bar Chart
    ax1 = plt.subplot2grid((24, 4), (2, 0), colspan=4, rowspan=3)
    domain_health = df.groupby('domain')['health_score'].mean().sort_values(ascending=False)
    bars = ax1.bar(range(len(domain_health)), domain_health.values, 
                   color=[colors['primary_green'] if x >= 80 else 
                          colors['accent_blue'] if x >= 60 else 
                          colors['secondary_gray'] for x in domain_health.values])
    ax1.set_title('Domain Performance', fontsize=16, fontweight='bold', color=colors['dark_gray'])
    ax1.set_xlabel('Domain', fontsize=12)
    ax1.set_ylabel('Average Health Score', fontsize=12)
    ax1.set_xticks(range(len(domain_health)))
    ax1.set_xticklabels(domain_health.index, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, domain_health.values)):
        ax1.text(bar.get_x() + bar.get_width()/2., value + 1,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Field Completeness vs Total Fields (Scatter plot with trend line)
    ax2 = plt.subplot2grid((24, 4), (6, 0), colspan=4, rowspan=3)
    scatter = ax2.scatter(df['total_fields'], df['completeness_percentage'], 
                         alpha=0.7, s=80, color=colors['primary_green'])
    
    # Add trend line
    z = np.polyfit(df['total_fields'], df['completeness_percentage'], 1)
    p = np.poly1d(z)
    ax2.plot(sorted(df['total_fields']), p(sorted(df['total_fields'])), 
             color=colors['accent_blue'], linestyle='--', linewidth=2, alpha=0.8, label='Trend Line')
    
    ax2.set_title('Field Completeness vs Total Fields', fontsize=16, fontweight='bold', color=colors['dark_gray'])
    ax2.set_xlabel('Total Fields', fontsize=12)
    ax2.set_ylabel('Completeness Percentage (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    # 4. NEW: Domain Health Trends Over Time
    ax3 = plt.subplot2grid((24, 4), (10, 0), colspan=4, rowspan=4)
    trends_summary = create_domain_health_trends(df, ax3, colors)
    
    # 5. Contract Performance Heatmap
    ax4 = plt.subplot2grid((24, 4), (15, 0), colspan=4, rowspan=4)
    
    # Prepare data for heatmap - get latest version of each contract
    df_latest = df.loc[df.groupby('contract_name')['validation_timestamp'].idxmax()]
    df_sorted = df_latest.sort_values('health_score', ascending=False)
    contract_names = [name[:15] + '...' if len(name) > 15 else name for name in df_sorted['contract_name']]
    health_scores = df_sorted['health_score'].values
    
    # Create a 2D array for heatmap (reshape into grid)
    n_contracts = len(health_scores)
    n_cols = 7  # Number of columns in heatmap
    n_rows = (n_contracts + n_cols - 1) // n_cols  # Calculate rows needed
    
    # Pad with zeros if necessary
    padded_scores = np.pad(health_scores, (0, n_rows * n_cols - n_contracts), 'constant', constant_values=0)
    padded_names = contract_names + [''] * (n_rows * n_cols - n_contracts)
    
    # Reshape into grid
    heatmap_data = padded_scores.reshape(n_rows, n_cols)
    name_grid = np.array(padded_names).reshape(n_rows, n_cols)
    
    # Create heatmap
    im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Add text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            if name_grid[i, j]:  # Only add text if there's a contract name
                score = heatmap_data[i, j]
                ax4.text(j, i, f'{name_grid[i, j]}\n{score:.1f}', 
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        color='white' if score < 70 else 'black')
    
    ax4.set_title('Contract Performance Heatmap', fontsize=16, fontweight='bold', color=colors['dark_gray'])
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Health Score', fontsize=12)
    
    # 6. Priority Recommendations (updated to include time series insights)
    ax5 = plt.subplot2grid((24, 4), (20, 0), colspan=4, rowspan=3)
    ax5.axis('off')
    
    # Calculate trend-based recommendations using the trends analysis
    declining_domains = []
    improving_domains = []
    if trends_summary:
        for trend in trends_summary:
            if "Declining" in trend:
                domain_name = trend.split(':')[0]
                declining_domains.append(domain_name)
            elif "Improving" in trend:
                domain_name = trend.split(':')[0]
                improving_domains.append(domain_name)
    
    worst_contract = df.nsmallest(1, 'health_score').iloc[0]['contract_name'] if len(df) > 0 else 'Unknown'
    worst_domain = df.groupby('domain')['health_score'].mean().idxmin() if len(df) > 0 else 'unknown'
    
    recommendations = []
    
    # Critical issues first
    if description_coverage < 50:
        recommendations.append(f"🚨 Critical: Implement mandatory description standards - only {description_coverage:.1f}% coverage")
    
    # Declining trends
    if declining_domains:
        recommendations.append(f"📉 High Priority: Address declining trends in {', '.join(declining_domains[:2])} domains")
    
    # Worst performing contracts  
    if len(df) > 0:
        recommendations.append(f"⚠️ High Priority: Address {worst_contract} critical issues (lowest health score)")
    
    # Domain focus
    recommendations.append(f"🎯 Medium Priority: Focus improvement efforts on '{worst_domain}' domain (lowest average score)")
    
    # General improvements
    if field_completeness < 80:
        recommendations.append(f"📊 Medium Priority: Improve field completeness to 80%+ (currently {field_completeness:.1f}%)")
    
    # Positive reinforcement
    if improving_domains:
        recommendations.append(f"✅ Good Progress: Continue improvements in {', '.join(improving_domains[:2])} domains")
    
    ax5.text(0.5, 0.95, 'Priority Recommendations', ha='center', va='top', 
             fontsize=16, fontweight='bold', color=colors['dark_gray'])
    
    priority_colors = [colors['warning_red'], colors['secondary_gray'], colors['secondary_gray'], 
                      colors['accent_blue'], colors['accent_blue']]
    
    for i, (rec, color) in enumerate(zip(recommendations, priority_colors)):
        y_pos = 0.8 - i * 0.15
        # Add colored rectangle
        rect = plt.Rectangle((0.05, y_pos-0.05), 0.9, 0.08, 
                           facecolor=color, alpha=0.2, transform=ax5.transAxes)
        ax5.add_patch(rect)
        
        ax5.text(0.08, y_pos, rec, ha='left', va='center', fontsize=11, 
                color=colors['dark_gray'], transform=ax5.transAxes)
    
    # Add footer
    fig.text(0.5, 0.02, 'Generated by Data Quality Analysis Tool | Siemens Energy | Report Date: September 2025', 
             ha='center', fontsize=10, color=colors['secondary_gray'])
    
    # Save the report
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.05)
    plt.savefig('data_quality_report.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def create_domain_health_trends(df, ax, colors):
    """Create optimized domain-level health score trends over time with clean area charts"""
    
    # Check if we have timestamp data
    if 'validation_timestamp' not in df.columns or df['validation_timestamp'].isna().all():
        # No time series data - show message
        ax.text(0.5, 0.5, 'Time Series Data Not Available\n\nGenerate reports over multiple time periods\nto see domain health trends', 
                ha='center', va='center', fontsize=14, color=colors['secondary_gray'],
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['light_gray'], alpha=0.5))
        ax.set_title('Domain Health Score Trends Over Time', fontsize=16, fontweight='bold', color=colors['dark_gray'])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return []
    
    # Define distinct colors for each domain - all different colors
    domain_color_map = {
        'salesforce': colors['primary_green'],      # Green
        'towerdatahub': colors['accent_blue'],      # Blue
        'projectconfig': colors['secondary_gray'],   # Gray
        'zcv': colors['warning_red'],               # Red  
        'sst': '#FF6B35',                          # Orange
        'datatypes': '#F7B731',                    # Yellow
        'stp': '#8E44AD',                          # Purple
        'sap_bw': '#E67E22',                       # Dark Orange
        'towerselect': '#1ABC9C',                  # Teal
        'default': '#95A5A6',                      # Light Gray
        'unknown': '#BDC3C7'                       # Lighter Gray
    }
    
    # Create time-based aggregation for cleaner visualization
    # Group by domain and 4-hour periods to show clear changes while reducing noise
    df['time_period'] = df['validation_timestamp'].dt.floor('4H')  # Round to nearest 4 hours
    
    # Aggregate by domain and time for cleaner visualization
    domain_time_data = df.groupby(['domain', 'time_period']).agg({
        'health_score': 'mean',
        'validation_timestamp': 'mean',
        'contract_name': 'count'  # Count of reports per time period
    }).reset_index()
    
    # Rename for clarity
    domain_time_data.rename(columns={'contract_name': 'report_count'}, inplace=True)
    
    domains = domain_time_data['domain'].unique()
    trends_summary = []
    
    # Plot each domain as an area chart
    for domain in domains:
        domain_data = domain_time_data[domain_time_data['domain'] == domain].sort_values('time_period')
        
        if len(domain_data) >= 1:
            color = domain_color_map.get(domain, '#95A5A6')
            timestamps = domain_data['validation_timestamp']
            health_scores = domain_data['health_score']
            
            if len(domain_data) >= 2:
                # Create smooth area chart
                ax.fill_between(timestamps, 0, health_scores, 
                              color=color, alpha=0.4, label=f'{domain.title()}')
                
                # Add line on top for clarity
                ax.plot(timestamps, health_scores, 
                       color=color, linewidth=3, marker='o', markersize=6,
                       markerfacecolor=color, markeredgecolor='white', markeredgewidth=2)
                
                # Calculate domain trend
                first_score = health_scores.iloc[0]
                last_score = health_scores.iloc[-1]
                trend_change = last_score - first_score
                avg_score = health_scores.mean()
                
                if abs(trend_change) > 1:  # Show trends for changes > 1 point
                    trend_direction = "↗ Improving" if trend_change > 0 else "↘ Declining"
                    trends_summary.append(f"{domain.title()}: {trend_direction} ({trend_change:+.1f}pts, avg: {avg_score:.1f})")
                else:
                    trends_summary.append(f"{domain.title()}: Stable ({trend_change:+.1f}pts, avg: {avg_score:.1f})")
            else:
                # Single point - show as prominent marker
                ax.scatter(timestamps, health_scores, 
                          s=120, color=color, alpha=0.8, label=f'{domain.title()} (single)',
                          edgecolors='white', linewidth=2, zorder=5)
                avg_score = health_scores.iloc[0]
                trends_summary.append(f"{domain.title()}: Single measurement ({avg_score:.1f}pts)")
    
    # Customize the plot
    ax.set_title('Domain Health Score Trends Over Time', fontsize=16, fontweight='bold', color=colors['dark_gray'])
    ax.set_xlabel('Date & Time', fontsize=12)
    ax.set_ylabel('Health Score', fontsize=12)
    
    # Create clean legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, frameon=True, 
             fancybox=True, shadow=True)
    
    # Grid and limits
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    # Format x-axis with better date formatting to clearly show changes
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # Every 4 hours to match aggregation
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=9)
    
    # Add minor ticks for better granularity
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
    ax.tick_params(which='minor', length=3)
    
    # Add trend summary in a clean box
    if trends_summary:
        trend_text = "\n".join(trends_summary[:6])  # Limit to 6 lines for readability
        ax.text(0.02, 0.98, f"Domain Trends:\n{trend_text}", transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.4", 
               facecolor='white', edgecolor=colors['secondary_gray'], alpha=0.95))
    
    # Add overall statistics
    total_domains = len(domains)
    improving_domains = len([t for t in trends_summary if "Improving" in t])
    declining_domains = len([t for t in trends_summary if "Declining" in t])
    
    stats_text = f"Domains: {total_domains} | ↗ {improving_domains} | ↘ {declining_domains}"
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['light_gray'], alpha=0.9))
    
    return trends_summary

def export_raw_json_reports_to_csv(reports_dir, output_csv):
    """Combine all JSON report files into a single CSV with all raw fields."""
    reports_dir = Path(reports_dir)
    all_data = []
    for json_file in reports_dir.glob("*_report.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                all_data.append(data)
            elif isinstance(data, list):
                all_data.extend(data)
    if not all_data:
        print("No JSON reports found.")
        return
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"✓ Exported all raw JSON report data to: {output_csv}")

def generate_executive_excel_report(df, output_path="data_quality_executive_report.xlsx"):
    """Generate a single Excel file with all requested executive analytics and sheets."""
    import pandas as pd
    from datetime import datetime

    # --- Sheet 1: Detailed Data with Calculated Fields ---
    df = df.copy()
    # Health Score Tier
    df['Health Score Tier'] = pd.cut(
        df['health_score'],
        bins=[-float('inf'), 40, 60, 80, 100],
        labels=['Critical', 'At Risk', 'Acceptable', 'Excellent']
    )
    # Maturity Index
    df['Maturity Index'] = (
        0.4 * df['validation_score'] +
        0.3 * df['completeness_score'] +
        0.2 * df['lint_score'] +
        0.1 * df['documentation_score']
    ).round(2)
    # Risk Score
    df['Risk Score'] = (100 - df['health_score']).round(2)
    # Completeness Ratio
    df['Completeness Ratio'] = ((df['total_fields'] - df['empty_fields_count']) / df['total_fields']).replace([np.inf, -np.inf], 0).fillna(0).round(3)
    # Documentation Gap
    df['Documentation Gap'] = (100 - df['documentation_score']).round(2)
    # Quality Velocity (change vs previous period)
    df = df.sort_values(['contract_name', 'validation_timestamp'])
    df['Quality Velocity'] = df.groupby('contract_name')['health_score'].diff().fillna(0).round(2)
    # Owner Performance Score
    owner_perf = df.groupby('owner')['health_score'].mean().round(2)
    df['Owner Performance Score'] = df['owner'].map(owner_perf)
    # Schema Complexity Index
    df['Schema Complexity Index'] = (df['total_fields'] * (1 - df['Completeness Ratio'])).round(2)
    # Days Since Last Update
    today = pd.Timestamp(datetime.now().date())
    df['Days Since Last Update'] = (today - pd.to_datetime(df['validation_timestamp']).dt.normalize()).dt.days
    # Compliance Flag
    df['Compliance Flag'] = np.where(
        (df['validation_score'] > 60) &
        (df['completeness_score'] > 60) &
        (df['lint_score'] > 60) &
        (df['documentation_score'] > 60),
        "Compliant", "Non-Compliant"
    )

    # --- Sheet 2: Executive Dashboard Metrics ---
    kpi = {}
    kpi['Portfolio Health Score'] = np.average(df['health_score'], weights=None)
    kpi['Total Contracts Monitored'] = df['contract_name'].nunique()
    kpi['Critical Contracts Count (<40)'] = (df['health_score'] < 40).sum()
    kpi['Average Documentation Coverage'] = df['documentation_score'].mean().round(2)
    # Week-over-Week Improvement Rate
    if 'report_week' in df.columns:
        week_health = df.groupby('report_week')['health_score'].mean()
        kpi['Week-over-Week Improvement Rate'] = (week_health.diff().mean() if len(week_health) > 1 else 0).round(2)
    else:
        kpi['Week-over-Week Improvement Rate'] = 0
    # Top 3 Problem Areas (by domain)
    kpi['Top 3 Problem Areas'] = ', '.join(df.groupby('domain')['health_score'].mean().sort_values().head(3).index)
    # Bottom 3 Performing Schemas
    kpi['Bottom 3 Performing Schemas'] = ', '.join(df.groupby('contract_name')['health_score'].mean().sort_values().head(3).index)
    kpi_df = pd.DataFrame(list(kpi.items()), columns=['KPI', 'Value'])

    # --- Sheet 3: Time Series Analysis ---
    df['report_date'] = pd.to_datetime(df['validation_timestamp']).dt.date
    time_series = df.groupby('report_date').agg(
        avg_health_score=('health_score', 'mean'),
        count_validations=('contract_name', 'count'),
        pass_rate=('validation_passed', 'mean')
    ).reset_index()
    time_series['Rolling 7-Day Average'] = time_series['avg_health_score'].rolling(7, min_periods=1).mean().round(2)
    time_series['Month'] = pd.to_datetime(time_series['report_date']).dt.to_period('M').dt.to_timestamp()
    month_avg = time_series.groupby('Month')['avg_health_score'].mean()
    time_series['Month-over-Month Growth'] = month_avg.pct_change().reindex(time_series['Month']).values
    # Simple 3-month forecast (linear extrapolation)
    if len(time_series) >= 2:
        x = np.arange(len(time_series))
        y = time_series['avg_health_score'].values
        coef = np.polyfit(x, y, 1)
        forecast = coef[0] * (x[-1] + np.arange(1, 4)) + coef[1]
        for i, val in enumerate(forecast, 1):
            time_series[f'Forecast_Month_{i}'] = val
    else:
        for i in range(1, 4):
            time_series[f'Forecast_Month_{i}'] = np.nan

    # --- Sheet 4: Owner Performance Matrix ---
    owner_matrix = df.groupby('owner').agg(
        contracts_owned=('contract_name', 'nunique'),
        avg_health_score=('health_score', 'mean'),
        best_contract=('health_score', lambda x: df.loc[x.idxmax(), 'contract_name']),
        worst_contract=('health_score', lambda x: df.loc[x.idxmin(), 'contract_name']),
        improvement_rate=('health_score', lambda x: x.diff().mean() if len(x) > 1 else 0),
        documentation_compliance_pct=('documentation_score', lambda x: (x > 60).mean() * 100),
        action_items_count=('needs_urgent_attention', 'sum')
    ).reset_index()

    # --- Sheet 5: Schema Deep Dive ---
    schema_deep = df.groupby('domain').agg(
        contract_count=('contract_name', 'nunique'),
        avg_health_score=('health_score', 'mean'),
        avg_validation_score=('validation_score', 'mean'),
        avg_completeness_score=('completeness_score', 'mean'),
        avg_lint_score=('lint_score', 'mean'),
        avg_documentation_score=('documentation_score', 'mean'),
        avg_field_complexity=('total_fields', 'mean'),
        avg_empty_fields=('empty_fields_count', 'mean'),
        avg_completeness_ratio=('Completeness Ratio', 'mean'),
        avg_schema_complexity_index=('Schema Complexity Index', 'mean'),
        failure_pattern=('has_validation_issues', 'mean'),
        benchmark_vs_portfolio=('health_score', lambda x: x.mean() - df['health_score'].mean())
    ).reset_index()

    # --- Write to Excel ---
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Detailed Data', index=False)
        kpi_df.to_excel(writer, sheet_name='Executive Dashboard Metrics', index=False)
        time_series.to_excel(writer, sheet_name='Time Series Analysis', index=False)
        owner_matrix.to_excel(writer, sheet_name='Owner Performance Matrix', index=False)
        schema_deep.to_excel(writer, sheet_name='Schema Deep Dive', index=False)
    print(f"✓ Executive Excel report generated: {output_path}")

def main():
    """Main function to generate the report and CSV files"""
    print("="*60)
    print("🚀 Starting Data Quality Report Generation...")
    print("="*60 + "\n")
    
    # Load all reports
    reports_dir = Path(".")
    reports_data = load_all_reports(reports_dir)
    
    if not reports_data:
        print("❌ No report files found! Make sure you're in the reports directory.")
        return
    
    print(f"✓ Loaded {len(reports_data)} reports successfully!\n")
    
    # Create DataFrame
    print("📊 Processing data...")
    df = create_dataframe(reports_data)
    
    # Generate comprehensive CSV files
    print("\n📁 Generating CSV files...")
    csv_files = generate_comprehensive_csv(df)
    
    # Create comprehensive visual report
    print("\n📈 Creating visual report...")
    create_comprehensive_report(df)
    
    # Generate executive Excel report
    print("\n📊 Creating executive Excel report...")
    generate_executive_excel_report(df)
    
    print("\n" + "="*60)
    print("✅ Report Generation Complete!")
    print("="*60)
    print("\n📋 Generated Files:")
    print("  • Visual Report: data_quality_report.png")
    for file_type, file_path in csv_files.items():
        if file_path:
            print(f"  • {file_type.capitalize()} CSV: {file_path.name}")
    print("  • Executive Excel Report: data_quality_executive_report.xlsx")
    
    print("\n💡 Next Steps:")
    print("  1. Review the visual report (data_quality_report.png)")
    print("  2. Import CSV files into your preferred analysis tool")
    print("  3. Use the index CSV to understand each file's purpose")
    print("  4. Focus on contracts marked as 'needs_urgent_attention'")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()