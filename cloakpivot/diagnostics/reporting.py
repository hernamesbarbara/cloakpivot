"""Diagnostic report generation for CloakPivot operations."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Union

from .collector import MaskingStatistics
from .coverage import CoverageMetrics


class ReportFormat(Enum):
    """Supported diagnostic report formats."""
    
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class ReportData:
    """
    Comprehensive data structure for diagnostic reports.
    
    Contains all information needed to generate diagnostic reports
    in various formats including statistics, coverage, performance,
    and recommendations.
    """
    
    statistics: MaskingStatistics
    coverage: CoverageMetrics
    performance: dict[str, Any]
    diagnostics: dict[str, Any]
    document_metadata: dict[str, Any]
    recommendations: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "statistics": self.statistics.to_dict(),
            "coverage": self.coverage.to_dict(),
            "performance": self.performance,
            "diagnostics": self.diagnostics,
            "document_metadata": self.document_metadata,
            "recommendations": self.recommendations,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class DiagnosticReporter:
    """
    Generates comprehensive diagnostic reports from masking operations.
    
    Creates detailed reports in multiple formats (JSON, HTML, Markdown)
    that provide insights into masking effectiveness, performance metrics,
    coverage analysis, and optimization recommendations.
    """
    
    def __init__(self) -> None:
        """Initialize the diagnostic reporter."""
        pass
    
    def generate_report(
        self,
        data: ReportData,
        format: ReportFormat
    ) -> str:
        """
        Generate a diagnostic report in the specified format.
        
        Args:
            data: Report data to include
            format: Output format for the report
            
        Returns:
            Formatted report as string
        """
        if format == ReportFormat.JSON:
            return self._generate_json_report(data)
        elif format == ReportFormat.HTML:
            return self._generate_html_report(data)
        elif format == ReportFormat.MARKDOWN:
            return self._generate_markdown_report(data)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def save_report(
        self,
        data: ReportData,
        output_path: Union[str, Path],
        format: ReportFormat
    ) -> None:
        """
        Generate and save a diagnostic report to file.
        
        Args:
            data: Report data to include
            output_path: Path to save the report
            format: Output format for the report
        """
        report_content = self.generate_report(data, format)
        
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def generate_summary(self, data: ReportData) -> dict[str, Any]:
        """
        Generate a concise summary of the diagnostic report.
        
        Args:
            data: Report data to summarize
            
        Returns:
            Dictionary with key summary metrics
        """
        return {
            "document": {
                "name": data.document_metadata.get("name", "Unknown"),
                "size_bytes": data.document_metadata.get("size_bytes", 0)
            },
            "entities": {
                "detected": data.statistics.total_entities_detected,
                "masked": data.statistics.total_entities_masked,
                "success_rate": data.statistics.masking_success_rate
            },
            "coverage": {
                "rate": data.coverage.overall_coverage_rate,
                "segments_covered": data.coverage.segments_with_entities,
                "total_segments": data.coverage.total_segments
            },
            "performance": {
                "total_time_seconds": data.performance.get("total_time_seconds", 0),
                "throughput": data.performance.get("throughput_entities_per_second", 0)
            },
            "issues": {
                "has_warnings": data.diagnostics.get("warning_count", 0) > 0,
                "has_errors": data.diagnostics.get("error_count", 0) > 0,
                "total_issues": data.diagnostics.get("warning_count", 0) + data.diagnostics.get("error_count", 0)
            }
        }
    
    def _generate_json_report(self, data: ReportData) -> str:
        """Generate JSON format report."""
        report_dict = data.to_dict()
        return json.dumps(report_dict, indent=2, ensure_ascii=False)
    
    def _generate_html_report(self, data: ReportData) -> str:
        """Generate HTML format report with visualizations."""
        summary = self.generate_summary(data)
        
        # Create entity distribution data for charts
        entity_labels = list(data.statistics.entity_counts_by_type.keys())
        entity_counts = list(data.statistics.entity_counts_by_type.values())
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CloakPivot Diagnostic Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .chart-container {{ width: 400px; height: 400px; margin: 20px auto; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        .recommendations {{ background: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #1e88e5; }}
        .warning {{ background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 10px 0; }}
        .error {{ background: #f8d7da; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>CloakPivot Diagnostic Report</h1>
            <p>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <p>Document: {summary['document']['name']}</p>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="metric-value">{summary['entities']['detected']}</div>
                <div class="metric-label">Entities Detected</div>
            </div>
            <div class="summary-card">
                <div class="metric-value">{summary['entities']['masked']}</div>
                <div class="metric-label">Entities Masked</div>
            </div>
            <div class="summary-card">
                <div class="metric-value">{summary['coverage']['rate']:.1%}</div>
                <div class="metric-label">Coverage Rate</div>
            </div>
            <div class="summary-card">
                <div class="metric-value">{summary['performance']['total_time_seconds']:.1f}s</div>
                <div class="metric-label">Total Time</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Entity Distribution</h2>
            <div class="chart-container">
                <canvas id="entity-distribution-chart"></canvas>
            </div>
        </div>
        
        <div class="section">
            <h2>Coverage Analysis</h2>
            <table>
                <tr>
                    <th>Section Type</th>
                    <th>Total Segments</th>
                    <th>Covered Segments</th>
                    <th>Coverage Rate</th>
                    <th>Entities</th>
                </tr>"""
        
        for section in data.coverage.section_coverage:
            html_content += f"""
                <tr>
                    <td>{section.section_type.title()}</td>
                    <td>{section.total_segments}</td>
                    <td>{section.segments_with_entities}</td>
                    <td>{section.coverage_rate:.1%}</td>
                    <td>{section.entity_count}</td>
                </tr>"""
        
        html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Processing Time</td>
                    <td>{data.performance.get('total_time_seconds', 0):.2f} seconds</td>
                </tr>
                <tr>
                    <td>Entity Detection Time</td>
                    <td>{data.performance.get('detection_time_seconds', 0):.2f} seconds</td>
                </tr>
                <tr>
                    <td>Masking Time</td>
                    <td>{data.performance.get('masking_time_seconds', 0):.2f} seconds</td>
                </tr>
                <tr>
                    <td>Throughput</td>
                    <td>{data.performance.get('throughput_entities_per_second', 0):.2f} entities/sec</td>
                </tr>
            </table>
        </div>"""
        
        # Add diagnostics section if there are issues
        if data.diagnostics.get("warning_count", 0) > 0 or data.diagnostics.get("error_count", 0) > 0:
            html_content += f"""
        <div class="section">
            <h2>Diagnostics</h2>"""
            
            for warning in data.diagnostics.get("warnings", []):
                html_content += f'<div class="warning"><strong>Warning:</strong> {warning}</div>'
            
            for error in data.diagnostics.get("errors", []):
                html_content += f'<div class="error"><strong>Error:</strong> {error}</div>'
            
            html_content += "</div>"
        
        # Add recommendations
        if data.recommendations:
            html_content += f"""
        <div class="section">
            <h2>Recommendations</h2>
            <div class="recommendations">
                <ul>"""
            
            for rec in data.recommendations:
                html_content += f"<li>{rec}</li>"
            
            html_content += """
                </ul>
            </div>
        </div>"""
        
        # Add chart JavaScript
        html_content += f"""
    </div>
    
    <script>
        // Entity Distribution Chart
        const ctx = document.getElementById('entity-distribution-chart').getContext('2d');
        new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(entity_labels)},
                datasets: [{{
                    data: {json.dumps(entity_counts)},
                    backgroundColor: [
                        '#FF6384',
                        '#36A2EB', 
                        '#FFCE56',
                        '#4BC0C0',
                        '#9966FF',
                        '#FF9F40'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }},
                    title: {{
                        display: true,
                        text: 'Entity Type Distribution'
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        return html_content
    
    def _generate_markdown_report(self, data: ReportData) -> str:
        """Generate Markdown format report."""
        summary = self.generate_summary(data)
        
        md_content = f"""# CloakPivot Diagnostic Report

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Document:** {summary['document']['name']}

## Summary

- **{summary['entities']['detected']} entities detected**, {summary['entities']['masked']} masked ({summary['entities']['success_rate']:.1%} success rate)
- **Coverage:** {summary['coverage']['rate']:.1%} of document segments contain detected PII
- **Processing Time:** {summary['performance']['total_time_seconds']:.2f} seconds
- **Throughput:** {summary['performance']['throughput']:.2f} entities/second

## Statistics

### Entity Counts by Type
"""
        
        for entity_type, count in data.statistics.entity_counts_by_type.items():
            md_content += f"- **{entity_type}:** {count}\n"
        
        md_content += f"""
### Strategy Usage
"""
        
        for strategy, count in data.statistics.strategy_usage.items():
            md_content += f"- **{strategy}:** {count} entities\n"
        
        md_content += f"""
## Coverage Analysis

- **Overall Coverage:** {data.coverage.overall_coverage_rate:.1%}
- **Segments with Entities:** {data.coverage.segments_with_entities}/{data.coverage.total_segments}
- **Entity Density:** {data.coverage.entity_density:.2f} entities per segment

### Section Breakdown
"""
        
        for section in data.coverage.section_coverage:
            md_content += f"- **{section.section_type.title()}:** {section.coverage_rate:.1%} coverage ({section.segments_with_entities}/{section.total_segments} segments, {section.entity_count} entities)\n"
        
        if data.coverage.coverage_gaps:
            md_content += f"""
### Coverage Gaps
{len(data.coverage.coverage_gaps)} segments without detected entities:
"""
            for gap in data.coverage.coverage_gaps[:5]:  # Show first 5 gaps
                md_content += f"- `{gap['node_id']}` ({gap['type']})\n"
            
            if len(data.coverage.coverage_gaps) > 5:
                md_content += f"- ... and {len(data.coverage.coverage_gaps) - 5} more\n"
        
        md_content += f"""
## Performance

- **Total Time:** {data.performance.get('total_time_seconds', 0):.2f} seconds
- **Detection Time:** {data.performance.get('detection_time_seconds', 0):.2f} seconds
- **Masking Time:** {data.performance.get('masking_time_seconds', 0):.2f} seconds
- **Serialization Time:** {data.performance.get('serialization_time_seconds', 0):.2f} seconds
"""
        
        # Add diagnostics if present
        if data.diagnostics.get("warning_count", 0) > 0 or data.diagnostics.get("error_count", 0) > 0:
            md_content += f"""
## Diagnostics

"""
            
            if data.diagnostics.get("warnings"):
                md_content += "### Warnings\n"
                for warning in data.diagnostics["warnings"]:
                    md_content += f"- ⚠️ {warning}\n"
            
            if data.diagnostics.get("errors"):
                md_content += "### Errors\n"
                for error in data.diagnostics["errors"]:
                    md_content += f"- ❌ {error}\n"
        
        # Add recommendations
        if data.recommendations:
            md_content += f"""
## Recommendations

"""
            for rec in data.recommendations:
                md_content += f"- 💡 {rec}\n"
        
        return md_content
    
    def _generate_recommendations(self, data: ReportData) -> list[str]:
        """Generate optimization recommendations based on report data."""
        recommendations = data.recommendations.copy()
        
        # Add automatic recommendations based on metrics
        if data.statistics.masking_success_rate < 0.8:
            recommendations.append("Review failed entity masking - success rate below 80%")
        
        if data.coverage.overall_coverage_rate < 0.5:
            recommendations.append("Consider tuning detection policies - low coverage detected")
        
        if data.performance.get("total_time_seconds", 0) > 10:
            recommendations.append("Performance optimization recommended - processing time exceeds 10 seconds")
        
        if data.diagnostics.get("error_count", 0) > 0:
            recommendations.append("Resolve processing errors to improve reliability")
        
        return recommendations