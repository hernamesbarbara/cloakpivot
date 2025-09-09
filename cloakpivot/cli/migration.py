"""CLI commands for migration utilities."""

import shutil
from pathlib import Path
from typing import Optional

import click

from ..migration import CloakMapMigrator, StrategyMigrator


@click.group()
def migrate() -> None:
    """CloakMap migration utilities."""
    pass


@migrate.command(name='upgrade-cloakmap')
@click.argument('cloakmap_path',
                type=click.Path(exists=True, path_type=Path))
@click.option('--output', type=click.Path(path_type=Path),
              help='Output path for migrated CloakMap')
@click.option('--backup/--no-backup', default=False,
              help='Create backup of original')
def upgrade_cloakmap(cloakmap_path: Path, output: Optional[Path],
                     backup: bool) -> None:
    """Migrate a single CloakMap to v2.0 format.

    Args:
        cloakmap_path: Path to the CloakMap file to migrate
        output: Optional output path for migrated file
        backup: Whether to create a backup of the original
    """
    migrator = CloakMapMigrator()
    
    # Check if already v2.0
    from ..core.cloakmap import CloakMap
    try:
        cloakmap = CloakMap.load_from_file(cloakmap_path)
        if cloakmap.version == "2.0":
            click.echo(f"CloakMap {cloakmap_path} already v2.0 format")
            return
    except Exception:
        pass  # Will be handled by migrator

    # Create backup if requested
    if backup:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = cloakmap_path.parent / f"{cloakmap_path.stem}.backup.{timestamp}.json"
        try:
            shutil.copy2(cloakmap_path, backup_path)
            click.echo(f"Backup created: {backup_path}")
        except Exception as e:
            click.echo(f"⚠️  Warning: Failed to create backup: {e}",
                       err=True)

    # Perform migration (in-place if no output specified)
    try:
        if not output:
            output = cloakmap_path  # In-place migration
        output_path = migrator.migrate_cloakmap(cloakmap_path, output)
        click.echo(f"CloakMap {cloakmap_path} upgraded successfully")
    except Exception as e:
        click.echo(f"❌ Migration failed: {e}", err=True)
        raise click.ClickException("Migration failed")


@migrate.command()
@click.argument('directory',
                type=click.Path(exists=True, file_okay=False,
                                path_type=Path))
@click.option('--pattern', default='*.cloakmap',
              help='File pattern to match')
@click.option('--dry-run', is_flag=True,
              help='Show what would be migrated without doing it')
def bulk(directory: Path, pattern: str, dry_run: bool) -> None:
    """Bulk migrate all CloakMaps in a directory.

    Args:
        directory: Directory containing CloakMap files
        pattern: Glob pattern to match files
        dry_run: Whether to perform a dry run
    """
    migrator = CloakMapMigrator()
    if dry_run:
        files = list(directory.glob(pattern))
        click.echo(f"Would migrate {len(files)} files:")
        for file in files:
            click.echo(f"  {file}")
        return

    # Perform bulk migration
    try:
        results = migrator.bulk_migrate(directory, pattern)
    except Exception as e:
        click.echo(f"❌ Bulk migration failed: {e}", err=True)
        raise click.ClickException("Bulk migration failed")

    # Display results
    click.echo(f"✅ Migrated: {len(results['migrated'])}")
    click.echo(f"⚠️  Skipped: {len(results['skipped'])}")
    click.echo(f"❌ Errors: {len(results['errors'])}")

    if results['migrated']:
        click.echo("\nMigrated files:")
        for item in results['migrated']:
            click.echo(f"  {item['source']} → {item['target']}")

    if results['skipped']:
        click.echo("\nSkipped files:")
        for item in results['skipped']:
            click.echo(f"  {item['file']}: {item['reason']}")

    if results['errors']:
        click.echo("\nErrors:", err=True)
        for error in results['errors']:
            click.echo(f"  {error['file']}: {error['error']}", err=True)


@migrate.command()
@click.argument('policy_file',
                type=click.Path(exists=True, path_type=Path))
@click.option('--backup/--no-backup', default=True,
              help='Create backup of original policy')
def policy(policy_file: Path, backup: bool) -> None:
    """Migrate policy file to Presidio-optimized format.

    Args:
        policy_file: Path to the Policy file to migrate
        backup: Whether to create a backup
    """
    migrator = StrategyMigrator()
    # Create backup if requested
    if backup:
        backup_path = policy_file.with_suffix('.backup.yml')
        try:
            shutil.copy2(policy_file, backup_path)
            click.echo(f"Backup created: {backup_path}")
        except Exception as e:
            click.echo(f"⚠️  Warning: Failed to create backup: {e}",
                       err=True)

    try:
        output_path = migrator.migrate_policy_file(policy_file)
        click.echo(f"✅ Policy migrated: {policy_file} → {output_path}")
    except Exception as e:
        click.echo(f"❌ Policy migration failed: {e}", err=True)
        raise click.ClickException("Policy migration failed")


@migrate.command()
@click.argument('directory',
                type=click.Path(exists=True, file_okay=False,
                                path_type=Path))
@click.option('--pattern', default='*.yml',
              help='File pattern to match policy files')
@click.option('--dry-run', is_flag=True,
              help='Show what would be migrated without doing it')
def bulk_policies(directory: Path, pattern: str, dry_run: bool) -> None:
    """Bulk migrate all policy files in a directory.

    Args:
        directory: Directory containing policy files
        pattern: Glob pattern to match policy files
        dry_run: Whether to perform a dry run
    """
    migrator = StrategyMigrator()
    if dry_run:
        files = [f for f in directory.glob(pattern)
                 if not f.name.endswith('.presidio.yml')]
        click.echo(f"Would migrate {len(files)} policy files:")
        for file in files:
            click.echo(f"  {file}")
        return

    # Perform bulk migration
    try:
        results = migrator.bulk_migrate_policies(directory, pattern)
    except Exception as e:
        click.echo(f"❌ Bulk policy migration failed: {e}", err=True)
        raise click.ClickException("Bulk policy migration failed")

    # Display results
    click.echo(f"✅ Migrated: {len(results['migrated'])}")
    click.echo(f"⚠️  Skipped: {len(results['skipped'])}")
    click.echo(f"❌ Errors: {len(results['errors'])}")

    if results['migrated']:
        click.echo("\nMigrated policies:")
        for item in results['migrated']:
            click.echo(f"  {item['source']} → {item['target']}")

    if results['skipped']:
        click.echo("\nSkipped policies:")
        for item in results['skipped']:
            click.echo(f"  {item['file']}: {item['reason']}")

    if results['errors']:
        click.echo("\nErrors:", err=True)
        for error in results['errors']:
            click.echo(f"  {error['file']}: {error['error']}", err=True)


@migrate.command()
def status() -> None:
    """Show migration timeline and deprecation status."""
    from ..migration import DeprecationManager

    click.echo("=== CloakPivot Migration Timeline ===\n")

    timeline = DeprecationManager.get_deprecation_timeline()
    for quarter, milestone in timeline.items():
        click.echo(f"  {quarter}: {milestone}")

    click.echo("\n=== Component Deprecation Status ===\n")

    components = [
        "legacy_engine",
        "strategy_applicator",
        "v1_cloakmap",
        "manual_entity_detection"
    ]

    for component in components:
        status = DeprecationManager.check_deprecation_status(component)
        if status:
            click.echo(f"  ⚠️  {component}: {status}")
        else:
            click.echo(f"  ✅ {component}: Not deprecated")

    click.echo("\n=== Migration Resources ===\n")
    click.echo("  📚 Documentation: "
               "https://cloakpivot.readthedocs.io/migration/")
    click.echo("  🐛 Report Issues: "
               "https://github.com/cloakpivot/issues")
    click.echo("  💬 Community: https://cloakpivot.slack.com")


@migrate.command()
@click.option('--check-only', is_flag=True,
              help='Only check for legacy usage, do not migrate')
def check_legacy(check_only: bool) -> None:
    """Check for legacy engine usage and v1.0 CloakMaps.

    Args:
        check_only: Whether to only check without offering migration
    """
    import os
    # Check environment variable
    use_presidio = os.getenv("CLOAKPIVOT_USE_PRESIDIO_ENGINE",
                              "").lower() in ("true", "1", "yes", "on")

    click.echo("=== Legacy Usage Check ===\n")

    if use_presidio:
        click.echo("✅ Presidio engine is enabled (via environment variable)")
    else:
        click.echo("⚠️  Legacy engine is active")
        click.echo("   Set CLOAKPIVOT_USE_PRESIDIO_ENGINE=true to use "
                   "Presidio engine")

    # Check for v1.0 CloakMaps in current directory
    click.echo("\n=== CloakMap Version Check ===\n")

    cloakmap_files = list(Path.cwd().glob("**/*.cloakmap"))
    v1_count = 0
    v2_count = 0

    if cloakmap_files:
        from ..core.cloakmap import CloakMap

        for file in cloakmap_files:
            try:
                cm = CloakMap.load_from_file(file)
                if cm.version == "1.0":
                    v1_count += 1
                    click.echo(f"  ⚠️  v1.0: {file}")
                else:
                    v2_count += 1
            except Exception:
                pass

        click.echo(f"\nFound {len(cloakmap_files)} CloakMap files:")
        click.echo(f"  v1.0 (legacy): {v1_count}")
        click.echo(f"  v2.0 (Presidio): {v2_count}")

        if v1_count > 0 and not check_only:
            if click.confirm(f"\nMigrate {v1_count} v1.0 CloakMaps to v2.0?"):
                migrator = CloakMapMigrator()
                results = migrator.bulk_migrate(Path.cwd(), "**/*.cloakmap")
                click.echo(f"✅ Migrated {len(results['migrated'])} files")
    else:
        click.echo("No CloakMap files found in current directory")