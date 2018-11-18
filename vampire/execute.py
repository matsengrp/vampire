"""
Execute commands locally or via sbatch.
"""

import click
import os
import subprocess


sbatch_prefix = """
#!/bin/bash
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p campus
#SBATCH --mem=31000
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matsen@fredhutch.org
hostname
source activate py36
cd /home/matsen/re/vampire/vampire/
"""


@click.command()
@click.option('--clusters', default='', help='Clusters to submit to. Default is local execution.')
@click.argument('sources')
@click.argument('targets')
@click.argument('to_execute_f_string')
def cli(clusters, sources, targets, to_execute_f_string):
    """
    Execute a command with certain sources and targets.
    """

    to_execute = to_execute_f_string.format(sources=sources, targets=targets)

    if clusters == '':
        click.echo("Executing locally:")
        click.echo(to_execute)
        return subprocess.check_output(to_execute, shell=True)

    # Put the batch script in the directory of the first target.
    execution_dir = os.path.dirname(targets.split()[0])
    script_name = 'job.sh'
    with open(os.path.join(execution_dir, script_name), 'w') as fp:
        fp.write(sbatch_prefix)
        fp.write(to_execute+'\n')

    return subprocess.check_output(f'cd {execution_dir}; sbatch --clusters {clusters} {script_name}', shell=True)


if __name__ == '__main__':
    cli()
